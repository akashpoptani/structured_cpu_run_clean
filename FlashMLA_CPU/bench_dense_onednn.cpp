#include <dnnl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <omp.h>
#include <immintrin.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <dnnl.hpp>

using namespace dnnl;
using bf16 = uint16_t;
using fp32 = float;

// --- HELPERS ---
fp32 bf16_to_fp32(bf16 val) {
    uint32_t bits = static_cast<uint32_t>(val) << 16;
    fp32 res;
    std::memcpy(&res, &bits, sizeof(res));
    return res;
}
// --- FLATTEN FUNCTION ---
std::vector<bf16> flatten_kv_cache(
    const bf16* K_paged, const int32_t* BlockTable, const int32_t* SeqLens,
    int B, int D, int BlockSize, int max_blocks_per_batch, size_t& out_max_seq_len
) {
    if (BlockSize <= 0) BlockSize = 64;
    int max_len = 0;
    for(int b=0; b<B; b++) if(SeqLens[b] > max_len) max_len = SeqLens[b];
    
    // Align to 16 for AMX tiling
    size_t max_len_aligned = ((size_t)max_len + 16 - 1) / 16 * 16;
    if (max_len_aligned == 0) max_len_aligned = 16;
    out_max_seq_len = max_len_aligned;

    std::vector<bf16> flat_kv((size_t)B * max_len_aligned * D, 0);

    #pragma omp parallel for schedule(dynamic)
    for (int b = 0; b < B; b++) {
        int seq_len = SeqLens[b];
        int num_blocks = (seq_len + BlockSize - 1) / BlockSize;
        bf16* batch_dst_start = flat_kv.data() + ((size_t)b * max_len_aligned * D);
        for (int i = 0; i < num_blocks; i++) {
            int block_id = BlockTable[b * max_blocks_per_batch + i];
            bf16* dst_block = batch_dst_start + ((size_t)i * BlockSize * D);
            if (block_id >= 0) {
                const bf16* src_block = K_paged + ((size_t)block_id * BlockSize * D);
                // Calculate how many tokens are actually left to copy in this block
                int tokens_to_copy = std::min(BlockSize, seq_len - (i * BlockSize));
                if (tokens_to_copy > 0) {
                    std::memcpy(dst_block, src_block, (size_t)tokens_to_copy * D * sizeof(bf16));
                }
            }
        }
    }
    return flat_kv;
}

// --- FILE IO ---
std::vector<char> read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) exit(1);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    return buffer;
}

struct Config { int B, H, D, BlockSize; float scale; };

struct BatchOps {
    matmul qk_p;
    eltwise_forward scale_p;
    softmax_forward softmax_p;
    matmul pv_p;
    
    memory q_mem, k_mem, s_mem, p_mem, v_mem, out_mem;
    std::unordered_map<int, memory> qk_args, scale_args, softmax_args, pv_args;
};

struct DNNLContextMacro {
    engine eng;
    stream strm;
    std::vector<BatchOps> batch_ops;
    
    // Intermediate global memory buffers (allocated on heap)
    std::vector<float> scores_buffer;
    std::vector<bf16> probs_buffer;

    DNNLContextMacro() : eng(engine::kind::cpu, 0), strm(eng) {}

    void init(const bf16* Q, const bf16* KV_flat, bf16* Out, const int32_t* SeqLens, 
              int B, int H, int D, int Dv, float scale, size_t max_seq_len) {
        
        // Allocate intermediate buffers for S (float32) and P (bf16)
        scores_buffer.resize((size_t)B * H * max_seq_len, 0.0f);
        probs_buffer.resize((size_t)B * H * max_seq_len, 0);

        primitive_attr attr_bf16;
        attr_bf16.set_fpmath_mode(fpmath_mode::bf16);

        for (int b = 0; b < B; ++b) {
            BatchOps ops;
            int seq_len = SeqLens[b];
            
            // --- 1. SIMPLE 2D LOGICAL SHAPES ---
            memory::dims q_dims = {H, D};                      // 128 x 576
            memory::dims k_dims = {D, seq_len};                // 576 x seq_len
            memory::dims s_dims = {H, seq_len};                // 128 x seq_len
            memory::dims p_dims = {H, seq_len};                // 128 x seq_len
            memory::dims v_dims = {seq_len, Dv};               // seq_len x 512
            memory::dims out_dims = {H, Dv};                   // 128 x 512

            // --- 2. SIMPLE STRIDES ---
            memory::dims q_strides = {D, 1};                   // Standard contiguous
            
            // K's physical memory is [seq_len, D]. We want it logically as [D, seq_len].
            // To "transpose" it for free, we swap the strides!
            memory::dims k_strides = {1, D};                   
            
            memory::dims s_strides = {seq_len, 1};             // Standard contiguous
            memory::dims p_strides = {seq_len, 1};             // Standard contiguous
            
            // V's physical memory is [seq_len, D] (576 wide). We only want Dv (512) columns.
            memory::dims v_strides = {D, 1};                   
            memory::dims out_strides = {Dv, 1};                // Standard contiguous

            // --- 3. CREATE DESCRIPTORS ---
            auto q_md = memory::desc(q_dims, memory::data_type::bf16, q_strides);
            auto k_md = memory::desc(k_dims, memory::data_type::bf16, k_strides);
            auto s_md = memory::desc(s_dims, memory::data_type::f32, s_strides);
            // 🔥 DATA TYPE CHANGE: p_md is now bf16 🔥
            auto p_md = memory::desc(p_dims, memory::data_type::bf16, p_strides);
            auto v_md = memory::desc(v_dims, memory::data_type::bf16, v_strides);
            auto out_md = memory::desc(out_dims, memory::data_type::bf16, out_strides);
            // 2. Create Primitives
            ops.qk_p = matmul(matmul::primitive_desc(eng, q_md, k_md, s_md, attr_bf16));
            
            // Scale primitive (Linear Eltwise: y = alpha * x + beta)
            ops.scale_p = eltwise_forward(eltwise_forward::primitive_desc(
                eng, prop_kind::forward_inference, algorithm::eltwise_linear, s_md, s_md, scale, 0.0f));
            
            // Softmax along axis 2 (seq_len)
            // 🔥 FIXED AXIS: Softmax along axis 1 (the seq_len dimension) 🔥
            ops.softmax_p = softmax_forward(softmax_forward::primitive_desc(
                eng, prop_kind::forward_inference, algorithm::softmax_accurate, s_md, p_md, 1));            

            ops.pv_p = matmul(matmul::primitive_desc(eng, p_md, v_md, out_md, attr_bf16));

            // 3. Bind Memory Pointers
            const bf16* q_ptr = Q + ((size_t)b * H * D);
            const bf16* kv_ptr = KV_flat + ((size_t)b * max_seq_len * D);
            float* s_ptr = scores_buffer.data() + ((size_t)b * H * max_seq_len);
            bf16* p_ptr = probs_buffer.data() + ((size_t)b * H * max_seq_len);
            bf16* out_ptr = Out + ((size_t)b * H * Dv);

            ops.q_mem = memory(q_md, eng, (void*)q_ptr);
            ops.k_mem = memory(k_md, eng, (void*)kv_ptr);
            ops.s_mem = memory(s_md, eng, (void*)s_ptr);
            ops.p_mem = memory(p_md, eng, (void*)p_ptr);
            ops.v_mem = memory(v_md, eng, (void*)kv_ptr); // V uses the same physical KV buffer
            ops.out_mem = memory(out_md, eng, (void*)out_ptr);

            // 4. Pre-build execution arguments
            ops.qk_args = {{DNNL_ARG_SRC, ops.q_mem}, {DNNL_ARG_WEIGHTS, ops.k_mem}, {DNNL_ARG_DST, ops.s_mem}};
            ops.scale_args = {{DNNL_ARG_SRC, ops.s_mem}, {DNNL_ARG_DST, ops.s_mem}}; // In-place scaling
            ops.softmax_args = {{DNNL_ARG_SRC, ops.s_mem}, {DNNL_ARG_DST, ops.p_mem}}; // Softmax auto-casts f32->bf16
            ops.pv_args = {{DNNL_ARG_SRC, ops.p_mem}, {DNNL_ARG_WEIGHTS, ops.v_mem}, {DNNL_ARG_DST, ops.out_mem}};

            batch_ops.push_back(ops);
        }
    }
};

void dense_attention_onednn_no_custom_tiling(DNNLContextMacro& ctx, int B) {
    // Look at how clean this is! No OpenMP, no chunking, no intrinsics.
    // oneDNN automatically spawns threads across the 128 Heads.
    // Structures to store cumulative times (in milliseconds)
    double qk_total = 0, scale_total = 0, softmax_total = 0, pv_total = 0;

    auto start_all = std::chrono::high_resolution_clock::now();

    for (int b = 0; b < B; ++b) {
        auto& ops = ctx.batch_ops[b];

        // 1. Q * K^T
        auto t1 = std::chrono::high_resolution_clock::now();
        ops.qk_p.execute(ctx.strm, ops.qk_args);
        // ctx.strm.wait(); // Ensure completion for accurate timing
        auto t2 = std::chrono::high_resolution_clock::now();
        qk_total += std::chrono::duration<double, std::milli>(t2 - t1).count();

        // 2. Scores * Scale
        auto t3 = std::chrono::high_resolution_clock::now();
        ops.scale_p.execute(ctx.strm, ops.scale_args);
        // ctx.strm.wait();
        auto t4 = std::chrono::high_resolution_clock::now();
        scale_total += std::chrono::duration<double, std::milli>(t4 - t3).count();

        // 3. Softmax
        auto t5 = std::chrono::high_resolution_clock::now();
        ops.softmax_p.execute(ctx.strm, ops.softmax_args);
        // ctx.strm.wait();
        auto t6 = std::chrono::high_resolution_clock::now();
        softmax_total += std::chrono::duration<double, std::milli>(t6 - t5).count();

        // 4. Probs * V
        auto t7 = std::chrono::high_resolution_clock::now();
        ops.pv_p.execute(ctx.strm, ops.pv_args);
        // ctx.strm.wait();
        auto t8 = std::chrono::high_resolution_clock::now();
        pv_total += std::chrono::duration<double, std::milli>(t8 - t7).count();
    }

    auto end_all = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end_all - start_all).count();

    // Output results
    printf("\n--- Breakdown (Total for %d batches) ---\n", B);
    printf("1. Q*K^T:   %10.4f ms\n", qk_total);
    printf("2. Scale:   %10.4f ms\n", scale_total);
    printf("3. Softmax: %10.4f ms\n", softmax_total);
    printf("4. Probs*V: %10.4f ms\n", pv_total);
    printf("---------------------------------------\n");
    printf("Total Loop Time: %10.4f ms\n", total_ms);
    ctx.strm.wait();
}

void dense_attention_onednn_no_custom_tiling_no_timers(DNNLContextMacro& ctx, int B) {
    // Look at how clean this is! No OpenMP, no chunking, no intrinsics.
    // oneDNN automatically spawns threads across the 128 Heads.
    // Structures to store cumulative times (in milliseconds)
    // double qk_total = 0, scale_total = 0, softmax_total = 0, pv_total = 0;

    // auto start_all = std::chrono::high_resolution_clock::now();

    for (int b = 0; b < B; ++b) {
        auto& ops = ctx.batch_ops[b];

        // 1. Q * K^T
        // auto t1 = std::chrono::high_resolution_clock::now();
        ops.qk_p.execute(ctx.strm, ops.qk_args);
        // ctx.strm.wait(); // Ensure completion for accurate timing
        // auto t2 = std::chrono::high_resolution_clock::now();
        // qk_total += std::chrono::duration<double, std::milli>(t2 - t1).count();

        // 2. Scores * Scale
        // auto t3 = std::chrono::high_resolution_clock::now();
        ops.scale_p.execute(ctx.strm, ops.scale_args);
        // ctx.strm.wait();
        // auto t4 = std::chrono::high_resolution_clock::now();
        // scale_total += std::chrono::duration<double, std::milli>(t4 - t3).count();

        // 3. Softmax
        // auto t5 = std::chrono::high_resolution_clock::now();
        ops.softmax_p.execute(ctx.strm, ops.softmax_args);
        // ctx.strm.wait();
        // auto t6 = std::chrono::high_resolution_clock::now();
        // softmax_total += std::chrono::duration<double, std::milli>(t6 - t5).count();

        // 4. Probs * V
        // auto t7 = std::chrono::high_resolution_clock::now();
        ops.pv_p.execute(ctx.strm, ops.pv_args);
        // ctx.strm.wait();
        // auto t8 = std::chrono::high_resolution_clock::now();
        // pv_total += std::chrono::duration<double, std::milli>(t8 - t7).count();
    }

    // auto end_all = std::chrono::high_resolution_clock::now();
    // double total_ms = std::chrono::duration<double, std::milli>(end_all - start_all).count();

    // // Output results
    // printf("\n--- Breakdown (Total for %d batches) ---\n", B);
    // printf("1. Q*K^T:   %10.4f ms\n", qk_total);
    // printf("2. Scale:   %10.4f ms\n", scale_total);
    // printf("3. Softmax: %10.4f ms\n", softmax_total);
    // printf("4. Probs*V: %10.4f ms\n", pv_total);
    // printf("---------------------------------------\n");
    // printf("Total Loop Time: %10.4f ms\n", total_ms);
    ctx.strm.wait();
}

int main() {
    // 1. Force oneDNN to respect OpenMP and not oversubscribe
    // setenv("DNNL_MAX_CONCURRENCY", "1", 1);
    // ⚠️ WE LET ONEDNN USE ALL CORES ⚠️
    // unsetenv("DNNL_MAX_CONCURRENCY");
    std::cout << "⚡ LOADING DATA..." << std::endl;
    auto cfg_data = read_file("bins/config.bin");
    Config* cfg = reinterpret_cast<Config*>(cfg_data.data());
    std::cout << "Batch Size (B): " << cfg->B << std::endl;
    int max_threads = omp_get_max_threads();
    std::cout << "OpenMP max threads: " << max_threads << std::endl;
    auto q_bin = read_file("bins/q.bin");
    auto k_bin = read_file("bins/k_cache.bin");
    auto bt_bin = read_file("bins/block_table.bin");
    auto sl_bin = read_file("bins/seqlens.bin");

    bf16* Q = reinterpret_cast<bf16*>(q_bin.data());
    bf16* K_paged = reinterpret_cast<bf16*>(k_bin.data());
    // std::cout << "Q[0]: " << static_cast<float>(Q[0]) << std::endl;
    int32_t* BT = reinterpret_cast<int32_t*>(bt_bin.data());
    // std::cout << "BT[0]: " << BT[0] << std::endl;
    // std::cout << "BT[1]: " << BT[1] << std::endl;
    int32_t* SL = reinterpret_cast<int32_t*>(sl_bin.data());
    // std::cout << "SL[0]: " << SL[0] << std::endl;
    // std::cout << "SL[1]: " << SL[1] << std::endl;

    size_t bt_elems = bt_bin.size() / sizeof(int32_t);
    int max_blocks_per_batch = bt_elems / cfg->B;
    int Dv = 512;
    std::vector<bf16> out_cpu(cfg->B * cfg->H * Dv);

    std::cout << "📦 FLATTENING KV CACHE..." << std::endl;
    size_t kv_stride = 0;
    auto start_flat = std::chrono::high_resolution_clock::now();
    std::vector<bf16> KV_flat = flatten_kv_cache(K_paged, BT, SL, cfg->B, cfg->D, cfg->BlockSize, max_blocks_per_batch, kv_stride);
    auto end_flat = std::chrono::high_resolution_clock::now();
    std::cout << "  Flattening time: " << std::chrono::duration<double, std::milli>(end_flat - start_flat).count() << " ms" << std::endl;

    std::cout << "🛠️  COMPILING 4D TENSOR KERNELS..." << std::endl;
    auto start_jit = std::chrono::high_resolution_clock::now();
    DNNLContextMacro ctx;
    ctx.init(Q, KV_flat.data(), out_cpu.data(), SL, cfg->B, cfg->H, cfg->D, Dv, cfg->scale, kv_stride);
    auto end_jit = std::chrono::high_resolution_clock::now();
    std::cout << "✅ JIT Time: " << std::chrono::duration<double, std::milli>(end_jit - start_jit).count() << " ms" << std::endl;

    std::cout << "🚀 RUNNING 4D KERNEL (WARMUP)..." << std::endl;
    dense_attention_onednn_no_custom_tiling(ctx, cfg->B);

    auto gold_bin = read_file("bins/out_golden.bin");
    float* gold = reinterpret_cast<float*>(gold_bin.data());
    int mismatch_count = 0;
    double max_diff = 0.0;
    for (size_t i = 0; i < out_cpu.size(); i++) {
        float cpu = bf16_to_fp32(out_cpu[i]);
        if (std::abs(cpu - gold[i]) > max_diff) max_diff = std::abs(cpu - gold[i]);
        if (std::isnan(cpu)) {
             std::cout << "❌ CPU produced NaN at index " << i << std::endl;
             return 1;
        }
        if (cpu != gold[i] && std::abs(cpu - gold[i]) > 0.05) {
            if (mismatch_count < 10) {
                std::cout << "❌ MISMATCH at index " << i << ": CPU = " << cpu << ", GPU = " << gold[i] << std::endl;
                mismatch_count++;
            }
        }
    }
    std::cout << "🔎 Max Diff: " << max_diff << (max_diff < 0.1 ? " (PASS)" : " (FAIL)") << std::endl;

    std::cout << "⏱️  BENCHMARKING 10 ITERATIONS..." << std::endl;
    int iterations = 10;
    auto start = std::chrono::high_resolution_clock::now();
    
    for(int i = 0; i < iterations; i++) {
        dense_attention_onednn_no_custom_tiling(ctx, cfg->B);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
    std::cout << "✅ Average Compute Time: " << ms << " ms" << std::endl;

    return 0;
}