#include <dnnl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <cstring>
#include <omp.h>
#include <immintrin.h>
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

bf16 fp32_to_bf16(fp32 val) {
    uint32_t bits;
    std::memcpy(&bits, &val, sizeof(bits));
    return static_cast<bf16>(bits >> 16);
}

std::vector<char> read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Failed to open " << path << std::endl;
        exit(1);
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    return buffer;
}

struct Config { int B, H, D, NumTokens; float scale; };

// --- DEEPSEEK FP8 TOKEN LAYOUT (656 Bytes Total) ---
struct DeepSeekToken {
    uint8_t quantized_nope[512]; // 512 FP8 (e4m3) values
    float scales[4];             // 4 FP32 scales (1 per 128 FP8s)
    bf16 rope[64];               // 64 BF16 unquantized values
} __attribute__((packed));

// Global LUT to simulate native FP8 decoding latency
float fp8_e4m3_lut[256];
void init_fp8_lut() {
    for(int i=0; i<256; i++) fp8_e4m3_lut[i] = (float(i) - 128.0f) * 0.01f; 
}

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
    std::vector<float> scores_buffer;
    std::vector<bf16> probs_buffer;

    DNNLContextMacro() : eng(engine::kind::cpu, 0), strm(eng) {}

    void init(const bf16* Q, bf16* Out, int B, int H, int D, int Dv, float scale, int seq_len, bf16* DenseKV_scratch) {
        scores_buffer.resize((size_t)B * H * seq_len, 0.0f);
        probs_buffer.resize((size_t)B * H * seq_len, 0);

        primitive_attr attr_bf16;
        attr_bf16.set_fpmath_mode(fpmath_mode::bf16);

        for (int b = 0; b < B; ++b) {
            BatchOps ops;
            memory::dims q_dims = {H, D};                      
            memory::dims k_dims = {D, seq_len};                
            memory::dims s_dims = {H, seq_len};                
            memory::dims p_dims = {H, seq_len};                
            memory::dims v_dims = {seq_len, Dv};               
            memory::dims out_dims = {H, Dv};                   

            memory::dims q_strides = {D, 1};                   
            memory::dims k_strides = {1, D};                   
            memory::dims s_strides = {seq_len, 1};             
            memory::dims p_strides = {seq_len, 1};             
            memory::dims v_strides = {D, 1};                   
            memory::dims out_strides = {Dv, 1};                

            auto q_md = memory::desc(q_dims, memory::data_type::bf16, q_strides);
            auto k_md = memory::desc(k_dims, memory::data_type::bf16, k_strides);
            auto s_md = memory::desc(s_dims, memory::data_type::f32, s_strides);
            auto p_md = memory::desc(p_dims, memory::data_type::bf16, p_strides);
            auto v_md = memory::desc(v_dims, memory::data_type::bf16, v_strides);
            auto out_md = memory::desc(out_dims, memory::data_type::bf16, out_strides);

            ops.qk_p = matmul(matmul::primitive_desc(eng, q_md, k_md, s_md, attr_bf16));
            ops.scale_p = eltwise_forward(eltwise_forward::primitive_desc(
                eng, prop_kind::forward_inference, algorithm::eltwise_linear, s_md, s_md, scale, 0.0f));
            ops.softmax_p = softmax_forward(softmax_forward::primitive_desc(
                eng, prop_kind::forward_inference, algorithm::softmax_accurate, s_md, p_md, 1));            
            ops.pv_p = matmul(matmul::primitive_desc(eng, p_md, v_md, out_md, attr_bf16));

            const bf16* q_ptr = Q + ((size_t)b * H * D);
            const bf16* kv_ptr = DenseKV_scratch + ((size_t)b * seq_len * D);
            float* s_ptr = scores_buffer.data() + ((size_t)b * H * seq_len);
            bf16* p_ptr = probs_buffer.data() + ((size_t)b * H * seq_len);
            bf16* out_ptr = Out + ((size_t)b * H * Dv);

            ops.q_mem = memory(q_md, eng, (void*)q_ptr);
            ops.k_mem = memory(k_md, eng, (void*)kv_ptr);
            ops.s_mem = memory(s_md, eng, (void*)s_ptr);
            ops.p_mem = memory(p_md, eng, (void*)p_ptr);
            ops.v_mem = memory(v_md, eng, (void*)kv_ptr); 
            ops.out_mem = memory(out_md, eng, (void*)out_ptr);

            ops.qk_args = {{DNNL_ARG_SRC, ops.q_mem}, {DNNL_ARG_WEIGHTS, ops.k_mem}, {DNNL_ARG_DST, ops.s_mem}};
            ops.scale_args = {{DNNL_ARG_SRC, ops.s_mem}, {DNNL_ARG_DST, ops.s_mem}}; 
            ops.softmax_args = {{DNNL_ARG_SRC, ops.s_mem}, {DNNL_ARG_DST, ops.p_mem}}; 
            ops.pv_args = {{DNNL_ARG_SRC, ops.p_mem}, {DNNL_ARG_WEIGHTS, ops.v_mem}, {DNNL_ARG_DST, ops.out_mem}};
            batch_ops.push_back(ops);
        }
    }
};

// 🔥 FULLY ENCAPSULATED SPARSE EXECUTION FUNCTION 🔥
void sparse_attention_onednn(
    DNNLContextMacro& ctx, 
    int B, int D, int active_seq_len,
    const DeepSeekToken* K_flat_fp8, 
    const int32_t* active_indices, 
    bf16* DenseKV_scratch // Pre-bound to oneDNN primitives
) {
    // --- Initialize Thread-Local Timers for Inner Loop ---
    int max_threads = omp_get_max_threads();
    std::vector<double> t_gather_thread(max_threads, 0.0);
    std::vector<double> t_dequant_thread(max_threads, 0.0);

    // --- Initialize oneDNN Timers ---
    double t_qk = 0.0;
    double t_scale = 0.0;
    double t_softmax = 0.0;
    double t_pv = 0.0;

    #pragma omp for schedule(static)
    for (int t = 0; t < active_seq_len; t++) {
        // ⏱️ TIMER 1A: GATHER
        double t_g_start = omp_get_wtime();
        int tid = omp_get_thread_num();
        int physical_token_idx = active_indices[t];
        
        // // 1. PERFECT PREFETCH: Tell the CPU to fetch the NEXT token's payload from DDR5 right now
        // if (t + 1 < active_seq_len) {
        //     const char* next_addr = (const char*)&K_flat_fp8[active_indices[t+1]];
        //     _mm_prefetch(next_addr, _MM_HINT_T0);          // Fetch first 64 bytes
        //     _mm_prefetch(next_addr + 64, _MM_HINT_T0);     // Fetch next cache line... etc
        // }

        const DeepSeekToken* __restrict src_token = &K_flat_fp8[physical_token_idx];     
        bf16* __restrict dst_token = DenseKV_scratch + (t * D);

        t_gather_thread[tid] += (omp_get_wtime() - t_g_start);

        // ⏱️ TIMER 1B: DEQUANT + RoPE
        double t_dq_start = omp_get_wtime();

        // 2. Vectorized Dequantization 
        for (int g = 0; g < 4; g++) { 
            __m512 v_scale = _mm512_set1_ps(src_token->scales[g]); 
            
            for (int v = 0; v < 128; v += 16) {
                __m128i fp8_16 = _mm_loadu_si128((__m128i*)&src_token->quantized_nope[g * 128 + v]);
                __m512i idx_32 = _mm512_cvtepu8_epi32(fp8_16);
                __m512 fp32_16 = _mm512_i32gather_ps(idx_32, fp8_e4m3_lut, 4); 
                fp32_16 = _mm512_mul_ps(fp32_16, v_scale);
                __m256i bf16_16 = (__m256i)_mm512_cvtneps_pbh(fp32_16);
                _mm256_storeu_si256((__m256i*)&dst_token[g * 128 + v], bf16_16);
            }
        }

        // 3. AVX-512 RoPE Copy (Replaces memcpy)
        // 128 bytes = exactly two 512-bit registers
        __m512i rope_0 = _mm512_loadu_si512((const void*)&src_token->rope[0]);
        __m512i rope_1 = _mm512_loadu_si512((const void*)&src_token->rope[32]);
        _mm512_storeu_si512((void*)&dst_token[512], rope_0);
        _mm512_storeu_si512((void*)&dst_token[544], rope_1);

        t_dequant_thread[tid] += (omp_get_wtime() - t_dq_start);
    }

    // Aggregate maximum time across threads for the OpenMP loop
    double max_gather = 0.0;
    double max_dequant = 0.0;
    for (int i = 0; i < max_threads; i++) {
        max_gather = std::max(max_gather, t_gather_thread[i]);
        max_dequant = std::max(max_dequant, t_dequant_thread[i]);
    }

    // 2. Fire the oneDNN AMX Math on the dense scratchpad
    for (int b = 0; b < B; ++b) {
        auto& ops = ctx.batch_ops[b];
        
        // ⏱️ TIMER 2: QK MatMul
        double t0 = omp_get_wtime();
        ops.qk_p.execute(ctx.strm, ops.qk_args);
        t_qk += (omp_get_wtime() - t0);
        
        // ⏱️ TIMER 3: Scale
        double t1 = omp_get_wtime();
        ops.scale_p.execute(ctx.strm, ops.scale_args);
        t_scale += (omp_get_wtime() - t1);
        
        // ⏱️ TIMER 4: Softmax
        double t2 = omp_get_wtime();
        ops.softmax_p.execute(ctx.strm, ops.softmax_args);
        t_softmax += (omp_get_wtime() - t2);
        
        // ⏱️ TIMER 5: PV MatMul
        double t3 = omp_get_wtime();
        ops.pv_p.execute(ctx.strm, ops.pv_args);
        t_pv += (omp_get_wtime() - t3);
    }
    ctx.strm.wait();

    // --- Print Results ---
    printf("\n========= AMX ONEDNN PIPELINE PROFILING ========\n");
    printf("1a. OMP Gather Loop    | %8.4f ms\n", max_gather * 1000.0);
    printf("1b. OMP Dequant + RoPE | %8.4f ms\n", max_dequant * 1000.0);
    printf("2. oneDNN Q * K^T      | %8.4f ms\n", t_qk * 1000.0);
    printf("3. oneDNN Scale        | %8.4f ms\n", t_scale * 1000.0);
    printf("4. oneDNN Softmax      | %8.4f ms\n", t_softmax * 1000.0);
    printf("5. oneDNN Scores * V   | %8.4f ms\n", t_pv * 1000.0);
    printf("------------------------------------------------\n");
    printf("Total Time             | %8.4f ms\n", (max_gather + max_dequant + t_qk + t_scale + t_softmax + t_pv) * 1000.0);
    printf("================================================\n\n");
}

// 🔥 FULLY ENCAPSULATED SPARSE EXECUTION FUNCTION 🔥
void sparse_attention_onednn_separated_dequant_and_gather(
    DNNLContextMacro& ctx, 
    int B, int D, int active_seq_len,
    const DeepSeekToken* K_flat_fp8, 
    const int32_t* active_indices, 
    bf16* DenseKV_scratch // Pre-bound to oneDNN primitives
) {
    // --- Initialize Thread-Local Timers for Inner Loop ---
    int max_threads = omp_get_max_threads();
    std::vector<double> t_gather_thread(max_threads, 0.0);
    std::vector<double> t_dequant_thread(max_threads, 0.0);

    // --- Initialize oneDNN Timers ---
    double t_qk = 0.0;
    double t_scale = 0.0;
    double t_softmax = 0.0;
    double t_pv = 0.0;

    #pragma omp for schedule(static)
    for (int t = 0; t < active_seq_len; t++) {
        // ⏱️ TIMER 1A: GATHER
        double t_g_start = omp_get_wtime();
        int tid = omp_get_thread_num();
        int physical_token_idx = active_indices[t];
        
        // // 1. PERFECT PREFETCH: Tell the CPU to fetch the NEXT token's payload from DDR5 right now
        // if (t + 1 < active_seq_len) {
        //     const char* next_addr = (const char*)&K_flat_fp8[active_indices[t+1]];
        //     _mm_prefetch(next_addr, _MM_HINT_T0);          // Fetch first 64 bytes
        //     _mm_prefetch(next_addr + 64, _MM_HINT_T0);     // Fetch next cache line... etc
        // }

        const DeepSeekToken* __restrict src_token = &K_flat_fp8[physical_token_idx];     
        bf16* __restrict dst_token = DenseKV_scratch + (t * D);

        t_gather_thread[tid] += (omp_get_wtime() - t_g_start);

        // ⏱️ TIMER 1B: DEQUANT + RoPE
        double t_dq_start = omp_get_wtime();

        // 2. Vectorized Dequantization 
        for (int g = 0; g < 4; g++) { 
            __m512 v_scale = _mm512_set1_ps(src_token->scales[g]); 
            
            for (int v = 0; v < 128; v += 16) {
                __m128i fp8_16 = _mm_loadu_si128((__m128i*)&src_token->quantized_nope[g * 128 + v]);
                __m512i idx_32 = _mm512_cvtepu8_epi32(fp8_16);
                __m512 fp32_16 = _mm512_i32gather_ps(idx_32, fp8_e4m3_lut, 4); 
                fp32_16 = _mm512_mul_ps(fp32_16, v_scale);
                __m256i bf16_16 = (__m256i)_mm512_cvtneps_pbh(fp32_16);
                _mm256_storeu_si256((__m256i*)&dst_token[g * 128 + v], bf16_16);
            }
        }

        // 3. AVX-512 RoPE Copy (Replaces memcpy)
        // 128 bytes = exactly two 512-bit registers
        __m512i rope_0 = _mm512_loadu_si512((const void*)&src_token->rope[0]);
        __m512i rope_1 = _mm512_loadu_si512((const void*)&src_token->rope[32]);
        _mm512_storeu_si512((void*)&dst_token[512], rope_0);
        _mm512_storeu_si512((void*)&dst_token[544], rope_1);

        t_dequant_thread[tid] += (omp_get_wtime() - t_dq_start);
    }

    // Aggregate maximum time across threads for the OpenMP loop
    double max_gather = 0.0;
    double max_dequant = 0.0;
    for (int i = 0; i < max_threads; i++) {
        max_gather = std::max(max_gather, t_gather_thread[i]);
        max_dequant = std::max(max_dequant, t_dequant_thread[i]);
    }

    // 2. Fire the oneDNN AMX Math on the dense scratchpad
    for (int b = 0; b < B; ++b) {
        auto& ops = ctx.batch_ops[b];
        
        // ⏱️ TIMER 2: QK MatMul
        double t0 = omp_get_wtime();
        ops.qk_p.execute(ctx.strm, ops.qk_args);
        t_qk += (omp_get_wtime() - t0);
        
        // ⏱️ TIMER 3: Scale
        double t1 = omp_get_wtime();
        ops.scale_p.execute(ctx.strm, ops.scale_args);
        t_scale += (omp_get_wtime() - t1);
        
        // ⏱️ TIMER 4: Softmax
        double t2 = omp_get_wtime();
        ops.softmax_p.execute(ctx.strm, ops.softmax_args);
        t_softmax += (omp_get_wtime() - t2);
        
        // ⏱️ TIMER 5: PV MatMul
        double t3 = omp_get_wtime();
        ops.pv_p.execute(ctx.strm, ops.pv_args);
        t_pv += (omp_get_wtime() - t3);
    }
    ctx.strm.wait();

    // --- Print Results ---
    printf("\n========= AMX ONEDNN PIPELINE PROFILING ========\n");
    printf("1a. OMP Gather Loop    | %8.4f ms\n", max_gather * 1000.0);
    printf("1b. OMP Dequant + RoPE | %8.4f ms\n", max_dequant * 1000.0);
    printf("2. oneDNN Q * K^T      | %8.4f ms\n", t_qk * 1000.0);
    printf("3. oneDNN Scale        | %8.4f ms\n", t_scale * 1000.0);
    printf("4. oneDNN Softmax      | %8.4f ms\n", t_softmax * 1000.0);
    printf("5. oneDNN Scores * V   | %8.4f ms\n", t_pv * 1000.0);
    printf("------------------------------------------------\n");
    printf("Total Time             | %8.4f ms\n", (max_gather + max_dequant + t_qk + t_scale + t_softmax + t_pv) * 1000.0);
    printf("================================================\n\n");
}

int main() {
    unsetenv("DNNL_MAX_CONCURRENCY");
    init_fp8_lut(); 
    
    std::cout << "⚡ LOADING SPARSE DATA..." << std::endl;
    auto cfg_data = read_file("sparse_bins/config.bin");
    Config* cfg = reinterpret_cast<Config*>(cfg_data.data());
    // std::cout << cfg;
    std::cout << "Batch Size: " << cfg->B << " | Active Tokens: " << cfg->NumTokens << std::endl;
    
    auto q_bin = read_file("sparse_bins/q.bin");
    auto k_bin = read_file("sparse_bins/k_cache.bin");
    auto idx_bin = read_file("sparse_bins/indices.bin");

    bf16* Q = reinterpret_cast<bf16*>(q_bin.data());
    const DeepSeekToken* K_flat_fp8 = reinterpret_cast<const DeepSeekToken*>(k_bin.data());
    int32_t* active_indices = reinterpret_cast<int32_t*>(idx_bin.data());

    int Dv = 512;
    std::vector<bf16> out_cpu(cfg->B * cfg->H * Dv);

    // Scratchpad to hold the 2048 dense bf16 tokens
    std::vector<bf16> DenseKV_scratch((size_t)cfg->B * cfg->NumTokens * cfg->D, 0);

    std::cout << "🛠️  COMPILING ONEDNN PIPELINE..." << std::endl;
    auto start_jit = std::chrono::high_resolution_clock::now();
    DNNLContextMacro ctx;
    ctx.init(Q, out_cpu.data(), cfg->B, cfg->H, cfg->D, Dv, cfg->scale, cfg->NumTokens, DenseKV_scratch.data());
    auto end_jit = std::chrono::high_resolution_clock::now();
    std::cout << "✅ JIT Time: " << std::chrono::duration<double, std::milli>(end_jit - start_jit).count() << " ms" << std::endl;

    std::cout << "🚀 RUNNING SPARSE KERNEL (WARMUP)..." << std::endl;
    sparse_attention_onednn(ctx, cfg->B, cfg->D, cfg->NumTokens, K_flat_fp8, active_indices, DenseKV_scratch.data());

    auto gold_bin = read_file("sparse_bins/out_golden.bin");
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
        sparse_attention_onednn(ctx, cfg->B, cfg->D, cfg->NumTokens, K_flat_fp8, active_indices, DenseKV_scratch.data());
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
    std::cout << "✅ Average Compute Time (Gather + Dequant + AMX): " << ms << " ms" << std::endl;

    return 0;
}