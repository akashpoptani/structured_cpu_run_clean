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

using bf16 = uint16_t;
using fp32 = float;

typedef struct __attribute__((aligned(64))) {
    uint8_t  palette_id;
    uint8_t  start_row;
    uint8_t  reserved_0[14];
    uint16_t colsb[16]; 
    uint8_t  rows[16];
} tile_config_t;


// --- HELPERS ---
fp32 bf16_to_fp32(bf16 val) {
    uint32_t bits = static_cast<uint32_t>(val) << 16;
    fp32 res;
    std::memcpy(&res, &bits, sizeof(res));
    return res;
}
// --- AMX SYSCALLS ---
#define XFEATURE_XTILECFG 17
#define XFEATURE_XTILEDATA 18
#define ARCH_GET_XCOMP_PERM     0x1022
#define ARCH_REQ_XCOMP_PERM     0x1023

bool init_amx() {
    unsigned long bitmask = 0;
    long status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);
    if (0 != status) return false;
    if (bitmask & (1 << XFEATURE_XTILEDATA)) return true;

    status = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
    return (0 == status);
}

// --- HELPERS ---
inline __m512 cvt_bf16_to_fp32(const bf16* src) {
    __m256i half = _mm256_loadu_si256((__m256i const*)src);
    __m512i int32s = _mm512_cvtepu16_epi32(half);
    __m512i shifted = _mm512_slli_epi32(int32s, 16);
    return _mm512_castsi512_ps(shifted);
}

inline void store_fp32_as_bf16(bf16* dst, __m512 val) {
    __m512i int_val = _mm512_castps_si512(val);
    __m512i shifted = _mm512_srli_epi32(int_val, 16);
    __m256i bf16_vals = _mm512_cvtepi32_epi16(shifted);
    _mm256_storeu_si256((__m256i*)dst, bf16_vals);
}

// Fast Exp (Same as before)
inline __m512 fast_exp_512(__m512 x) {
    __m512 c1 = _mm512_set1_ps(1.442695f);
    __m512 max_x = _mm512_set1_ps(88.0f);
    __m512 min_x = _mm512_set1_ps(-88.0f);
    x = _mm512_min_ps(x, max_x);
    x = _mm512_max_ps(x, min_x);
    __m512 t = _mm512_mul_ps(x, c1);
    __m512 r = _mm512_roundscale_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 f = _mm512_sub_ps(t, r);
    __m512 p = _mm512_set1_ps(0.00000198f);
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(0.0001398f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(0.0083334f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(0.0416657f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(0.1666666f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(0.5000000f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.0000000f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.0000000f));
    __m512i i = _mm512_cvtps_epi32(r);
    return _mm512_scalef_ps(p, _mm512_cvtepi32_ps(i));
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

// // Fast AVX-512 Exponential Approximation (Bypasses SVML)
// inline __m512 fast_exp_512(__m512 x) {
//     // 1. Multiply by log2(e)
//     __m512 log2e = _mm512_set1_ps(1.4426950408889634f);
//     __m512 y = _mm512_mul_ps(x, log2e);
    
//     // 2. Add magic constant 126.94269504f
//     __m512 magic = _mm512_set1_ps(126.94269504f);
//     __m512i i = _mm512_cvtps_epi32(_mm512_add_ps(y, magic));
    
//     // 3. Shift into the exponent bits of IEEE-754 FP32
//     return _mm512_castsi512_ps(_mm512_slli_epi32(i, 23));
// }

// --- FLATTEN FUNCTION ---
std::vector<bf16> flatten_kv_cache(
    const bf16* K_paged, const int32_t* BlockTable, const int32_t* SeqLens,
    int B, int D, int BlockSize, int max_blocks_per_batch, size_t& out_max_seq_len
) {
    if (BlockSize <= 0) BlockSize = 64;
    int max_len = 0;
    for(int b=0; b<B; b++) if(SeqLens[b] > max_len) max_len = SeqLens[b];

    // Align to BlockSize (and a 16 minimum for AMX). The inner loop below
    // memcpies whole `BlockSize` chunks per block; aligning only to 16 caused
    // the last block of any non-BlockSize-aligned sequence to write past the
    // batch's slice — that's the `double free or corruption (out)` we saw at
    // exit. ASAN trace: heap-buffer-overflow at flatten_kv_cache memcpy in
    // /scratch/.../flashmla_5c_asan_<jobid>.out (job 27159350).
    size_t align = (size_t)BlockSize;
    if (align < 16) align = 16;
    size_t max_len_aligned = ((size_t)max_len + align - 1) / align * align;
    if (max_len_aligned == 0) max_len_aligned = align;
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
                std::memcpy(dst_block, src_block, BlockSize * D * sizeof(bf16));
            }
        }
    }
    return flat_kv;
}

// --- PACKING HELPER (Crucial for AMX) ---
// Packs a [16 Tokens x 32 Dim] block of K into AMX B-Tile layout.
// Input: K[16 tokens][32 dims] (Row Major in memory)
// Output: 16 Rows, where each row contains 16 pairs of (K[2r], K[2r+1])
inline void pack_k_chunk_amx(const bf16* k_src, bf16* k_dst, int stride_d) {
    // Unroll the loops for speed since sizes are fixed (16x32)
    for (int r = 0; r < 16; r++) {
        // We pack 16 pairs (32 elements) per row
        // r corresponds to the AMX row index.
        // In the B-matrix, Row 'r' holds data for K-dimension indices 2*r and 2*r+1 across all 16 tokens.
        
        const bf16* src_row_0 = k_src + (2 * r);      // K dim 2r
        const bf16* src_row_1 = k_src + (2 * r + 1);  // K dim 2r+1
        
        bf16* dst_row = k_dst + (r * 32); // 32 bf16s per AMX row (64 bytes)

        #pragma GCC unroll 16
        for (int c = 0; c < 16; c++) {
            // c is the token index
            // We read from: src + c*stride_d
            dst_row[2*c]     = src_row_0[c * stride_d];
            dst_row[2*c + 1] = src_row_1[c * stride_d];
        }
    }
}

// --- FLATTENED AMX KERNEL ---
void dense_attention_amx_flat_timers(
    const bf16* Q, 
    const bf16* KV_flat, // Flattened Buffer [B, Stride, D]
    const int32_t* SeqLens,
    bf16* Out,
    int B, int H, int D, int BlockSize, float scale,
    size_t kv_stride // Stride (Max Seq Len Aligned)
) {
    int Dv = 512;
    int HEAD_GROUP = 16;

    #pragma omp parallel 
    {
        // 1. Init AMX
        init_amx(); // Ensure this is called in main or here
        static tile_config_t cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.palette_id = 1;
        cfg.start_row = 0;
        cfg.rows[0] = 16; cfg.colsb[0] = 64;
        cfg.rows[1] = 16; cfg.colsb[1] = 64;
        cfg.rows[2] = 16; cfg.colsb[2] = 64;
        _tile_loadconfig(&cfg);

        alignas(64) bf16 k_pack[64 * 576];

        // // --- TIMING VARIABLES ---
        // double t_pack = 0.0;
        // double t_amx = 0.0;
        // double t_softmax = 0.0;
        // double t_writeback=0.0;
        double t_valmul=0.0;

        #pragma omp for collapse(2) schedule(dynamic, 1)
        for (int b = 0; b < B; b++) {
            for (int h_start = 0; h_start < H; h_start += HEAD_GROUP) {
                
                float m_prev[16];
                float l_prev[16];
                alignas(64) float acc_o[16][512]; 

                for(int h=0; h<16; h++) {
                    m_prev[h] = -INFINITY;
                    l_prev[h] = 0.0f;
                    for(int d=0; d<512; d+=16) _mm512_store_ps(&acc_o[h][d], _mm512_setzero_ps());
                }

                const bf16* q_ptr_base = Q + (b * H * D) + (h_start * D);
                int seq_len = SeqLens[b];
                int num_blocks = (seq_len + BlockSize - 1) / BlockSize;

                // --- DIRECT FLATTENED POINTER ---
                const bf16* kv_batch_start = KV_flat + ((size_t)b * kv_stride * D);

                for (int i = 0; i < num_blocks; i++) {
                    // Direct access: No BlockTable lookup
                    const bf16* kv_block_ptr = kv_batch_start + ((size_t)i * BlockSize * D);

                    // --- TIMER START: PACKING ---
                    // auto t0 = std::chrono::high_resolution_clock::now();
                    
                    // kv_block_ptr IS k_raw
                    for(int t=0; t<64; t+=16) {
                        for(int d=0; d<576; d+=32) {
                            bf16* dst_tile = &k_pack[t*576 + d*16];
                            pack_k_chunk_amx(kv_block_ptr + t*D + d, dst_tile, D);
                        }
                    }
                    
                    // auto t1 = std::chrono::high_resolution_clock::now();
                    // t_pack += std::chrono::duration<double, std::milli>(t1 - t0).count();
                    // --- TIMER END: PACKING ---

                    int valid_tokens = (i < num_blocks - 1) ? 64 : (seq_len - i*64);

                    for(int t=0; t<64; t+=16) {
                        if (t >= valid_tokens) break;

                        // --- TIMER START: AMX COMPUTE ---
                        // auto t2 = std::chrono::high_resolution_clock::now();
                        
                        _tile_zero(0);
                        for (int k=0; k<576; k+=32) {
                            _tile_loadd(1, q_ptr_base + k, 1152);
                            _tile_loadd(2, &k_pack[t*576 + k*16], 64);
                            _tile_dpbf16ps(0, 1, 2);
                        }
                        alignas(64) float scores[16][16];
                        _tile_stored(0, scores, 64);
                        
                        // auto t3 = std::chrono::high_resolution_clock::now();
                        // t_amx += std::chrono::duration<double, std::milli>(t3 - t2).count();
                        // // --- TIMER END: AMX COMPUTE ---

                        // // --- TIMER START: SOFTMAX ---
                        // auto t4 = std::chrono::high_resolution_clock::now();
                        
                        // V is the first 512 elements of the same KV row
                        const bf16* v_base = kv_block_ptr + t*D;

                        for(int h=0; h<16; h++) {
                            float m_block = -INFINITY;
                            int limit = std::min(16, valid_tokens - t);
                            for(int tok=0; tok<limit; tok++) {
                                scores[h][tok] *= scale;
                                if (scores[h][tok] > m_block) m_block = scores[h][tok];
                            }
                            
                            float m_new = std::max(m_prev[h], m_block);
                            if(m_new == -INFINITY) m_new = 0.0f;
                            float alpha = std::exp(m_prev[h] - m_new);
                            float beta = std::exp(m_block - m_new);
                            
                            __m512 alpha_v = _mm512_set1_ps(alpha);
                            for(int d=0; d<512; d+=16) 
                                _mm512_store_ps(&acc_o[h][d], _mm512_mul_ps(_mm512_load_ps(&acc_o[h][d]), alpha_v));

                            float l_block = 0.0f;
                            for(int tok=0; tok<limit; tok++) {
                                float prob = std::exp(scores[h][tok] - m_block) * beta;
                                l_block += prob;
                                __m512 prob_v = _mm512_set1_ps(prob);
                                const bf16* v_row = v_base + tok*D;
                                // auto t8 = std::chrono::high_resolution_clock::now();
                                for(int d=0; d<512; d+=16) {
                                    __m512 v_val = cvt_bf16_to_fp32(v_row + d);
                                    _mm512_store_ps(&acc_o[h][d], _mm512_fmadd_ps(prob_v, v_val, _mm512_load_ps(&acc_o[h][d])));
                                }
                                // auto t9 = std::chrono::high_resolution_clock::now();
                                // t_valmul += std::chrono::duration<double, std::milli>(t9 - t8).count();
                            }
                            l_prev[h] = (l_prev[h] * alpha) + l_block;
                            m_prev[h] = m_new;
                        }
                        
                        // auto t5 = std::chrono::high_resolution_clock::now();
                        // t_softmax += std::chrono::duration<double, std::milli>(t5 - t4).count();
                        // // --- TIMER END: SOFTMAX ---
                    }
                }
                // // --- TIMER START: Writeback ---
                // auto t6 = std::chrono::high_resolution_clock::now();
                // Writeback
                for(int h=0; h<16; h++) {
                    bf16* out_ptr = Out + (b * H * Dv) + ((h_start + h) * Dv);
                    float denom = l_prev[h] == 0.0f ? 1e-10f : l_prev[h];
                    __m512 denom_v = _mm512_set1_ps(denom);
                    for(int d=0; d<512; d+=16) {
                        store_fp32_as_bf16(out_ptr + d, _mm512_div_ps(_mm512_load_ps(&acc_o[h][d]), denom_v));
                    }
                }
                // auto t7 = std::chrono::high_resolution_clock::now();
                // t_writeback += std::chrono::duration<double, std::milli>(t7 - t6).count();
            }
        }
        
        // Print Stats from Thread 0
        if (omp_get_thread_num() == 0) {
        //     printf("\n[Thread 0 Profile]\n");
        //     printf("  Packing K: %.2f ms\n", t_pack);
        //     printf("  AMX Math : %.2f ms\n", t_amx);
        //     printf("  Softmax/V: %.2f ms\n", t_softmax);
            printf("  Value Mul : %.2f ms\n", t_valmul);
        //     printf("  Writeback : %.2f ms\n", t_writeback);
        //     printf("----------------------\n");
        }
    }
}

void dense_attention_amx_flat_timers2(
    const bf16* Q, 
    const bf16* KV_flat, 
    const int32_t* SeqLens,
    bf16* Out,
    int B, int H, int D, int BlockSize, float scale,
    size_t kv_stride 
) {
    int Dv = 512;
    int HEAD_GROUP = 16;
    int H_GROUPS = H / HEAD_GROUP;
    
    // We split the Sequence Length into 16 chunks to generate enough tasks for 96 cores.
    // Total Tasks = B (1) * H_GROUPS (8) * NUM_CHUNKS (12) = 96 Tasks. 
    int NUM_CHUNKS = 12; 

    // --- TEMPORARY BUFFERS FOR MAP-REDUCE ---
    // Size: B * H_GROUPS * NUM_CHUNKS * HEAD_GROUP
    int num_states = B * H_GROUPS * NUM_CHUNKS * HEAD_GROUP;
    std::vector<float> global_m(num_states, -INFINITY);
    std::vector<float> global_l(num_states, 0.0f);
    std::vector<float> global_acc(num_states * Dv, 0.0f);

    #pragma omp parallel 
    {
        init_amx(); // Ensure this is called in main or here
        // 1. Init AMX (Once per thread)
        static tile_config_t cfg;
        memset(&cfg, 0, sizeof(cfg));

        cfg.palette_id = 1;
        cfg.start_row = 0;
        cfg.rows[0] = 16; cfg.colsb[0] = 64;
        cfg.rows[1] = 16; cfg.colsb[1] = 64;
        cfg.rows[2] = 16; cfg.colsb[2] = 64;
        _tile_loadconfig(&cfg);

        alignas(64) bf16 k_pack[64 * 576];

        // ==========================================
        // PHASE 1: THE MAP PHASE (Parallel over Sequence)
        // ==========================================
        #pragma omp for collapse(3) schedule(dynamic, 1)
        for (int b = 0; b < B; b++) {
            for (int h_g = 0; h_g < H_GROUPS; h_g++) {
                for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {
                    
                    int h_start = h_g * HEAD_GROUP;
                    const bf16* q_ptr_base = Q + (b * H * D) + (h_start * D);
                    int seq_len = SeqLens[b];
                    int num_blocks = (seq_len + BlockSize - 1) / BlockSize;

                    // Calculate sequence chunk boundaries
                    int blocks_per_chunk = (num_blocks + NUM_CHUNKS - 1) / NUM_CHUNKS;
                    int block_start = chunk * blocks_per_chunk;
                    int block_end = std::min(block_start + blocks_per_chunk, num_blocks);

                    // Thread-Local State
                    float m_local[16];
                    float l_local[16];
                    alignas(64) float acc_local[16][512]; 

                    for(int h=0; h<16; h++) {
                        m_local[h] = -INFINITY;
                        l_local[h] = 0.0f;
                        for(int d=0; d<512; d+=16) _mm512_store_ps(&acc_local[h][d], _mm512_setzero_ps());
                    }

                    const bf16* kv_batch_start = KV_flat + ((size_t)b * kv_stride * D);

                    // Loop ONLY over this thread's assigned chunk of blocks
                    for (int i = block_start; i < block_end; i++) {
                        const bf16* kv_block_ptr = kv_batch_start + ((size_t)i * BlockSize * D);

                        // Pack K Chunk
                        for(int t=0; t<64; t+=16) {
                            for(int d=0; d<576; d+=32) {
                                pack_k_chunk_amx(kv_block_ptr + t*D + d, &k_pack[t*576 + d*16], D);
                            }
                        }

                        int valid_tokens = (i < num_blocks - 1) ? 64 : (seq_len - i*64);

                        for(int t=0; t<64; t+=16) {
                            if (t >= valid_tokens) break;

                            _tile_zero(0);
                            for (int k=0; k<576; k+=32) {
                                _tile_loadd(1, q_ptr_base + k, 1152);
                                _tile_loadd(2, &k_pack[t*576 + k*16], 64);
                                _tile_dpbf16ps(0, 1, 2);
                            }
                            alignas(64) float scores[16][16];
                            _tile_stored(0, scores, 64);

                            const bf16* v_base = kv_block_ptr + t*D;

                            for(int h=0; h<16; h++) {
                                float m_block = -INFINITY;
                                int limit = std::min(16, valid_tokens - t);
                                for(int tok=0; tok<limit; tok++) {
                                    scores[h][tok] *= scale;
                                    if (scores[h][tok] > m_block) m_block = scores[h][tok];
                                }
                                
                                float m_new = std::max(m_local[h], m_block);
                                if(m_new == -INFINITY) m_new = 0.0f;
                                float alpha = std::exp(m_local[h] - m_new);
                                float beta = std::exp(m_block - m_new);
                                
                                __m512 alpha_v = _mm512_set1_ps(alpha);
                                for(int d=0; d<512; d+=16) 
                                    _mm512_store_ps(&acc_local[h][d], _mm512_mul_ps(_mm512_load_ps(&acc_local[h][d]), alpha_v));

                                float l_block = 0.0f;
                                for(int tok=0; tok<limit; tok++) {
                                    float prob = std::exp(scores[h][tok] - m_block) * beta;
                                    l_block += prob;
                                    __m512 prob_v = _mm512_set1_ps(prob);
                                    const bf16* v_row = v_base + tok*D;
                                    
                                    for(int d=0; d<512; d+=16) {
                                        __m512 v_val = cvt_bf16_to_fp32(v_row + d);
                                        _mm512_store_ps(&acc_local[h][d], _mm512_fmadd_ps(prob_v, v_val, _mm512_load_ps(&acc_local[h][d])));
                                    }
                                }
                                m_local[h] = m_new;
                                l_local[h] = (l_local[h] * alpha) + l_block;
                            }
                        }
                    }

                    // Save Local State to Global Temporary Buffers
                    int state_idx_base = (((b * H_GROUPS) + h_g) * NUM_CHUNKS + chunk) * HEAD_GROUP;
                    for (int h = 0; h < 16; h++) {
                        global_m[state_idx_base + h] = m_local[h];
                        global_l[state_idx_base + h] = l_local[h];
                        for (int d = 0; d < 512; d++) {
                            global_acc[(state_idx_base + h) * Dv + d] = acc_local[h][d];
                        }
                    }
                } // End Chunk Loop
            } // End Head Group Loop
        } // End Batch Loop

        // Implicit OpenMP Barrier here guarantees all chunks are processed.

        // ==========================================
        // PHASE 2: THE REDUCE PHASE (Merge Chunks)
        // ==========================================
        #pragma omp for collapse(2) schedule(dynamic, 1)
        for (int b = 0; b < B; b++) {
            for (int h_g = 0; h_g < H_GROUPS; h_g++) {
                
                int h_start = h_g * HEAD_GROUP;

                for (int h = 0; h < 16; h++) {
                    float m_final = -INFINITY;
                    float l_final = 0.0f;
                    alignas(64) float acc_final[512] = {0};

                    // Merge all chunks for this specific head
                    for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {
                        int state_idx = (((b * H_GROUPS) + h_g) * NUM_CHUNKS + chunk) * HEAD_GROUP + h;
                        
                        float m_chunk = global_m[state_idx];
                        float l_chunk = global_l[state_idx];
                        float* acc_chunk = &global_acc[state_idx * Dv];

                        if (m_chunk == -INFINITY) continue;

                        float m_new = std::max(m_final, m_chunk);
                        float alpha_final = std::exp(m_final - m_new);
                        float alpha_chunk = std::exp(m_chunk - m_new);

                        // AVX-512 Reduction Math
                        __m512 a_f_v = _mm512_set1_ps(alpha_final);
                        __m512 a_c_v = _mm512_set1_ps(alpha_chunk);

                        for (int d = 0; d < 512; d+=16) {
                            __m512 acc_f = _mm512_load_ps(&acc_final[d]);
                            __m512 acc_c = _mm512_loadu_ps(&acc_chunk[d]);
                            acc_f = _mm512_mul_ps(acc_f, a_f_v);
                            acc_f = _mm512_fmadd_ps(acc_c, a_c_v, acc_f);
                            _mm512_store_ps(&acc_final[d], acc_f);
                        }

                        l_final = (l_final * alpha_final) + (l_chunk * alpha_chunk);
                        m_final = m_new;
                    }

                    // Writeback to final Output
                    bf16* out_ptr = Out + (b * H * Dv) + ((h_start + h) * Dv);
                    float denom = (l_final == 0.0f) ? 1e-10f : l_final;
                    __m512 denom_v = _mm512_set1_ps(denom);
                    
                    for(int d=0; d<512; d+=16) {
                        __m512 acc = _mm512_load_ps(&acc_final[d]);
                        store_fp32_as_bf16(out_ptr + d, _mm512_div_ps(acc, denom_v));
                    }
                }
            }
        }
    } // End Parallel
}

void dense_attention_amx_flat_timers3(
    const bf16* Q, 
    const bf16* KV_flat, 
    const int32_t* SeqLens,
    bf16* Out,
    int B, int H, int D, int BlockSize, float scale,
    size_t kv_stride 
) {
    int Dv = 512;
    int HEAD_GROUP = 16;
    int H_GROUPS = H / HEAD_GROUP;
    int NUM_CHUNKS = 12; 

    int num_states = B * H_GROUPS * NUM_CHUNKS * HEAD_GROUP;
    std::vector<float> global_m(num_states, -INFINITY);
    std::vector<float> global_l(num_states, 0.0f);
    std::vector<float> global_acc(num_states * Dv, 0.0f);

    // --- PROFILING ARRAYS (One slot per thread to avoid race conditions) ---
    int max_threads = omp_get_max_threads();
    std::vector<double> t_pack(max_threads, 0.0);
    std::vector<double> t_qkt(max_threads, 0.0);
    std::vector<double> t_softmax_v(max_threads, 0.0);
    std::vector<double> t_v_only(max_threads, 0.0);
    std::vector<double> t_map_total(max_threads, 0.0);
    std::vector<double> t_barrier(max_threads, 0.0);
    std::vector<double> t_reduce(max_threads, 0.0);

    #pragma omp parallel 
    {
        int tid = omp_get_thread_num();
        init_amx(); // Wakes up OS AMX context

        static tile_config_t cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.palette_id = 1;
        cfg.start_row = 0;
        cfg.rows[0] = 16; cfg.colsb[0] = 64;
        cfg.rows[1] = 16; cfg.colsb[1] = 64;
        cfg.rows[2] = 16; cfg.colsb[2] = 64;
        _tile_loadconfig(&cfg);

        alignas(64) bf16 k_pack[64 * 576];

        double map_start = omp_get_wtime();

        // ==========================================
        // PHASE 1: MAP PHASE (nowait allows us to measure barrier)
        // ==========================================
        #pragma omp for collapse(3) schedule(dynamic, 1) nowait
        for (int b = 0; b < B; b++) {
            for (int h_g = 0; h_g < H_GROUPS; h_g++) {
                for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {
                    
                    int h_start = h_g * HEAD_GROUP;
                    const bf16* q_ptr_base = Q + (b * H * D) + (h_start * D);
                    int seq_len = SeqLens[b];
                    int num_blocks = (seq_len + BlockSize - 1) / BlockSize;

                    int blocks_per_chunk = (num_blocks + NUM_CHUNKS - 1) / NUM_CHUNKS;
                    int block_start = chunk * blocks_per_chunk;
                    int block_end = std::min(block_start + blocks_per_chunk, num_blocks);

                    float m_local[16];
                    float l_local[16];
                    alignas(64) float acc_local[16][512]; 

                    for(int h=0; h<16; h++) {
                        m_local[h] = -INFINITY;
                        l_local[h] = 0.0f;
                        for(int d=0; d<512; d+=16) _mm512_store_ps(&acc_local[h][d], _mm512_setzero_ps());
                    }

                    const bf16* kv_batch_start = KV_flat + ((size_t)b * kv_stride * D);

                    for (int i = block_start; i < block_end; i++) {
                        const bf16* kv_block_ptr = kv_batch_start + ((size_t)i * BlockSize * D);

                        // ⏱️ TIMER: PACKING
                        // double t0 = omp_get_wtime();
                        for(int t=0; t<64; t+=16) {
                            for(int d=0; d<576; d+=32) {
                                pack_k_chunk_amx(kv_block_ptr + t*D + d, &k_pack[t*576 + d*16], D);
                            }
                        }
                        // t_pack[tid] += (omp_get_wtime() - t0);

                        int valid_tokens = (i < num_blocks - 1) ? 64 : (seq_len - i*64);

                        for(int t=0; t<64; t+=16) {
                            if (t >= valid_tokens) break;

                            // ⏱️ TIMER: Q * K^T (AMX)
                            double t1 = omp_get_wtime();
                            _tile_zero(0);
                            for (int k=0; k<576; k+=32) {
                                // double t3 = omp_get_wtime();
                                _tile_loadd(1, q_ptr_base + k, 1152);
                                _tile_loadd(2, &k_pack[t*576 + k*16], 64);
                                // t_v_only[tid] += (omp_get_wtime() - t3);
                                // double t3 = omp_get_wtime();
                                _tile_dpbf16ps(0, 1, 2);
                                // t_v_only[tid] += (omp_get_wtime() - t3);
                            }
                            alignas(64) float scores[16][16];
                            // double t1 = omp_get_wtime();
                            _tile_stored(0, scores, 64);
                            t_qkt[tid] += (omp_get_wtime() - t1);

                            const bf16* v_base = kv_block_ptr + t*D;

                            // ⏱️ TIMER: SOFTMAX + V MUL
                            // double t2 = omp_get_wtime();
                            for(int h=0; h<16; h++) {
                                float m_block = -INFINITY;
                                int limit = std::min(16, valid_tokens - t);
                                for(int tok=0; tok<limit; tok++) {
                                    scores[h][tok] *= scale;
                                    if (scores[h][tok] > m_block) m_block = scores[h][tok];
                                }
                                
                                float m_new = std::max(m_local[h], m_block);
                                if(m_new == -INFINITY) m_new = 0.0f;
                                float alpha = std::exp(m_local[h] - m_new);
                                float beta = std::exp(m_block - m_new);
                                
                                __m512 alpha_v = _mm512_set1_ps(alpha);
                                for(int d=0; d<512; d+=16) 
                                    _mm512_store_ps(&acc_local[h][d], _mm512_mul_ps(_mm512_load_ps(&acc_local[h][d]), alpha_v));

                                float l_block = 0.0f;
                                
                                // ⏱️ TIMER: ONLY V MUL
                                // double t3 = omp_get_wtime();
                                for(int tok=0; tok<limit; tok++) {
                                    float prob = std::exp(scores[h][tok] - m_block) * beta;
                                    l_block += prob;
                                    __m512 prob_v = _mm512_set1_ps(prob);
                                    const bf16* v_row = v_base + tok*D;
                                    
                                    for(int d=0; d<512; d+=16) {
                                        __m512 v_val = cvt_bf16_to_fp32(v_row + d);
                                        _mm512_store_ps(&acc_local[h][d], _mm512_fmadd_ps(prob_v, v_val, _mm512_load_ps(&acc_local[h][d])));
                                    }
                                }
                                // t_v_only[tid] += (omp_get_wtime() - t3);

                                m_local[h] = m_new;
                                l_local[h] = (l_local[h] * alpha) + l_block;
                            }
                            // t_softmax_v[tid] += (omp_get_wtime() - t2);
                        }
                    }

                    int state_idx_base = (((b * H_GROUPS) + h_g) * NUM_CHUNKS + chunk) * HEAD_GROUP;
                    for (int h = 0; h < 16; h++) {
                        global_m[state_idx_base + h] = m_local[h];
                        global_l[state_idx_base + h] = l_local[h];
                        for (int d = 0; d < 512; d++) {
                            global_acc[(state_idx_base + h) * Dv + d] = acc_local[h][d];
                        }
                    }
                } 
            } 
        } 
        t_map_total[tid] = omp_get_wtime() - map_start;

        // ⏱️ TIMER: BARRIER (Wait for slowest thread to finish Map Phase)
        double b_start = omp_get_wtime();
        #pragma omp barrier
        t_barrier[tid] = omp_get_wtime() - b_start;

        // ==========================================
        // PHASE 2: REDUCE PHASE
        // ==========================================
        double r_start = omp_get_wtime();
        #pragma omp for collapse(2) schedule(dynamic, 1)
        for (int b = 0; b < B; b++) {
            for (int h_g = 0; h_g < H_GROUPS; h_g++) {
                
                int h_start = h_g * HEAD_GROUP;

                for (int h = 0; h < 16; h++) {
                    float m_final = -INFINITY;
                    float l_final = 0.0f;
                    alignas(64) float acc_final[512] = {0};

                    for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {
                        int state_idx = (((b * H_GROUPS) + h_g) * NUM_CHUNKS + chunk) * HEAD_GROUP + h;
                        
                        float m_chunk = global_m[state_idx];
                        float l_chunk = global_l[state_idx];
                        float* acc_chunk = &global_acc[state_idx * Dv];

                        if (m_chunk == -INFINITY) continue;

                        float m_new = std::max(m_final, m_chunk);
                        float alpha_final = std::exp(m_final - m_new);
                        float alpha_chunk = std::exp(m_chunk - m_new);

                        __m512 a_f_v = _mm512_set1_ps(alpha_final);
                        __m512 a_c_v = _mm512_set1_ps(alpha_chunk);

                        for (int d = 0; d < 512; d+=16) {
                            __m512 acc_f = _mm512_load_ps(&acc_final[d]);
                            __m512 acc_c = _mm512_loadu_ps(&acc_chunk[d]);
                            acc_f = _mm512_mul_ps(acc_f, a_f_v);
                            acc_f = _mm512_fmadd_ps(acc_c, a_c_v, acc_f);
                            _mm512_store_ps(&acc_final[d], acc_f);
                        }

                        l_final = (l_final * alpha_final) + (l_chunk * alpha_chunk);
                        m_final = m_new;
                    }

                    bf16* out_ptr = Out + (b * H * Dv) + ((h_start + h) * Dv);
                    float denom = (l_final == 0.0f) ? 1e-10f : l_final;
                    __m512 denom_v = _mm512_set1_ps(denom);
                    
                    for(int d=0; d<512; d+=16) {
                        __m512 acc = _mm512_load_ps(&acc_final[d]);
                        store_fp32_as_bf16(out_ptr + d, _mm512_div_ps(acc, denom_v));
                    }
                }
            }
        }
        t_reduce[tid] = omp_get_wtime() - r_start;
    } // End Parallel

    // --- AGGREGATE AND PRINT STATS ---
    double max_pack = 0, max_qkt = 0, max_sv = 0, max_v = 0, max_map = 0, max_bar = 0, max_red = 0;
    double sum_pack = 0, sum_qkt = 0, sum_sv = 0, sum_v = 0;
    int active_threads = 0;

    for (int i = 0; i < max_threads; i++) {
        if (t_map_total[i] > 0) { // Only count threads that actually got work
            active_threads++;
            sum_pack += t_pack[i]; max_pack = std::max(max_pack, t_pack[i]);
            sum_qkt += t_qkt[i];   max_qkt = std::max(max_qkt, t_qkt[i]);
            sum_sv += t_softmax_v[i]; max_sv = std::max(max_sv, t_softmax_v[i]);
            sum_v += t_v_only[i];  max_v = std::max(max_v, t_v_only[i]);
            max_map = std::max(max_map, t_map_total[i]);
            max_bar = std::max(max_bar, t_barrier[i]);
            max_red = std::max(max_red, t_reduce[i]);
        }
    }

    if (active_threads > 0) {
        printf("\n================ MICRO-PROFILING STATS ================\n");
        printf("Active Threads       : %d / %d\n", active_threads, max_threads);
        printf("-------------------------------------------------------\n");
        printf("COMPONENT              | AVERAGE TIME | MAX (CRITICAL) TIME\n");
        printf("-------------------------------------------------------\n");
        printf("1. K-Packing           | %8.2f ms | %8.2f ms\n", (sum_pack / active_threads)*1000.0, max_pack*1000.0);
        printf("2. AMX (Q * K^T)       | %8.2f ms | %8.2f ms\n", (sum_qkt / active_threads)*1000.0, max_qkt*1000.0);
        printf("3. Softmax + V_mul     | %8.2f ms | %8.2f ms\n", (sum_sv / active_threads)*1000.0, max_sv*1000.0);
        printf("   -> (V_mul only)     | %8.2f ms | %8.2f ms\n", (sum_v / active_threads)*1000.0, max_v*1000.0);
        printf("-------------------------------------------------------\n");
        printf("Total Map Phase        |      -       | %8.2f ms\n", max_map*1000.0);
        printf("Barrier Wait Time      |      -       | %8.2f ms\n", max_bar*1000.0);
        printf("Phase 2 (Reduce)       |      -       | %8.2f ms\n", max_red*1000.0);
        printf("=======================================================\n\n");
    }
}

void dense_attention_amx_flat_timers4(
    const bf16* Q, 
    const bf16* KV_flat, 
    const int32_t* SeqLens,
    bf16* Out,
    int B, int H, int D, int BlockSize, float scale,
    size_t kv_stride 
) {
    int Dv = 512;
    int HEAD_GROUP = 16;
    int H_GROUPS = H / HEAD_GROUP;
    int NUM_CHUNKS = 12; 

    int num_states = B * H_GROUPS * NUM_CHUNKS * HEAD_GROUP;
    std::vector<float> global_m(num_states, -INFINITY);
    std::vector<float> global_l(num_states, 0.0f);
    std::vector<float> global_acc(num_states * Dv, 0.0f);

    // --- PROFILING ARRAYS (One slot per thread to avoid race conditions) ---
    // int max_threads = omp_get_max_threads();
    // std::vector<double> t_pack(max_threads, 0.0);
    // std::vector<double> t_qkt(max_threads, 0.0);
    // std::vector<double> t_softmax_v(max_threads, 0.0);
    // std::vector<double> t_v_only(max_threads, 0.0);
    // std::vector<double> t_map_total(max_threads, 0.0);
    // std::vector<double> t_barrier(max_threads, 0.0);
    // std::vector<double> t_reduce(max_threads, 0.0);

    #pragma omp parallel 
    {
        int tid = omp_get_thread_num();
        init_amx(); // Wakes up OS AMX context

        // 🚨 UPDATE: Initialize ALL 8 AMX Tiles, not just 3
        static tile_config_t cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.palette_id = 1;
        cfg.start_row = 0;
        for(int i=0; i<8; i++) {
            cfg.rows[i] = 16; 
            cfg.colsb[i] = 64;
        }
        _tile_loadconfig(&cfg);

        alignas(64) bf16 k_pack[64 * 576];

        // double map_start = omp_get_wtime();

        // ==========================================
        // PHASE 1: MAP PHASE
        // ==========================================
        #pragma omp for collapse(3) schedule(dynamic, 1) nowait
        for (int b = 0; b < B; b++) {
            for (int h_g = 0; h_g < H_GROUPS; h_g++) {
                for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {
                    
                    int h_start = h_g * HEAD_GROUP;
                    const bf16* q_ptr_base = Q + (b * H * D) + (h_start * D);
                    int seq_len = SeqLens[b];
                    int num_blocks = (seq_len + BlockSize - 1) / BlockSize;

                    int blocks_per_chunk = (num_blocks + NUM_CHUNKS - 1) / NUM_CHUNKS;
                    int block_start = chunk * blocks_per_chunk;
                    int block_end = std::min(block_start + blocks_per_chunk, num_blocks);

                    float m_local[16];
                    float l_local[16];
                    alignas(64) float acc_local[16][512]; 

                    for(int h=0; h<16; h++) {
                        m_local[h] = -INFINITY;
                        l_local[h] = 0.0f;
                        for(int d=0; d<512; d+=16) _mm512_store_ps(&acc_local[h][d], _mm512_setzero_ps());
                    }

                    const bf16* kv_batch_start = KV_flat + ((size_t)b * kv_stride * D);

                    for (int i = block_start; i < block_end; i++) {
                        const bf16* kv_block_ptr = kv_batch_start + ((size_t)i * BlockSize * D);
                        int valid_tokens = (i < num_blocks - 1) ? 64 : (seq_len - i*64);

                        // ⏱️ TIMER: PACKING
                        // double t0 = omp_get_wtime();
                        // Pack the entire 64-token block at once
                        for(int t=0; t<64; t+=16) {
                            for(int d=0; d<576; d+=32) {
                                pack_k_chunk_amx(kv_block_ptr + t*D + d, &k_pack[t*576 + d*16], D);
                            }
                        }
                        // t_pack[tid] += (omp_get_wtime() - t0);

                        // ⏱️ TIMER: Q * K^T (4x UNROLLED AMX)
                        // double t1 = omp_get_wtime();
                        
                        _tile_zero(0); // Acc for Tokens 0-15
                        _tile_zero(1); // Acc for Tokens 16-31
                        _tile_zero(2); // Acc for Tokens 32-47
                        _tile_zero(3); // Acc for Tokens 48-63
                        
                        for (int k=0; k<576; k+=32) {
                            // Load Q exactly ONCE into TMM4
                            _tile_loadd(4, q_ptr_base + k, 1152);

                            // Dispatch math for all 4 Token chunks asynchronously
                            _tile_loadd(5, &k_pack[0*576 + k*16], 64);
                            _tile_dpbf16ps(0, 4, 5);

                            _tile_loadd(6, &k_pack[16*576 + k*16], 64);
                            _tile_dpbf16ps(1, 4, 6);

                            _tile_loadd(7, &k_pack[32*576 + k*16], 64);
                            _tile_dpbf16ps(2, 4, 7);

                            _tile_loadd(5, &k_pack[48*576 + k*16], 64); // Safe to reuse TMM5 here
                            _tile_dpbf16ps(3, 4, 5);
                        }
                        
                        // Stitch all 4 tiles into a contiguous [16][64] array seamlessly using a 256-byte stride
                        alignas(64) float scores[16][64];
                        _tile_stored(0, &scores[0][0],  256);
                        _tile_stored(1, &scores[0][16], 256);
                        _tile_stored(2, &scores[0][32], 256);
                        _tile_stored(3, &scores[0][48], 256);
                        
                        // t_qkt[tid] += (omp_get_wtime() - t1);

                        // ⏱️ TIMER: SOFTMAX + V MUL (Unified across all 64 tokens)
                        // double t2 = omp_get_wtime();
                        const bf16* v_base = kv_block_ptr;

                        for(int h=0; h<16; h++) {
                            float m_block = -INFINITY;
                            
                            // 1. Max over valid tokens
                            for(int tok=0; tok<valid_tokens; tok++) {
                                scores[h][tok] *= scale;
                                if (scores[h][tok] > m_block) m_block = scores[h][tok];
                            }
                            
                            float m_new = std::max(m_local[h], m_block);
                            if(m_new == -INFINITY) m_new = 0.0f;
                            float alpha = std::exp(m_local[h] - m_new);
                            float beta = std::exp(m_block - m_new);
                            
                            // 2. Rescale previous accumulator
                            __m512 alpha_v = _mm512_set1_ps(alpha);
                            for(int d=0; d<512; d+=16) 
                                _mm512_store_ps(&acc_local[h][d], _mm512_mul_ps(_mm512_load_ps(&acc_local[h][d]), alpha_v));

                            float l_block = 0.0f;
                            
                            // ⏱️ TIMER: ONLY V MUL
                            // double t3 = omp_get_wtime();
                            // 3. Compute Probabilities and multiply V
                            for(int tok=0; tok<valid_tokens; tok++) {
                                float prob = std::exp(scores[h][tok] - m_block) * beta;
                                l_block += prob;
                                
                                __m512 prob_v = _mm512_set1_ps(prob);
                                const bf16* v_row = v_base + tok*D;
                                
                                for(int d=0; d<512; d+=16) {
                                    __m512 v_val = cvt_bf16_to_fp32(v_row + d);
                                    _mm512_store_ps(&acc_local[h][d], _mm512_fmadd_ps(prob_v, v_val, _mm512_load_ps(&acc_local[h][d])));
                                }
                            }
                            // t_v_only[tid] += (omp_get_wtime() - t3);

                            m_local[h] = m_new;
                            l_local[h] = (l_local[h] * alpha) + l_block;
                        }
                        // t_softmax_v[tid] += (omp_get_wtime() - t2);
                    }

                    int state_idx_base = (((b * H_GROUPS) + h_g) * NUM_CHUNKS + chunk) * HEAD_GROUP;
                    for (int h = 0; h < 16; h++) {
                        global_m[state_idx_base + h] = m_local[h];
                        global_l[state_idx_base + h] = l_local[h];
                        for (int d = 0; d < 512; d++) {
                            global_acc[(state_idx_base + h) * Dv + d] = acc_local[h][d];
                        }
                    }
                } 
            } 
        } 
        // t_map_total[tid] = omp_get_wtime() - map_start;

        // ⏱️ TIMER: BARRIER
        // double b_start = omp_get_wtime();
        #pragma omp barrier
        // t_barrier[tid] = omp_get_wtime() - b_start;

        // ==========================================
        // PHASE 2: REDUCE PHASE
        // ==========================================
        // double r_start = omp_get_wtime();
        #pragma omp for collapse(2) schedule(dynamic, 1)
        for (int b = 0; b < B; b++) {
            for (int h_g = 0; h_g < H_GROUPS; h_g++) {
                
                int h_start = h_g * HEAD_GROUP;

                for (int h = 0; h < 16; h++) {
                    float m_final = -INFINITY;
                    float l_final = 0.0f;
                    alignas(64) float acc_final[512] = {0};

                    for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {
                        int state_idx = (((b * H_GROUPS) + h_g) * NUM_CHUNKS + chunk) * HEAD_GROUP + h;
                        
                        float m_chunk = global_m[state_idx];
                        float l_chunk = global_l[state_idx];
                        float* acc_chunk = &global_acc[state_idx * Dv];

                        if (m_chunk == -INFINITY) continue;

                        float m_new = std::max(m_final, m_chunk);
                        float alpha_final = std::exp(m_final - m_new);
                        float alpha_chunk = std::exp(m_chunk - m_new);

                        __m512 a_f_v = _mm512_set1_ps(alpha_final);
                        __m512 a_c_v = _mm512_set1_ps(alpha_chunk);

                        for (int d = 0; d < 512; d+=16) {
                            __m512 acc_f = _mm512_load_ps(&acc_final[d]);
                            __m512 acc_c = _mm512_loadu_ps(&acc_chunk[d]);
                            acc_f = _mm512_mul_ps(acc_f, a_f_v);
                            acc_f = _mm512_fmadd_ps(acc_c, a_c_v, acc_f);
                            _mm512_store_ps(&acc_final[d], acc_f);
                        }

                        l_final = (l_final * alpha_final) + (l_chunk * alpha_chunk);
                        m_final = m_new;
                    }

                    bf16* out_ptr = Out + (b * H * Dv) + ((h_start + h) * Dv);
                    float denom = (l_final == 0.0f) ? 1e-10f : l_final;
                    __m512 denom_v = _mm512_set1_ps(denom);
                    
                    for(int d=0; d<512; d+=16) {
                        __m512 acc = _mm512_load_ps(&acc_final[d]);
                        store_fp32_as_bf16(out_ptr + d, _mm512_div_ps(acc, denom_v));
                    }
                }
            }
        }
        // t_reduce[tid] = omp_get_wtime() - r_start;
    } // End Parallel

    // // --- AGGREGATE AND PRINT STATS ---
    // double max_pack = 0, max_qkt = 0, max_sv = 0, max_v = 0, max_map = 0, max_bar = 0, max_red = 0;
    // double sum_pack = 0, sum_qkt = 0, sum_sv = 0, sum_v = 0;
    // int active_threads = 0;

    // for (int i = 0; i < max_threads; i++) {
    //     // if (t_map_total[i] > 0) { 
    //         active_threads++;
    //         sum_pack += t_pack[i]; max_pack = std::max(max_pack, t_pack[i]);
    //         sum_qkt += t_qkt[i];   max_qkt = std::max(max_qkt, t_qkt[i]);
    //         sum_sv += t_softmax_v[i]; max_sv = std::max(max_sv, t_softmax_v[i]);
    //         sum_v += t_v_only[i];  max_v = std::max(max_v, t_v_only[i]);
    //         max_map = std::max(max_map, t_map_total[i]);
    //         max_bar = std::max(max_bar, t_barrier[i]);
    //         max_red = std::max(max_red, t_reduce[i]);
    //     // }
    // }

    // if (active_threads > 0) {
    //     printf("\n================ MICRO-PROFILING STATS ================\n");
    //     printf("Active Threads       : %d / %d\n", active_threads, max_threads);
    //     printf("-------------------------------------------------------\n");
    //     printf("COMPONENT              | AVERAGE TIME | MAX (CRITICAL) TIME\n");
    //     printf("-------------------------------------------------------\n");
    //     printf("1. K-Packing           | %8.2f ms | %8.2f ms\n", (sum_pack / active_threads)*1000.0, max_pack*1000.0);
    //     printf("2. AMX (Q * K^T)       | %8.2f ms | %8.2f ms\n", (sum_qkt / active_threads)*1000.0, max_qkt*1000.0);
    //     printf("3. Softmax + V_mul     | %8.2f ms | %8.2f ms\n", (sum_sv / active_threads)*1000.0, max_sv*1000.0);
    //     printf("   -> (V_mul only)     | %8.2f ms | %8.2f ms\n", (sum_v / active_threads)*1000.0, max_v*1000.0);
    //     printf("-------------------------------------------------------\n");
    //     printf("Total Map Phase        |      -       | %8.2f ms\n", max_map*1000.0);
    //     printf("Barrier Wait Time      |      -       | %8.2f ms\n", max_bar*1000.0);
    //     printf("Phase 2 (Reduce)       |      -       | %8.2f ms\n", max_red*1000.0);
    //     printf("=======================================================\n\n");
    // }
}

void dense_attention_amx_flat_timers5(
    const bf16* Q, 
    const bf16* KV_flat, 
    const int32_t* SeqLens,
    bf16* Out,
    int B, int H, int D, int BlockSize, float scale,
    size_t kv_stride 
) {
    int Dv = 512;
    int HEAD_GROUP = 16;
    int H_GROUPS = H / HEAD_GROUP;
    int NUM_CHUNKS = 12; 

    int num_states = B * H_GROUPS * NUM_CHUNKS * HEAD_GROUP;
    std::vector<float> global_m(num_states, -INFINITY);
    std::vector<float> global_l(num_states, 0.0f);
    std::vector<float> global_acc(num_states * Dv, 0.0f);

    int max_threads = omp_get_max_threads();
    std::vector<double> t_kpack(max_threads, 0.0);
    std::vector<double> t_qkt(max_threads, 0.0);
    std::vector<double> t_soft_cast(max_threads, 0.0);
    std::vector<double> t_amx_v(max_threads, 0.0);
    std::vector<double> t_map_total(max_threads, 0.0);

    #pragma omp parallel 
    {
        int tid = omp_get_thread_num();
        init_amx(); 

        static tile_config_t cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.palette_id = 1;
        cfg.start_row = 0;
        for(int i=0; i<8; i++) {
            cfg.rows[i] = 16; 
            cfg.colsb[i] = 64;
        }
        _tile_loadconfig(&cfg);

        alignas(64) bf16 k_pack[64 * 576];

        double map_start = omp_get_wtime();

        #pragma omp for collapse(3) schedule(dynamic, 1) nowait
        for (int b = 0; b < B; b++) {
            for (int h_g = 0; h_g < H_GROUPS; h_g++) {
                for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {
                    
                    int h_start = h_g * HEAD_GROUP;
                    const bf16* q_ptr_base = Q + (b * H * D) + (h_start * D);
                    int seq_len = SeqLens[b];
                    int num_blocks = (seq_len + BlockSize - 1) / BlockSize;

                    int blocks_per_chunk = (num_blocks + NUM_CHUNKS - 1) / NUM_CHUNKS;
                    int block_start = chunk * blocks_per_chunk;
                    int block_end = std::min(block_start + blocks_per_chunk, num_blocks);

                    float m_local[16];
                    float l_local[16];
                    alignas(64) float acc_local[16][512]; 

                    for(int h=0; h<16; h++) {
                        m_local[h] = -INFINITY;
                        l_local[h] = 0.0f;
                        for(int d=0; d<512; d+=16) _mm512_store_ps(&acc_local[h][d], _mm512_setzero_ps());
                    }

                    const bf16* kv_batch_start = KV_flat + ((size_t)b * kv_stride * D);

                    for (int i = block_start; i < block_end; i++) {
                        const bf16* kv_block_ptr = kv_batch_start + ((size_t)i * BlockSize * D);
                        int valid_tokens = (i < num_blocks - 1) ? 64 : (seq_len - i*64);

                        // ⏱️ TIMER: K-PACKING
                        double t0 = omp_get_wtime();
                        for(int t=0; t<64; t+=16) {
                            for(int d=0; d<576; d+=32) {
                                pack_k_chunk_amx(kv_block_ptr + t*D + d, &k_pack[t*576 + d*16], D);
                            }
                        }
                        t_kpack[tid] += (omp_get_wtime() - t0);

                        // ⏱️ TIMER: Q * K^T (4x UNROLLED)
                        double t1 = omp_get_wtime();
                        _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3); 
                        for (int k=0; k<576; k+=32) {
                            _tile_loadd(4, q_ptr_base + k, 1152);
                            _tile_loadd(5, &k_pack[0*576 + k*16], 64);   _tile_dpbf16ps(0, 4, 5);
                            _tile_loadd(6, &k_pack[16*576 + k*16], 64);  _tile_dpbf16ps(1, 4, 6);
                            _tile_loadd(7, &k_pack[32*576 + k*16], 64);  _tile_dpbf16ps(2, 4, 7);
                            _tile_loadd(5, &k_pack[48*576 + k*16], 64);  _tile_dpbf16ps(3, 4, 5);
                        }
                        alignas(64) float scores[16][64];
                        _tile_stored(0, &scores[0][0],  256);
                        _tile_stored(1, &scores[0][16], 256);
                        _tile_stored(2, &scores[0][32], 256);
                        _tile_stored(3, &scores[0][48], 256);
                        t_qkt[tid] += (omp_get_wtime() - t1);

                        // ⏱️ TIMER: SOFTMAX + BF16 CAST + AVX RESCALE
                        double t2 = omp_get_wtime();
                        alignas(64) bf16 scores_bf16[16][64];

                        for(int h=0; h<16; h++) {
                            float m_block = -INFINITY;
                            for(int tok=0; tok<valid_tokens; tok++) {
                                scores[h][tok] *= scale;
                                if (scores[h][tok] > m_block) m_block = scores[h][tok];
                            }
                            
                            float m_new = std::max(m_local[h], m_block);
                            if(m_new == -INFINITY) m_new = 0.0f;
                            float alpha = std::exp(m_local[h] - m_new);
                            float beta = std::exp(m_block - m_new);
                            
                            // AVX-512 Scalar Rescale
                            __m512 alpha_v = _mm512_set1_ps(alpha);
                            for(int d=0; d<512; d+=16) 
                                _mm512_store_ps(&acc_local[h][d], _mm512_mul_ps(_mm512_load_ps(&acc_local[h][d]), alpha_v));

                            float l_block = 0.0f;
                            
                            // FP32 -> BF16 Fast Cast
                            for(int tok=0; tok<64; tok+=16) {
                                __m512 prob_v = _mm512_setzero_ps();
                                if (tok < valid_tokens) {
                                    __m512 s_vec = _mm512_load_ps(&scores[h][tok]);
                                    __m512 m_vec = _mm512_set1_ps(m_block);
                                    __m512 b_vec = _mm512_set1_ps(beta);
                                    
                                    // Requires -mveclibabi=svml flag for GCC
                                    prob_v = _mm512_mul_ps(fast_exp_512(_mm512_sub_ps(s_vec, m_vec)), b_vec);
                                    l_block += _mm512_reduce_add_ps(prob_v);
                                }
                                
                                __m512i int_val = _mm512_castps_si512(prob_v);
                                __m512i shifted = _mm512_srli_epi32(int_val, 16);
                                __m256i bf16_vals = _mm512_cvtepi32_epi16(shifted);
                                _mm256_storeu_si256((__m256i*)&scores_bf16[h][tok], bf16_vals);
                            }
                            m_local[h] = m_new;
                            l_local[h] = (l_local[h] * alpha) + l_block;
                        }
                        t_soft_cast[tid] += (omp_get_wtime() - t2);

                        // ⏱️ TIMER: AMX V-MUL (UNPACKED STRIDED LOAD)
                        double t3 = omp_get_wtime();
                        
                        for(int d=0; d<512; d+=16) {
                            _tile_zero(0); // Holds partial Acc for Tokens 0-63
                            
                            // Chunk 1: Tokens 0-31
                            _tile_loadd(2, &scores_bf16[0][0], 128);       
                            // STRIDED LOAD (The Mathematical Trap)
                            _tile_loadd(3, kv_block_ptr + 0*D + d, D * sizeof(bf16));     
                            _tile_dpbf16ps(0, 2, 3);
                            
                            // Chunk 2: Tokens 32-63
                            _tile_loadd(4, &scores_bf16[0][32], 128);      
                            _tile_loadd(5, kv_block_ptr + 32*D + d, D * sizeof(bf16));    
                            _tile_dpbf16ps(0, 4, 5);
                            
                            alignas(64) float delta_o[16][16];
                            _tile_stored(0, delta_o, 64);
                            
                            // Add to AVX-512 Accumulator
                            for(int h=0; h<16; h++) {
                                __m512 acc = _mm512_load_ps(&acc_local[h][d]);
                                __m512 d0  = _mm512_load_ps(&delta_o[h][0]);
                                acc = _mm512_add_ps(acc, d0);
                                _mm512_store_ps(&acc_local[h][d], acc);
                            }
                        }
                        t_amx_v[tid] += (omp_get_wtime() - t3);
                    }

                    int state_idx_base = (((b * H_GROUPS) + h_g) * NUM_CHUNKS + chunk) * HEAD_GROUP;
                    for (int h = 0; h < 16; h++) {
                        global_m[state_idx_base + h] = m_local[h];
                        global_l[state_idx_base + h] = l_local[h];
                        for (int d = 0; d < 512; d++) {
                            global_acc[(state_idx_base + h) * Dv + d] = acc_local[h][d];
                        }
                    }
                } 
            } 
        } 
        t_map_total[tid] = omp_get_wtime() - map_start;

        #pragma omp barrier

        // ==========================================
        // PHASE 2: REDUCE PHASE 
        // ==========================================
        #pragma omp for collapse(2) schedule(dynamic, 1)
        for (int b = 0; b < B; b++) {
            for (int h_g = 0; h_g < H_GROUPS; h_g++) {
                int h_start = h_g * HEAD_GROUP;
                for (int h = 0; h < 16; h++) {
                    float m_final = -INFINITY;
                    float l_final = 0.0f;
                    alignas(64) float acc_final[512] = {0};

                    for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {
                        int state_idx = (((b * H_GROUPS) + h_g) * NUM_CHUNKS + chunk) * HEAD_GROUP + h;
                        float m_chunk = global_m[state_idx];
                        float l_chunk = global_l[state_idx];
                        float* acc_chunk = &global_acc[state_idx * Dv];

                        if (m_chunk == -INFINITY) continue;

                        float m_new = std::max(m_final, m_chunk);
                        float alpha_final = std::exp(m_final - m_new);
                        float alpha_chunk = std::exp(m_chunk - m_new);

                        __m512 a_f_v = _mm512_set1_ps(alpha_final);
                        __m512 a_c_v = _mm512_set1_ps(alpha_chunk);

                        for (int d = 0; d < 512; d+=16) {
                            __m512 acc_f = _mm512_load_ps(&acc_final[d]);
                            __m512 acc_c = _mm512_loadu_ps(&acc_chunk[d]);
                            acc_f = _mm512_mul_ps(acc_f, a_f_v);
                            acc_f = _mm512_fmadd_ps(acc_c, a_c_v, acc_f);
                            _mm512_store_ps(&acc_final[d], acc_f);
                        }

                        l_final = (l_final * alpha_final) + (l_chunk * alpha_chunk);
                        m_final = m_new;
                    }

                    bf16* out_ptr = Out + (b * H * Dv) + ((h_start + h) * Dv);
                    float denom = (l_final == 0.0f) ? 1e-10f : l_final;
                    __m512 denom_v = _mm512_set1_ps(denom);
                    
                    for(int d=0; d<512; d+=16) {
                        __m512 acc = _mm512_load_ps(&acc_final[d]);
                        store_fp32_as_bf16(out_ptr + d, _mm512_div_ps(acc, denom_v));
                    }
                }
            }
        }
    }

    // --- AGGREGATE STATS ---
    double max_kpack = 0, max_qkt = 0, max_soft = 0, max_amxv = 0, max_map = 0;
    int active_threads = 0;
    for (int i = 0; i < max_threads; i++) {
        if (t_map_total[i] > 0) { 
            active_threads++;
            max_kpack = std::max(max_kpack, t_kpack[i]);
            max_qkt = std::max(max_qkt, t_qkt[i]);
            max_soft = std::max(max_soft, t_soft_cast[i]);
            max_amxv = std::max(max_amxv, t_amx_v[i]);
            max_map = std::max(max_map, t_map_total[i]);
        }
    }

    if (active_threads > 0) {
        printf("\n========= STRIDED AMX (UNPACKED) PROFILING ========\n");
        printf("COMPONENT              | MAX (CRITICAL) TIME\n");
        printf("---------------------------------------------------\n");
        printf("1. K-Packing (VNNI)    | %8.2f ms\n", max_kpack*1000.0);
        printf("2. AMX (Q * K^T)       | %8.2f ms\n", max_qkt*1000.0);
        printf("3. Softmax+AVX+Cast    | %8.2f ms\n", max_soft*1000.0);
        printf("4. AMX (Scores * V)    | %8.2f ms\n", max_amxv*1000.0);
        printf("---------------------------------------------------\n");
        printf("Total Map Phase        | %8.2f ms\n", max_map*1000.0);
        printf("===================================================\n\n");
    }
}

void dense_attention_amx_flat_timers6(
    const bf16* Q, 
    const bf16* KV_flat, 
    const int32_t* SeqLens,
    bf16* Out,
    int B, int H, int D, int BlockSize, float scale,
    size_t kv_stride 
) {
    int Dv = 512;
    int HEAD_GROUP = 16;
    int H_GROUPS = H / HEAD_GROUP;
    int NUM_CHUNKS = 12; // Optimized for 72 cores (8 * 9 = 72)

    int num_states = B * H_GROUPS * NUM_CHUNKS * HEAD_GROUP;
    std::vector<float> global_m(num_states, -INFINITY);
    std::vector<float> global_l(num_states, 0.0f);
    std::vector<float> global_acc(num_states * Dv, 0.0f);

    int max_threads = omp_get_max_threads();
    std::vector<double> t_kpack(max_threads, 0.0);
    std::vector<double> t_qkt(max_threads, 0.0);
    std::vector<double> t_soft_cast(max_threads, 0.0);
    std::vector<double> t_vpack(max_threads, 0.0);
    std::vector<double> t_amx_v(max_threads, 0.0);
    std::vector<double> t_map_total(max_threads, 0.0);

    #pragma omp parallel 
    {
        int tid = omp_get_thread_num();
        init_amx(); 

        static tile_config_t cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.palette_id = 1;
        cfg.start_row = 0;
        for(int i=0; i<8; i++) {
            cfg.rows[i] = 16; 
            cfg.colsb[i] = 64;
        }
        _tile_loadconfig(&cfg);

        // L1 Cache Scratchpads for AMX VNNI Layout
        alignas(64) bf16 k_pack[64 * 576];
        alignas(64) bf16 v_pack[64 * 512]; 

        double map_start = omp_get_wtime();

        #pragma omp for collapse(3) schedule(dynamic, 1) nowait
        for (int b = 0; b < B; b++) {
            for (int h_g = 0; h_g < H_GROUPS; h_g++) {
                for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {
                    
                    int h_start = h_g * HEAD_GROUP;
                    const bf16* q_ptr_base = Q + (b * H * D) + (h_start * D);
                    int seq_len = SeqLens[b];
                    int num_blocks = (seq_len + BlockSize - 1) / BlockSize;

                    int blocks_per_chunk = (num_blocks + NUM_CHUNKS - 1) / NUM_CHUNKS;
                    int block_start = chunk * blocks_per_chunk;
                    int block_end = std::min(block_start + blocks_per_chunk, num_blocks);

                    float m_local[16];
                    float l_local[16];
                    alignas(64) float acc_local[16][512]; 

                    for(int h=0; h<16; h++) {
                        m_local[h] = -INFINITY;
                        l_local[h] = 0.0f;
                        for(int d=0; d<512; d+=16) _mm512_store_ps(&acc_local[h][d], _mm512_setzero_ps());
                    }

                    const bf16* kv_batch_start = KV_flat + ((size_t)b * kv_stride * D);

                    for (int i = block_start; i < block_end; i++) {
                        const bf16* kv_block_ptr = kv_batch_start + ((size_t)i * BlockSize * D);
                        int valid_tokens = (i < num_blocks - 1) ? 64 : (seq_len - i*64);

                        // ⏱️ 1. K-PACKING
                        double t0 = omp_get_wtime();
                        for(int t=0; t<64; t+=16) {
                            for(int d=0; d<576; d+=32) {
                                pack_k_chunk_amx(kv_block_ptr + t*D + d, &k_pack[t*576 + d*16], D);
                            }
                        }
                        t_kpack[tid] += (omp_get_wtime() - t0);

                        // ⏱️ 2. Q * K^T (AMX)
                        double t1 = omp_get_wtime();
                        _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3); 
                        for (int k=0; k<576; k+=32) {
                            _tile_loadd(4, q_ptr_base + k, 1152);
                            _tile_loadd(5, &k_pack[0*576 + k*16], 64);   _tile_dpbf16ps(0, 4, 5);
                            _tile_loadd(6, &k_pack[16*576 + k*16], 64);  _tile_dpbf16ps(1, 4, 6);
                            _tile_loadd(7, &k_pack[32*576 + k*16], 64);  _tile_dpbf16ps(2, 4, 7);
                            _tile_loadd(5, &k_pack[48*576 + k*16], 64);  _tile_dpbf16ps(3, 4, 5);
                        }
                        alignas(64) float scores[16][64];
                        _tile_stored(0, &scores[0][0],  256);
                        _tile_stored(1, &scores[0][16], 256);
                        _tile_stored(2, &scores[0][32], 256);
                        _tile_stored(3, &scores[0][48], 256);
                        t_qkt[tid] += (omp_get_wtime() - t1);

                        // ⏱️ 3. SOFTMAX + CAST
                        double t2 = omp_get_wtime();
                        alignas(64) bf16 scores_bf16[16][64];

                        for(int h=0; h<16; h++) {
                            float m_block = -INFINITY;
                            for(int tok=0; tok<valid_tokens; tok++) {
                                scores[h][tok] *= scale;
                                if (scores[h][tok] > m_block) m_block = scores[h][tok];
                            }
                            
                            float m_new = std::max(m_local[h], m_block);
                            if(m_new == -INFINITY) m_new = 0.0f;
                            float alpha = std::exp(m_local[h] - m_new);
                            float beta = std::exp(m_block - m_new);
                            
                            __m512 alpha_v = _mm512_set1_ps(alpha);
                            for(int d=0; d<512; d+=16) 
                                _mm512_store_ps(&acc_local[h][d], _mm512_mul_ps(_mm512_load_ps(&acc_local[h][d]), alpha_v));

                            float l_block = 0.0f;
                            
                            for(int tok=0; tok<64; tok+=16) {
                                __m512 prob_v = _mm512_setzero_ps();
                                if (tok < valid_tokens) {
                                    __m512 s_vec = _mm512_load_ps(&scores[h][tok]);
                                    __m512 m_vec = _mm512_set1_ps(m_block);
                                    __m512 b_vec = _mm512_set1_ps(beta);
                                    prob_v = _mm512_mul_ps(fast_exp_512(_mm512_sub_ps(s_vec, m_vec)), b_vec);
                                    l_block += _mm512_reduce_add_ps(prob_v);
                                }
                                
                                __m512i int_val = _mm512_castps_si512(prob_v);
                                __m512i shifted = _mm512_srli_epi32(int_val, 16);
                                __m256i bf16_vals = _mm512_cvtepi32_epi16(shifted);
                                _mm256_storeu_si256((__m256i*)&scores_bf16[h][tok], bf16_vals);
                            }
                            m_local[h] = m_new;
                            l_local[h] = (l_local[h] * alpha) + l_block;
                        }
                        t_soft_cast[tid] += (omp_get_wtime() - t2);

                        // ⏱️ 4. V-PACKING (VNNI Format for Tokens)
                        double t3 = omp_get_wtime();
                        for (int d_blk = 0; d_blk < 512; d_blk += 16) {
                            for (int t_blk = 0; t_blk < 64; t_blk += 32) {
                                bf16* dst = &v_pack[(d_blk / 16) * 1024 + (t_blk / 32) * 512];
                                for (int r = 0; r < 16; r++) {
                                    int tok = t_blk + r * 2;
                                    for (int c = 0; c < 16; c++) {
                                        bf16 val0 = (tok < valid_tokens) ? kv_block_ptr[tok * D + d_blk + c] : 0;
                                        bf16 val1 = (tok + 1 < valid_tokens) ? kv_block_ptr[(tok + 1) * D + d_blk + c] : 0;
                                        dst[r * 32 + c * 2 + 0] = val0;
                                        dst[r * 32 + c * 2 + 1] = val1;
                                    }
                                }
                            }
                        }
                        t_vpack[tid] += (omp_get_wtime() - t3);

                        // ⏱️ 5. P * V (AMX)
                        double t4 = omp_get_wtime();
                        
                        // Load P (16 Heads x 64 Tokens) cleanly into two tiles using 128-byte stride
                        _tile_loadd(4, &scores_bf16[0][0], 128);  // Tokens 0-31
                        _tile_loadd(5, &scores_bf16[0][32], 128); // Tokens 32-63
                        
                        for(int d=0; d<512; d+=16) {
                            _tile_zero(0); // Accumulate output features
                            
                            bf16* v_ptr_t0 = &v_pack[(d / 16) * 1024];
                            bf16* v_ptr_t32 = &v_pack[(d / 16) * 1024 + 512];
                            
                            _tile_loadd(6, v_ptr_t0, 64);   // Load Tokens 0-31
                            _tile_dpbf16ps(0, 4, 6);
                            
                            _tile_loadd(7, v_ptr_t32, 64);  // Load Tokens 32-63
                            _tile_dpbf16ps(0, 5, 7);
                            
                            alignas(64) float delta_o[16][16];
                            _tile_stored(0, delta_o, 64);
                            
                            // Add perfectly aligned output directly into AVX-512 state
                            for(int h=0; h<16; h++) {
                                __m512 acc = _mm512_load_ps(&acc_local[h][d]);
                                __m512 d0  = _mm512_load_ps(&delta_o[h][0]);
                                acc = _mm512_add_ps(acc, d0);
                                _mm512_store_ps(&acc_local[h][d], acc);
                            }
                        }
                        t_amx_v[tid] += (omp_get_wtime() - t4);
                    }

                    int state_idx_base = (((b * H_GROUPS) + h_g) * NUM_CHUNKS + chunk) * HEAD_GROUP;
                    for (int h = 0; h < 16; h++) {
                        global_m[state_idx_base + h] = m_local[h];
                        global_l[state_idx_base + h] = l_local[h];
                        for (int d = 0; d < 512; d++) {
                            global_acc[(state_idx_base + h) * Dv + d] = acc_local[h][d];
                        }
                    }
                } 
            } 
        } 
        t_map_total[tid] = omp_get_wtime() - map_start;

        #pragma omp barrier

        // ==========================================
        // PHASE 2: REDUCE PHASE 
        // ==========================================
        #pragma omp for collapse(2) schedule(dynamic, 1)
        for (int b = 0; b < B; b++) {
            for (int h_g = 0; h_g < H_GROUPS; h_g++) {
                int h_start = h_g * HEAD_GROUP;
                for (int h = 0; h < 16; h++) {
                    float m_final = -INFINITY;
                    float l_final = 0.0f;
                    alignas(64) float acc_final[512] = {0};

                    for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {
                        int state_idx = (((b * H_GROUPS) + h_g) * NUM_CHUNKS + chunk) * HEAD_GROUP + h;
                        float m_chunk = global_m[state_idx];
                        float l_chunk = global_l[state_idx];
                        float* acc_chunk = &global_acc[state_idx * Dv];

                        if (m_chunk == -INFINITY) continue;

                        float m_new = std::max(m_final, m_chunk);
                        float alpha_final = std::exp(m_final - m_new);
                        float alpha_chunk = std::exp(m_chunk - m_new);

                        __m512 a_f_v = _mm512_set1_ps(alpha_final);
                        __m512 a_c_v = _mm512_set1_ps(alpha_chunk);

                        for (int d = 0; d < 512; d+=16) {
                            __m512 acc_f = _mm512_load_ps(&acc_final[d]);
                            __m512 acc_c = _mm512_loadu_ps(&acc_chunk[d]);
                            acc_f = _mm512_mul_ps(acc_f, a_f_v);
                            acc_f = _mm512_fmadd_ps(acc_c, a_c_v, acc_f);
                            _mm512_store_ps(&acc_final[d], acc_f);
                        }

                        l_final = (l_final * alpha_final) + (l_chunk * alpha_chunk);
                        m_final = m_new;
                    }

                    bf16* out_ptr = Out + (b * H * Dv) + ((h_start + h) * Dv);
                    float denom = (l_final == 0.0f) ? 1e-10f : l_final;
                    __m512 denom_v = _mm512_set1_ps(denom);
                    
                    for(int d=0; d<512; d+=16) {
                        __m512 acc = _mm512_load_ps(&acc_final[d]);
                        store_fp32_as_bf16(out_ptr + d, _mm512_div_ps(acc, denom_v));
                    }
                }
            }
        }
    }

    // --- AGGREGATE STATS ---
    double max_kpack = 0, max_qkt = 0, max_soft = 0, max_vpack = 0, max_amxv = 0, max_map = 0;
    int active_threads = 0;
    for (int i = 0; i < max_threads; i++) {
        if (t_map_total[i] > 0) { 
            active_threads++;
            max_kpack = std::max(max_kpack, t_kpack[i]);
            max_qkt = std::max(max_qkt, t_qkt[i]);
            max_soft = std::max(max_soft, t_soft_cast[i]);
            max_vpack = std::max(max_vpack, t_vpack[i]);
            max_amxv = std::max(max_amxv, t_amx_v[i]);
            max_map = std::max(max_map, t_map_total[i]);
        }
    }

    if (active_threads > 0) {
        printf("\n========= FULL AMX PIPELINE PROFILING ========\n");
        printf("COMPONENT              | MAX (CRITICAL) TIME\n");
        printf("---------------------------------------------------\n");
        printf("1. K-Packing (L1 VNNI) | %8.2f ms\n", max_kpack*1000.0);
        printf("2. AMX (Q * K^T)       | %8.2f ms\n", max_qkt*1000.0);
        printf("3. Softmax+AVX+Cast    | %8.2f ms\n", max_soft*1000.0);
        printf("4. V-Packing (L1 VNNI) | %8.2f ms\n", max_vpack*1000.0);
        printf("5. AMX (Scores * V)    | %8.2f ms\n", max_amxv*1000.0);
        printf("---------------------------------------------------\n");
        printf("Total Map Phase        | %8.2f ms\n", max_map*1000.0);
        printf("===================================================\n\n");
    }
}

void dense_attention_amx_flat_timers6_temp(
    const bf16* Q, 
    const bf16* KV_flat, 
    const int32_t* SeqLens,
    bf16* Out,
    int B, int H, int D, int BlockSize, float scale,
    size_t kv_stride 
) {
    int Dv = 512;
    int HEAD_GROUP = 16;
    int H_GROUPS = H / HEAD_GROUP;
    int NUM_CHUNKS = 12; // Optimized for 72 cores (8 * 9 = 72)

    int num_states = B * H_GROUPS * NUM_CHUNKS * HEAD_GROUP;
    std::vector<float> global_m(num_states, -INFINITY);
    std::vector<float> global_l(num_states, 0.0f);
    std::vector<float> global_acc(num_states * Dv, 0.0f);

    int max_threads = omp_get_max_threads();
    std::vector<double> t_kpack(max_threads, 0.0);
    std::vector<double> t_qkt(max_threads, 0.0);
    std::vector<double> t_soft_cast(max_threads, 0.0);
    std::vector<double> t_vpack(max_threads, 0.0);
    std::vector<double> t_amx_v(max_threads, 0.0);
    std::vector<double> t_map_total(max_threads, 0.0);
    
    // --- NEW TIMERS ---
    std::vector<double> t_barrier(max_threads, 0.0);
    std::vector<double> t_reduce(max_threads, 0.0);

    #pragma omp parallel 
    {
        int tid = omp_get_thread_num();
        init_amx(); 

        static tile_config_t cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.palette_id = 1;
        cfg.start_row = 0;
        for(int i=0; i<8; i++) {
            cfg.rows[i] = 16; 
            cfg.colsb[i] = 64;
        }
        _tile_loadconfig(&cfg);

        // L1 Cache Scratchpads for AMX VNNI Layout
        alignas(64) bf16 k_pack[64 * 576];
        alignas(64) bf16 v_pack[64 * 512]; 

        double map_start = omp_get_wtime();

        #pragma omp for collapse(3) schedule(dynamic, 1) nowait
        for (int b = 0; b < B; b++) {
            for (int h_g = 0; h_g < H_GROUPS; h_g++) {
                for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {
                    
                    int h_start = h_g * HEAD_GROUP;
                    const bf16* q_ptr_base = Q + (b * H * D) + (h_start * D);
                    int seq_len = SeqLens[b];
                    int num_blocks = (seq_len + BlockSize - 1) / BlockSize;

                    int blocks_per_chunk = (num_blocks + NUM_CHUNKS - 1) / NUM_CHUNKS;
                    int block_start = chunk * blocks_per_chunk;
                    int block_end = std::min(block_start + blocks_per_chunk, num_blocks);

                    float m_local[16];
                    float l_local[16];
                    alignas(64) float acc_local[16][512]; 

                    for(int h=0; h<16; h++) {
                        m_local[h] = -INFINITY;
                        l_local[h] = 0.0f;
                        for(int d=0; d<512; d+=16) _mm512_store_ps(&acc_local[h][d], _mm512_setzero_ps());
                    }

                    const bf16* kv_batch_start = KV_flat + ((size_t)b * kv_stride * D);

                    for (int i = block_start; i < block_end; i++) {
                        const bf16* kv_block_ptr = kv_batch_start + ((size_t)i * BlockSize * D);
                        int valid_tokens = (i < num_blocks - 1) ? 64 : (seq_len - i*64);

                        // ⏱️ 1. K-PACKING
                        double t0 = omp_get_wtime();
                        for(int t=0; t<64; t+=16) {
                            for(int d=0; d<576; d+=32) {
                                pack_k_chunk_amx(kv_block_ptr + t*D + d, &k_pack[t*576 + d*16], D);
                            }
                        }
                        t_kpack[tid] += (omp_get_wtime() - t0);

                        // ⏱️ 2. Q * K^T (AMX)
                        double t1 = omp_get_wtime();
                        _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3); 
                        for (int k=0; k<576; k+=32) {
                            _tile_loadd(4, q_ptr_base + k, 1152);
                            _tile_loadd(5, &k_pack[0*576 + k*16], 64);   _tile_dpbf16ps(0, 4, 5);
                            _tile_loadd(6, &k_pack[16*576 + k*16], 64);  _tile_dpbf16ps(1, 4, 6);
                            _tile_loadd(7, &k_pack[32*576 + k*16], 64);  _tile_dpbf16ps(2, 4, 7);
                            _tile_loadd(5, &k_pack[48*576 + k*16], 64);  _tile_dpbf16ps(3, 4, 5);
                        }
                        alignas(64) float scores[16][64];
                        _tile_stored(0, &scores[0][0],  256);
                        _tile_stored(1, &scores[0][16], 256);
                        _tile_stored(2, &scores[0][32], 256);
                        _tile_stored(3, &scores[0][48], 256);
                        t_qkt[tid] += (omp_get_wtime() - t1);

                        // ⏱️ 3. SOFTMAX + CAST
                        double t2 = omp_get_wtime();
                        alignas(64) bf16 scores_bf16[16][64];

                        for(int h=0; h<16; h++) {
                            float m_block = -INFINITY;
                            for(int tok=0; tok<valid_tokens; tok++) {
                                scores[h][tok] *= scale;
                                if (scores[h][tok] > m_block) m_block = scores[h][tok];
                            }
                            
                            float m_new = std::max(m_local[h], m_block);
                            if(m_new == -INFINITY) m_new = 0.0f;
                            float alpha = std::exp(m_local[h] - m_new);
                            float beta = std::exp(m_block - m_new);
                            
                            __m512 alpha_v = _mm512_set1_ps(alpha);
                            for(int d=0; d<512; d+=16) 
                                _mm512_store_ps(&acc_local[h][d], _mm512_mul_ps(_mm512_load_ps(&acc_local[h][d]), alpha_v));

                            float l_block = 0.0f;
                            
                            for(int tok=0; tok<64; tok+=16) {
                                __m512 prob_v = _mm512_setzero_ps();
                                if (tok < valid_tokens) {
                                    __m512 s_vec = _mm512_load_ps(&scores[h][tok]);
                                    __m512 m_vec = _mm512_set1_ps(m_block);
                                    __m512 b_vec = _mm512_set1_ps(beta);
                                    prob_v = _mm512_mul_ps(fast_exp_512(_mm512_sub_ps(s_vec, m_vec)), b_vec);
                                    l_block += _mm512_reduce_add_ps(prob_v);
                                }
                                
                                __m512i int_val = _mm512_castps_si512(prob_v);
                                __m512i shifted = _mm512_srli_epi32(int_val, 16);
                                __m256i bf16_vals = _mm512_cvtepi32_epi16(shifted);
                                _mm256_storeu_si256((__m256i*)&scores_bf16[h][tok], bf16_vals);
                            }
                            m_local[h] = m_new;
                            l_local[h] = (l_local[h] * alpha) + l_block;
                        }
                        t_soft_cast[tid] += (omp_get_wtime() - t2);

                        // ⏱️ 4. V-PACKING (VNNI Format for Tokens)
                        double t3 = omp_get_wtime();
                        for (int d_blk = 0; d_blk < 512; d_blk += 16) {
                            for (int t_blk = 0; t_blk < 64; t_blk += 32) {
                                bf16* dst = &v_pack[(d_blk / 16) * 1024 + (t_blk / 32) * 512];
                                for (int r = 0; r < 16; r++) {
                                    int tok = t_blk + r * 2;
                                    for (int c = 0; c < 16; c++) {
                                        bf16 val0 = (tok < valid_tokens) ? kv_block_ptr[tok * D + d_blk + c] : 0;
                                        bf16 val1 = (tok + 1 < valid_tokens) ? kv_block_ptr[(tok + 1) * D + d_blk + c] : 0;
                                        dst[r * 32 + c * 2 + 0] = val0;
                                        dst[r * 32 + c * 2 + 1] = val1;
                                    }
                                }
                            }
                        }
                        t_vpack[tid] += (omp_get_wtime() - t3);

                        // ⏱️ 5. P * V (AMX)
                        double t4 = omp_get_wtime();
                        
                        // Load P (16 Heads x 64 Tokens) cleanly into two tiles using 128-byte stride
                        _tile_loadd(4, &scores_bf16[0][0], 128);  // Tokens 0-31
                        _tile_loadd(5, &scores_bf16[0][32], 128); // Tokens 32-63
                        
                        for(int d=0; d<512; d+=16) {
                            _tile_zero(0); // Accumulate output features
                            
                            bf16* v_ptr_t0 = &v_pack[(d / 16) * 1024];
                            bf16* v_ptr_t32 = &v_pack[(d / 16) * 1024 + 512];
                            
                            _tile_loadd(6, v_ptr_t0, 64);   // Load Tokens 0-31
                            _tile_dpbf16ps(0, 4, 6);
                            
                            _tile_loadd(7, v_ptr_t32, 64);  // Load Tokens 32-63
                            _tile_dpbf16ps(0, 5, 7);
                            
                            alignas(64) float delta_o[16][16];
                            _tile_stored(0, delta_o, 64);
                            
                            // Add perfectly aligned output directly into AVX-512 state
                            for(int h=0; h<16; h++) {
                                __m512 acc = _mm512_load_ps(&acc_local[h][d]);
                                __m512 d0  = _mm512_load_ps(&delta_o[h][0]);
                                acc = _mm512_add_ps(acc, d0);
                                _mm512_store_ps(&acc_local[h][d], acc);
                            }
                        }
                        t_amx_v[tid] += (omp_get_wtime() - t4);
                    }

                    int state_idx_base = (((b * H_GROUPS) + h_g) * NUM_CHUNKS + chunk) * HEAD_GROUP;
                    for (int h = 0; h < 16; h++) {
                        global_m[state_idx_base + h] = m_local[h];
                        global_l[state_idx_base + h] = l_local[h];
                        for (int d = 0; d < 512; d++) {
                            global_acc[(state_idx_base + h) * Dv + d] = acc_local[h][d];
                        }
                    }
                } 
            } 
        } 
        t_map_total[tid] = omp_get_wtime() - map_start;

        // ⏱️ TIMER: BARRIER
        double b_start = omp_get_wtime();
        #pragma omp barrier
        t_barrier[tid] = omp_get_wtime() - b_start;

        // ==========================================
        // PHASE 2: REDUCE PHASE 
        // ==========================================
        double r_start = omp_get_wtime();
        #pragma omp for collapse(2) schedule(dynamic, 1)
        for (int b = 0; b < B; b++) {
            for (int h_g = 0; h_g < H_GROUPS; h_g++) {
                int h_start = h_g * HEAD_GROUP;
                for (int h = 0; h < 16; h++) {
                    float m_final = -INFINITY;
                    float l_final = 0.0f;
                    alignas(64) float acc_final[512] = {0};

                    for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {
                        int state_idx = (((b * H_GROUPS) + h_g) * NUM_CHUNKS + chunk) * HEAD_GROUP + h;
                        float m_chunk = global_m[state_idx];
                        float l_chunk = global_l[state_idx];
                        float* acc_chunk = &global_acc[state_idx * Dv];

                        if (m_chunk == -INFINITY) continue;

                        float m_new = std::max(m_final, m_chunk);
                        float alpha_final = std::exp(m_final - m_new);
                        float alpha_chunk = std::exp(m_chunk - m_new);

                        __m512 a_f_v = _mm512_set1_ps(alpha_final);
                        __m512 a_c_v = _mm512_set1_ps(alpha_chunk);

                        for (int d = 0; d < 512; d+=16) {
                            __m512 acc_f = _mm512_load_ps(&acc_final[d]);
                            __m512 acc_c = _mm512_loadu_ps(&acc_chunk[d]);
                            acc_f = _mm512_mul_ps(acc_f, a_f_v);
                            acc_f = _mm512_fmadd_ps(acc_c, a_c_v, acc_f);
                            _mm512_store_ps(&acc_final[d], acc_f);
                        }

                        l_final = (l_final * alpha_final) + (l_chunk * alpha_chunk);
                        m_final = m_new;
                    }

                    bf16* out_ptr = Out + (b * H * Dv) + ((h_start + h) * Dv);
                    float denom = (l_final == 0.0f) ? 1e-10f : l_final;
                    __m512 denom_v = _mm512_set1_ps(denom);
                    
                    for(int d=0; d<512; d+=16) {
                        __m512 acc = _mm512_load_ps(&acc_final[d]);
                        store_fp32_as_bf16(out_ptr + d, _mm512_div_ps(acc, denom_v));
                    }
                }
            }
        }
        t_reduce[tid] = omp_get_wtime() - r_start;
    } // End Parallel

    // --- AGGREGATE STATS ---
    double max_kpack = 0, max_qkt = 0, max_soft = 0, max_vpack = 0, max_amxv = 0, max_map = 0, max_bar = 0, max_red = 0;
    int active_threads = 0;
    
    for (int i = 0; i < max_threads; i++) {
        if (t_map_total[i] > 0) { 
            active_threads++;
            max_kpack = std::max(max_kpack, t_kpack[i]);
            max_qkt = std::max(max_qkt, t_qkt[i]);
            max_soft = std::max(max_soft, t_soft_cast[i]);
            max_vpack = std::max(max_vpack, t_vpack[i]);
            max_amxv = std::max(max_amxv, t_amx_v[i]);
            max_map = std::max(max_map, t_map_total[i]);
            max_bar = std::max(max_bar, t_barrier[i]);
            max_red = std::max(max_red, t_reduce[i]);
        }
    }

    if (active_threads > 0) {
        printf("\n========= FULL AMX PIPELINE PROFILING ========\n");
        printf("COMPONENT              | MAX (CRITICAL) TIME\n");
        printf("---------------------------------------------------\n");
        printf("1. K-Packing (L1 VNNI) | %8.2f ms\n", max_kpack*1000.0);
        printf("2. AMX (Q * K^T)       | %8.2f ms\n", max_qkt*1000.0);
        printf("3. Softmax+AVX+Cast    | %8.2f ms\n", max_soft*1000.0);
        printf("4. V-Packing (L1 VNNI) | %8.2f ms\n", max_vpack*1000.0);
        printf("5. AMX (Scores * V)    | %8.2f ms\n", max_amxv*1000.0);
        printf("---------------------------------------------------\n");
        printf("Total Map Phase        | %8.2f ms\n", max_map*1000.0);
        printf("Barrier Wait Time      | %8.2f ms\n", max_bar*1000.0);
        printf("Phase 2 (Reduce)       | %8.2f ms\n", max_red*1000.0);
        printf("===================================================\n\n");
    }
}

void dense_attention_amx_flat_timers7(
    const bf16* Q, 
    const bf16* KV_flat, 
    const int32_t* SeqLens,
    bf16* Out,
    int B, int H, int D, int BlockSize, float scale,
    size_t kv_stride 
) {
    int Dv = 512;
    int HEAD_GROUP = 16;
    int H_GROUPS = H / HEAD_GROUP;
    int NUM_CHUNKS = 12; // Optimized for 72 cores (8 * 9 = 72)

    int num_states = B * H_GROUPS * NUM_CHUNKS * HEAD_GROUP;
    std::vector<float> global_m(num_states, -INFINITY);
    std::vector<float> global_l(num_states, 0.0f);
    std::vector<float> global_acc(num_states * Dv, 0.0f);

    int max_threads = omp_get_max_threads();
    std::vector<double> t_kpack(max_threads, 0.0);
    std::vector<double> t_qkt(max_threads, 0.0);
    std::vector<double> t_soft_cast(max_threads, 0.0);
    std::vector<double> t_vpack(max_threads, 0.0);
    std::vector<double> t_amx_v(max_threads, 0.0);
    std::vector<double> t_map_total(max_threads, 0.0);

    #pragma omp parallel 
    {
        int tid = omp_get_thread_num();
        init_amx(); 

        static tile_config_t cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.palette_id = 1;
        cfg.start_row = 0;
        for(int i=0; i<8; i++) {
            cfg.rows[i] = 16; 
            cfg.colsb[i] = 64;
        }
        _tile_loadconfig(&cfg);

        // L1 Cache Scratchpads for AMX VNNI Layout
        alignas(64) bf16 k_pack[64 * 576];
        alignas(64) bf16 v_pack[64 * 512]; 

        double map_start = omp_get_wtime();

        #pragma omp for collapse(3) schedule(dynamic, 1) nowait
        for (int b = 0; b < B; b++) {
            for (int h_g = 0; h_g < H_GROUPS; h_g++) {
                for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {
                    
                    int h_start = h_g * HEAD_GROUP;
                    const bf16* q_ptr_base = Q + (b * H * D) + (h_start * D);
                    int seq_len = SeqLens[b];
                    int num_blocks = (seq_len + BlockSize - 1) / BlockSize;

                    int blocks_per_chunk = (num_blocks + NUM_CHUNKS - 1) / NUM_CHUNKS;
                    int block_start = chunk * blocks_per_chunk;
                    int block_end = std::min(block_start + blocks_per_chunk, num_blocks);

                    float m_local[16];
                    float l_local[16];
                    alignas(64) float acc_local[16][512]; 

                    for(int h=0; h<16; h++) {
                        m_local[h] = -INFINITY;
                        l_local[h] = 0.0f;
                        for(int d=0; d<512; d+=16) _mm512_store_ps(&acc_local[h][d], _mm512_setzero_ps());
                    }

                    const bf16* kv_batch_start = KV_flat + ((size_t)b * kv_stride * D);

                    for (int i = block_start; i < block_end; i++) {
                        const bf16* kv_block_ptr = kv_batch_start + ((size_t)i * BlockSize * D);
                        int valid_tokens = (i < num_blocks - 1) ? 64 : (seq_len - i*64);

                        // ⏱️ 1. K-PACKING
                        // double t0 = omp_get_wtime();
                        for(int t=0; t<64; t+=16) {
                            for(int d=0; d<576; d+=32) {
                                pack_k_chunk_amx(kv_block_ptr + t*D + d, &k_pack[t*576 + d*16], D);
                            }
                        }
                        // t_kpack[tid] += (omp_get_wtime() - t0);

                        // ⏱️ 2. Q * K^T (AMX)
                        // double t1 = omp_get_wtime();
                        _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3); 
                        for (int k=0; k<576; k+=32) {
                            _tile_loadd(4, q_ptr_base + k, 1152);
                            _tile_loadd(5, &k_pack[0*576 + k*16], 64);   _tile_dpbf16ps(0, 4, 5);
                            _tile_loadd(6, &k_pack[16*576 + k*16], 64);  _tile_dpbf16ps(1, 4, 6);
                            _tile_loadd(7, &k_pack[32*576 + k*16], 64);  _tile_dpbf16ps(2, 4, 7);
                            _tile_loadd(5, &k_pack[48*576 + k*16], 64);  _tile_dpbf16ps(3, 4, 5);
                        }
                        alignas(64) float scores[16][64];
                        _tile_stored(0, &scores[0][0],  256);
                        _tile_stored(1, &scores[0][16], 256);
                        _tile_stored(2, &scores[0][32], 256);
                        _tile_stored(3, &scores[0][48], 256);
                        // t_qkt[tid] += (omp_get_wtime() - t1);

                        // // ⏱️ 3. SOFTMAX + CAST
                        // double t2 = omp_get_wtime();
                        alignas(64) bf16 scores_bf16[16][64];

                        for(int h=0; h<16; h++) {
                            float m_block = -INFINITY;
                            for(int tok=0; tok<valid_tokens; tok++) {
                                scores[h][tok] *= scale;
                                if (scores[h][tok] > m_block) m_block = scores[h][tok];
                            }
                            
                            float m_new = std::max(m_local[h], m_block);
                            if(m_new == -INFINITY) m_new = 0.0f;
                            float alpha = std::exp(m_local[h] - m_new);
                            float beta = std::exp(m_block - m_new);
                            
                            __m512 alpha_v = _mm512_set1_ps(alpha);
                            for(int d=0; d<512; d+=16) 
                                _mm512_store_ps(&acc_local[h][d], _mm512_mul_ps(_mm512_load_ps(&acc_local[h][d]), alpha_v));

                            float l_block = 0.0f;
                            
                            for(int tok=0; tok<64; tok+=16) {
                                __m512 prob_v = _mm512_setzero_ps();
                                if (tok < valid_tokens) {
                                    __m512 s_vec = _mm512_load_ps(&scores[h][tok]);
                                    __m512 m_vec = _mm512_set1_ps(m_block);
                                    __m512 b_vec = _mm512_set1_ps(beta);
                                    prob_v = _mm512_mul_ps(fast_exp_512(_mm512_sub_ps(s_vec, m_vec)), b_vec);
                                    l_block += _mm512_reduce_add_ps(prob_v);
                                }
                                
                                __m512i int_val = _mm512_castps_si512(prob_v);
                                __m512i shifted = _mm512_srli_epi32(int_val, 16);
                                __m256i bf16_vals = _mm512_cvtepi32_epi16(shifted);
                                _mm256_storeu_si256((__m256i*)&scores_bf16[h][tok], bf16_vals);
                            }
                            m_local[h] = m_new;
                            l_local[h] = (l_local[h] * alpha) + l_block;
                        }
                        // t_soft_cast[tid] += (omp_get_wtime() - t2);

                        // // ⏱️ 4. V-PACKING (VNNI Format for Tokens)
                        // double t3 = omp_get_wtime();
                        for (int d_blk = 0; d_blk < 512; d_blk += 16) {
                            for (int t_blk = 0; t_blk < 64; t_blk += 32) {
                                bf16* dst = &v_pack[(d_blk / 16) * 1024 + (t_blk / 32) * 512];
                                for (int r = 0; r < 16; r++) {
                                    int tok = t_blk + r * 2;
                                    for (int c = 0; c < 16; c++) {
                                        bf16 val0 = (tok < valid_tokens) ? kv_block_ptr[tok * D + d_blk + c] : 0;
                                        bf16 val1 = (tok + 1 < valid_tokens) ? kv_block_ptr[(tok + 1) * D + d_blk + c] : 0;
                                        dst[r * 32 + c * 2 + 0] = val0;
                                        dst[r * 32 + c * 2 + 1] = val1;
                                    }
                                }
                            }
                        }
                        // t_vpack[tid] += (omp_get_wtime() - t3);

                        // ⏱️ 5. P * V (AMX)
                        // double t4 = omp_get_wtime();
                        
                        // Load P (16 Heads x 64 Tokens) cleanly into two tiles using 128-byte stride
                        _tile_loadd(4, &scores_bf16[0][0], 128);  // Tokens 0-31
                        _tile_loadd(5, &scores_bf16[0][32], 128); // Tokens 32-63
                        
                        for(int d=0; d<512; d+=16) {
                            _tile_zero(0); // Accumulate output features
                            
                            bf16* v_ptr_t0 = &v_pack[(d / 16) * 1024];
                            bf16* v_ptr_t32 = &v_pack[(d / 16) * 1024 + 512];
                            
                            _tile_loadd(6, v_ptr_t0, 64);   // Load Tokens 0-31
                            _tile_dpbf16ps(0, 4, 6);
                            
                            _tile_loadd(7, v_ptr_t32, 64);  // Load Tokens 32-63
                            _tile_dpbf16ps(0, 5, 7);
                            
                            alignas(64) float delta_o[16][16];
                            _tile_stored(0, delta_o, 64);
                            
                            // Add perfectly aligned output directly into AVX-512 state
                            for(int h=0; h<16; h++) {
                                __m512 acc = _mm512_load_ps(&acc_local[h][d]);
                                __m512 d0  = _mm512_load_ps(&delta_o[h][0]);
                                acc = _mm512_add_ps(acc, d0);
                                _mm512_store_ps(&acc_local[h][d], acc);
                            }
                        }
                        // t_amx_v[tid] += (omp_get_wtime() - t4);
                    }

                    int state_idx_base = (((b * H_GROUPS) + h_g) * NUM_CHUNKS + chunk) * HEAD_GROUP;
                    for (int h = 0; h < 16; h++) {
                        global_m[state_idx_base + h] = m_local[h];
                        global_l[state_idx_base + h] = l_local[h];
                        for (int d = 0; d < 512; d++) {
                            global_acc[(state_idx_base + h) * Dv + d] = acc_local[h][d];
                        }
                    }
                } 
            } 
        } 
        // t_map_total[tid] = omp_get_wtime() - map_start;

        #pragma omp barrier

        // ==========================================
        // PHASE 2: REDUCE PHASE 
        // ==========================================
        #pragma omp for collapse(2) schedule(dynamic, 1)
        for (int b = 0; b < B; b++) {
            for (int h_g = 0; h_g < H_GROUPS; h_g++) {
                int h_start = h_g * HEAD_GROUP;
                for (int h = 0; h < 16; h++) {
                    float m_final = -INFINITY;
                    float l_final = 0.0f;
                    alignas(64) float acc_final[512] = {0};

                    for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {
                        int state_idx = (((b * H_GROUPS) + h_g) * NUM_CHUNKS + chunk) * HEAD_GROUP + h;
                        float m_chunk = global_m[state_idx];
                        float l_chunk = global_l[state_idx];
                        float* acc_chunk = &global_acc[state_idx * Dv];

                        if (m_chunk == -INFINITY) continue;

                        float m_new = std::max(m_final, m_chunk);
                        float alpha_final = std::exp(m_final - m_new);
                        float alpha_chunk = std::exp(m_chunk - m_new);

                        __m512 a_f_v = _mm512_set1_ps(alpha_final);
                        __m512 a_c_v = _mm512_set1_ps(alpha_chunk);

                        for (int d = 0; d < 512; d+=16) {
                            __m512 acc_f = _mm512_load_ps(&acc_final[d]);
                            __m512 acc_c = _mm512_loadu_ps(&acc_chunk[d]);
                            acc_f = _mm512_mul_ps(acc_f, a_f_v);
                            acc_f = _mm512_fmadd_ps(acc_c, a_c_v, acc_f);
                            _mm512_store_ps(&acc_final[d], acc_f);
                        }

                        l_final = (l_final * alpha_final) + (l_chunk * alpha_chunk);
                        m_final = m_new;
                    }

                    bf16* out_ptr = Out + (b * H * Dv) + ((h_start + h) * Dv);
                    float denom = (l_final == 0.0f) ? 1e-10f : l_final;
                    __m512 denom_v = _mm512_set1_ps(denom);
                    
                    for(int d=0; d<512; d+=16) {
                        __m512 acc = _mm512_load_ps(&acc_final[d]);
                        store_fp32_as_bf16(out_ptr + d, _mm512_div_ps(acc, denom_v));
                    }
                }
            }
        }
    }

    // // --- AGGREGATE STATS ---
    // double max_kpack = 0, max_qkt = 0, max_soft = 0, max_vpack = 0, max_amxv = 0, max_map = 0;
    // int active_threads = 0;
    // for (int i = 0; i < max_threads; i++) {
    //     if (t_map_total[i] > 0) { 
    //         active_threads++;
    //         max_kpack = std::max(max_kpack, t_kpack[i]);
    //         max_qkt = std::max(max_qkt, t_qkt[i]);
    //         max_soft = std::max(max_soft, t_soft_cast[i]);
    //         max_vpack = std::max(max_vpack, t_vpack[i]);
    //         max_amxv = std::max(max_amxv, t_amx_v[i]);
    //         max_map = std::max(max_map, t_map_total[i]);
    //     }
    // }

    // if (active_threads > 0) {
    //     printf("\n========= FULL AMX PIPELINE PROFILING ========\n");
    //     printf("COMPONENT              | MAX (CRITICAL) TIME\n");
    //     printf("---------------------------------------------------\n");
    //     printf("1. K-Packing (L1 VNNI) | %8.2f ms\n", max_kpack*1000.0);
    //     printf("2. AMX (Q * K^T)       | %8.2f ms\n", max_qkt*1000.0);
    //     printf("3. Softmax+AVX+Cast    | %8.2f ms\n", max_soft*1000.0);
    //     printf("4. V-Packing (L1 VNNI) | %8.2f ms\n", max_vpack*1000.0);
    //     printf("5. AMX (Scores * V)    | %8.2f ms\n", max_amxv*1000.0);
    //     printf("---------------------------------------------------\n");
    //     printf("Total Map Phase        | %8.2f ms\n", max_map*1000.0);
    //     printf("===================================================\n\n");
    // }
}

void dense_attention_amx_profile_phases_only(
    const bf16* Q, 
    const bf16* KV_flat, 
    const int32_t* SeqLens,
    bf16* Out,
    int B, int H, int D, int BlockSize, float scale,
    size_t kv_stride 
) {
    int Dv = 512;
    int HEAD_GROUP = 16;
    int H_GROUPS = H / HEAD_GROUP;
    int NUM_CHUNKS = 12; 

    int num_states = B * H_GROUPS * NUM_CHUNKS * HEAD_GROUP;
    std::vector<float> global_m(num_states, -INFINITY);
    std::vector<float> global_l(num_states, 0.0f);
    std::vector<float> global_acc(num_states * Dv, 0.0f);

    int max_threads = omp_get_max_threads();
    std::vector<double> t_map_total(max_threads, 0.0);
    std::vector<double> t_barrier(max_threads, 0.0);
    std::vector<double> t_reduce(max_threads, 0.0);

    #pragma omp parallel 
    {
        int tid = omp_get_thread_num();
        init_amx(); 

        static tile_config_t cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.palette_id = 1;
        cfg.start_row = 0;
        for(int i=0; i<8; i++) {
            cfg.rows[i] = 16; 
            cfg.colsb[i] = 64;
        }
        _tile_loadconfig(&cfg);

        alignas(64) bf16 k_pack[64 * 576];
        alignas(64) bf16 v_pack[64 * 512]; 

        // ⏱️ TIMER: MAP PHASE START
        double map_start = omp_get_wtime();

        #pragma omp for collapse(3) schedule(dynamic, 1) nowait
        for (int b = 0; b < B; b++) {
            for (int h_g = 0; h_g < H_GROUPS; h_g++) {
                for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {
                    
                    int h_start = h_g * HEAD_GROUP;
                    const bf16* q_ptr_base = Q + (b * H * D) + (h_start * D);
                    int seq_len = SeqLens[b];
                    int num_blocks = (seq_len + BlockSize - 1) / BlockSize;

                    int blocks_per_chunk = (num_blocks + NUM_CHUNKS - 1) / NUM_CHUNKS;
                    int block_start = chunk * blocks_per_chunk;
                    int block_end = std::min(block_start + blocks_per_chunk, num_blocks);

                    float m_local[16];
                    float l_local[16];
                    alignas(64) float acc_local[16][512]; 

                    for(int h=0; h<16; h++) {
                        m_local[h] = -INFINITY;
                        l_local[h] = 0.0f;
                        for(int d=0; d<512; d+=16) _mm512_store_ps(&acc_local[h][d], _mm512_setzero_ps());
                    }

                    const bf16* kv_batch_start = KV_flat + ((size_t)b * kv_stride * D);

                    for (int i = block_start; i < block_end; i++) {
                        const bf16* kv_block_ptr = kv_batch_start + ((size_t)i * BlockSize * D);
                        int valid_tokens = (i < num_blocks - 1) ? 64 : (seq_len - i*64);

                        // 1. K-PACKING (No Timer)
                        for(int t=0; t<64; t+=16) {
                            for(int d=0; d<576; d+=32) {
                                pack_k_chunk_amx(kv_block_ptr + t*D + d, &k_pack[t*576 + d*16], D);
                            }
                        }

                        // 2. Q * K^T (No Timer)
                        _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3); 
                        for (int k=0; k<576; k+=32) {
                            _tile_loadd(4, q_ptr_base + k, 1152);
                            _tile_loadd(5, &k_pack[0*576 + k*16], 64);   _tile_dpbf16ps(0, 4, 5);
                            _tile_loadd(6, &k_pack[16*576 + k*16], 64);  _tile_dpbf16ps(1, 4, 6);
                            _tile_loadd(7, &k_pack[32*576 + k*16], 64);  _tile_dpbf16ps(2, 4, 7);
                            _tile_loadd(5, &k_pack[48*576 + k*16], 64);  _tile_dpbf16ps(3, 4, 5);
                        }
                        alignas(64) float scores[16][64];
                        _tile_stored(0, &scores[0][0],  256);
                        _tile_stored(1, &scores[0][16], 256);
                        _tile_stored(2, &scores[0][32], 256);
                        _tile_stored(3, &scores[0][48], 256);

                        // 3. SOFTMAX + CAST (No Timer)
                        alignas(64) bf16 scores_bf16[16][64];
                        for(int h=0; h<16; h++) {
                            float m_block = -INFINITY;
                            for(int tok=0; tok<valid_tokens; tok++) {
                                scores[h][tok] *= scale;
                                if (scores[h][tok] > m_block) m_block = scores[h][tok];
                            }
                            
                            float m_new = std::max(m_local[h], m_block);
                            if(m_new == -INFINITY) m_new = 0.0f;
                            float alpha = std::exp(m_local[h] - m_new);
                            float beta = std::exp(m_block - m_new);
                            
                            __m512 alpha_v = _mm512_set1_ps(alpha);
                            for(int d=0; d<512; d+=16) 
                                _mm512_store_ps(&acc_local[h][d], _mm512_mul_ps(_mm512_load_ps(&acc_local[h][d]), alpha_v));

                            float l_block = 0.0f;
                            
                            for(int tok=0; tok<64; tok+=16) {
                                __m512 prob_v = _mm512_setzero_ps();
                                if (tok < valid_tokens) {
                                    __m512 s_vec = _mm512_load_ps(&scores[h][tok]);
                                    __m512 m_vec = _mm512_set1_ps(m_block);
                                    __m512 b_vec = _mm512_set1_ps(beta);
                                    prob_v = _mm512_mul_ps(fast_exp_512(_mm512_sub_ps(s_vec, m_vec)), b_vec);
                                    l_block += _mm512_reduce_add_ps(prob_v);
                                }
                                
                                __m512i int_val = _mm512_castps_si512(prob_v);
                                __m512i shifted = _mm512_srli_epi32(int_val, 16);
                                __m256i bf16_vals = _mm512_cvtepi32_epi16(shifted);
                                _mm256_storeu_si256((__m256i*)&scores_bf16[h][tok], bf16_vals);
                            }
                            m_local[h] = m_new;
                            l_local[h] = (l_local[h] * alpha) + l_block;
                        }

                        // 4. V-PACKING (No Timer)
                        for (int d_blk = 0; d_blk < 512; d_blk += 16) {
                            for (int t_blk = 0; t_blk < 64; t_blk += 32) {
                                bf16* dst = &v_pack[(d_blk / 16) * 1024 + (t_blk / 32) * 512];
                                for (int r = 0; r < 16; r++) {
                                    int tok = t_blk + r * 2;
                                    for (int c = 0; c < 16; c++) {
                                        bf16 val0 = (tok < valid_tokens) ? kv_block_ptr[tok * D + d_blk + c] : 0;
                                        bf16 val1 = (tok + 1 < valid_tokens) ? kv_block_ptr[(tok + 1) * D + d_blk + c] : 0;
                                        dst[r * 32 + c * 2 + 0] = val0;
                                        dst[r * 32 + c * 2 + 1] = val1;
                                    }
                                }
                            }
                        }

                        // 5. P * V (AMX) (No Timer)
                        _tile_loadd(4, &scores_bf16[0][0], 128); 
                        _tile_loadd(5, &scores_bf16[0][32], 128);
                        
                        for(int d=0; d<512; d+=16) {
                            _tile_zero(0);
                            bf16* v_ptr_t0 = &v_pack[(d / 16) * 1024];
                            bf16* v_ptr_t32 = &v_pack[(d / 16) * 1024 + 512];
                            
                            _tile_loadd(6, v_ptr_t0, 64);   _tile_dpbf16ps(0, 4, 6);
                            _tile_loadd(7, v_ptr_t32, 64);  _tile_dpbf16ps(0, 5, 7);
                            
                            alignas(64) float delta_o[16][16];
                            _tile_stored(0, delta_o, 64);
                            
                            for(int h=0; h<16; h++) {
                                __m512 acc = _mm512_load_ps(&acc_local[h][d]);
                                __m512 d0  = _mm512_load_ps(&delta_o[h][0]);
                                _mm512_store_ps(&acc_local[h][d], _mm512_add_ps(acc, d0));
                            }
                        }
                    }

                    int state_idx_base = (((b * H_GROUPS) + h_g) * NUM_CHUNKS + chunk) * HEAD_GROUP;
                    for (int h = 0; h < 16; h++) {
                        global_m[state_idx_base + h] = m_local[h];
                        global_l[state_idx_base + h] = l_local[h];
                        for (int d = 0; d < 512; d++) {
                            global_acc[(state_idx_base + h) * Dv + d] = acc_local[h][d];
                        }
                    }
                } 
            } 
        } 
        t_map_total[tid] = omp_get_wtime() - map_start;

        // ⏱️ TIMER: BARRIER
        double b_start = omp_get_wtime();
        #pragma omp barrier
        t_barrier[tid] = omp_get_wtime() - b_start;

        // ⏱️ TIMER: REDUCE PHASE
        double r_start = omp_get_wtime();
        #pragma omp for collapse(2) schedule(dynamic, 1)
        for (int b = 0; b < B; b++) {
            for (int h_g = 0; h_g < H_GROUPS; h_g++) {
                int h_start = h_g * HEAD_GROUP;
                for (int h = 0; h < 16; h++) {
                    float m_final = -INFINITY;
                    float l_final = 0.0f;
                    alignas(64) float acc_final[512] = {0};

                    for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {
                        int state_idx = (((b * H_GROUPS) + h_g) * NUM_CHUNKS + chunk) * HEAD_GROUP + h;
                        float m_chunk = global_m[state_idx];
                        float l_chunk = global_l[state_idx];
                        float* acc_chunk = &global_acc[state_idx * Dv];

                        if (m_chunk == -INFINITY) continue;

                        float m_new = std::max(m_final, m_chunk);
                        float alpha_final = std::exp(m_final - m_new);
                        float alpha_chunk = std::exp(m_chunk - m_new);

                        __m512 a_f_v = _mm512_set1_ps(alpha_final);
                        __m512 a_c_v = _mm512_set1_ps(alpha_chunk);

                        for (int d = 0; d < 512; d+=16) {
                            __m512 acc_f = _mm512_load_ps(&acc_final[d]);
                            __m512 acc_c = _mm512_loadu_ps(&acc_chunk[d]);
                            acc_f = _mm512_mul_ps(acc_f, a_f_v);
                            acc_f = _mm512_fmadd_ps(acc_c, a_c_v, acc_f);
                            _mm512_store_ps(&acc_final[d], acc_f);
                        }

                        l_final = (l_final * alpha_final) + (l_chunk * alpha_chunk);
                        m_final = m_new;
                    }

                    bf16* out_ptr = Out + (b * H * Dv) + ((h_start + h) * Dv);
                    float denom = (l_final == 0.0f) ? 1e-10f : l_final;
                    __m512 denom_v = _mm512_set1_ps(denom);
                    
                    for(int d=0; d<512; d+=16) {
                        __m512 acc = _mm512_load_ps(&acc_final[d]);
                        store_fp32_as_bf16(out_ptr + d, _mm512_div_ps(acc, denom_v));
                    }
                }
            }
        }
        t_reduce[tid] = omp_get_wtime() - r_start;
    } 

    double max_map = 0, max_bar = 0, max_red = 0;
    int active_threads = 0;
    
    for (int i = 0; i < max_threads; i++) {
        if (t_map_total[i] > 0) { 
            active_threads++;
            max_map = std::max(max_map, t_map_total[i]);
            max_bar = std::max(max_bar, t_barrier[i]);
            max_red = std::max(max_red, t_reduce[i]);
        }
    }

    if (active_threads > 0) {
        printf("\n========= PHASE-ONLY PROFILING ========\n");
        printf("Total Map Phase        | %8.2f ms\n", max_map*1000.0);
        printf("Barrier Wait Time      | %8.2f ms\n", max_bar*1000.0);
        printf("Phase 2 (Reduce)       | %8.2f ms\n", max_red*1000.0);
        printf("=======================================\n\n");
    }
}

void dense_attention_amx_profile_kpack_only(
    const bf16* Q, 
    const bf16* KV_flat, 
    const int32_t* SeqLens,
    bf16* Out,
    int B, int H, int D, int BlockSize, float scale,
    size_t kv_stride 
) {
    // ... [Exact same initialization and arrays as Phase-Level] ...
    int Dv = 512; int HEAD_GROUP = 16; int H_GROUPS = H / HEAD_GROUP; int NUM_CHUNKS = 12; 
    int num_states = B * H_GROUPS * NUM_CHUNKS * HEAD_GROUP;
    std::vector<float> global_m(num_states, -INFINITY);
    std::vector<float> global_l(num_states, 0.0f);
    std::vector<float> global_acc(num_states * Dv, 0.0f);

    int max_threads = omp_get_max_threads();
    std::vector<double> t_kpack(max_threads, 0.0);
    std::vector<double> t_map_total(max_threads, 0.0);
    std::vector<double> t_barrier(max_threads, 0.0);
    std::vector<double> t_reduce(max_threads, 0.0);

    #pragma omp parallel 
    {
        int tid = omp_get_thread_num();
        init_amx(); 
        static tile_config_t cfg; memset(&cfg, 0, sizeof(cfg)); cfg.palette_id = 1; cfg.start_row = 0;
        for(int i=0; i<8; i++) { cfg.rows[i] = 16; cfg.colsb[i] = 64; }
        _tile_loadconfig(&cfg);

        alignas(64) bf16 k_pack[64 * 576];
        alignas(64) bf16 v_pack[64 * 512]; 

        double map_start = omp_get_wtime();

        #pragma omp for collapse(3) schedule(dynamic, 1) nowait
        for (int b = 0; b < B; b++) {
            for (int h_g = 0; h_g < H_GROUPS; h_g++) {
                for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {
                    
                    int h_start = h_g * HEAD_GROUP;
                    const bf16* q_ptr_base = Q + (b * H * D) + (h_start * D);
                    int seq_len = SeqLens[b];
                    int num_blocks = (seq_len + BlockSize - 1) / BlockSize;
                    int blocks_per_chunk = (num_blocks + NUM_CHUNKS - 1) / NUM_CHUNKS;
                    int block_start = chunk * blocks_per_chunk;
                    int block_end = std::min(block_start + blocks_per_chunk, num_blocks);

                    float m_local[16]; float l_local[16]; alignas(64) float acc_local[16][512]; 
                    for(int h=0; h<16; h++) { m_local[h] = -INFINITY; l_local[h] = 0.0f; for(int d=0; d<512; d+=16) _mm512_store_ps(&acc_local[h][d], _mm512_setzero_ps()); }

                    const bf16* kv_batch_start = KV_flat + ((size_t)b * kv_stride * D);

                    for (int i = block_start; i < block_end; i++) {
                        const bf16* kv_block_ptr = kv_batch_start + ((size_t)i * BlockSize * D);
                        int valid_tokens = (i < num_blocks - 1) ? 64 : (seq_len - i*64);

                        // ⏱️ TIMER: K-PACKING ONLY
                        double t0 = omp_get_wtime();
                        for(int t=0; t<64; t+=16) {
                            for(int d=0; d<576; d+=32) {
                                pack_k_chunk_amx(kv_block_ptr + t*D + d, &k_pack[t*576 + d*16], D);
                            }
                        }
                        t_kpack[tid] += (omp_get_wtime() - t0);

                        // ... [Rest of the inner loop computation IDENTICAL to Phase-Level Version (NO TIMERS)] ...
                        // (Q*K^T, Softmax, V-Pack, and P*V go here exactly as written in Phase-Only above)
                        
                        // 2. Q * K^T (No Timer)
                        _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3); 
                        for (int k=0; k<576; k+=32) {
                            _tile_loadd(4, q_ptr_base + k, 1152);
                            _tile_loadd(5, &k_pack[0*576 + k*16], 64);   _tile_dpbf16ps(0, 4, 5);
                            _tile_loadd(6, &k_pack[16*576 + k*16], 64);  _tile_dpbf16ps(1, 4, 6);
                            _tile_loadd(7, &k_pack[32*576 + k*16], 64);  _tile_dpbf16ps(2, 4, 7);
                            _tile_loadd(5, &k_pack[48*576 + k*16], 64);  _tile_dpbf16ps(3, 4, 5);
                        }
                        alignas(64) float scores[16][64];
                        _tile_stored(0, &scores[0][0],  256);
                        _tile_stored(1, &scores[0][16], 256);
                        _tile_stored(2, &scores[0][32], 256);
                        _tile_stored(3, &scores[0][48], 256);

                        // 3. SOFTMAX + CAST (No Timer)
                        alignas(64) bf16 scores_bf16[16][64];
                        for(int h=0; h<16; h++) {
                            float m_block = -INFINITY;
                            for(int tok=0; tok<valid_tokens; tok++) {
                                scores[h][tok] *= scale;
                                if (scores[h][tok] > m_block) m_block = scores[h][tok];
                            }
                            
                            float m_new = std::max(m_local[h], m_block);
                            if(m_new == -INFINITY) m_new = 0.0f;
                            float alpha = std::exp(m_local[h] - m_new);
                            float beta = std::exp(m_block - m_new);
                            
                            __m512 alpha_v = _mm512_set1_ps(alpha);
                            for(int d=0; d<512; d+=16) 
                                _mm512_store_ps(&acc_local[h][d], _mm512_mul_ps(_mm512_load_ps(&acc_local[h][d]), alpha_v));

                            float l_block = 0.0f;
                            for(int tok=0; tok<64; tok+=16) {
                                __m512 prob_v = _mm512_setzero_ps();
                                if (tok < valid_tokens) {
                                    __m512 s_vec = _mm512_load_ps(&scores[h][tok]);
                                    __m512 m_vec = _mm512_set1_ps(m_block);
                                    __m512 b_vec = _mm512_set1_ps(beta);
                                    prob_v = _mm512_mul_ps(fast_exp_512(_mm512_sub_ps(s_vec, m_vec)), b_vec);
                                    l_block += _mm512_reduce_add_ps(prob_v);
                                }
                                __m512i int_val = _mm512_castps_si512(prob_v);
                                __m512i shifted = _mm512_srli_epi32(int_val, 16);
                                __m256i bf16_vals = _mm512_cvtepi32_epi16(shifted);
                                _mm256_storeu_si256((__m256i*)&scores_bf16[h][tok], bf16_vals);
                            }
                            m_local[h] = m_new;
                            l_local[h] = (l_local[h] * alpha) + l_block;
                        }

                        // 4. V-PACKING (No Timer)
                        for (int d_blk = 0; d_blk < 512; d_blk += 16) {
                            for (int t_blk = 0; t_blk < 64; t_blk += 32) {
                                bf16* dst = &v_pack[(d_blk / 16) * 1024 + (t_blk / 32) * 512];
                                for (int r = 0; r < 16; r++) {
                                    int tok = t_blk + r * 2;
                                    for (int c = 0; c < 16; c++) {
                                        bf16 val0 = (tok < valid_tokens) ? kv_block_ptr[tok * D + d_blk + c] : 0;
                                        bf16 val1 = (tok + 1 < valid_tokens) ? kv_block_ptr[(tok + 1) * D + d_blk + c] : 0;
                                        dst[r * 32 + c * 2 + 0] = val0;
                                        dst[r * 32 + c * 2 + 1] = val1;
                                    }
                                }
                            }
                        }

                        // 5. P * V (No Timer)
                        _tile_loadd(4, &scores_bf16[0][0], 128); 
                        _tile_loadd(5, &scores_bf16[0][32], 128);
                        for(int d=0; d<512; d+=16) {
                            _tile_zero(0);
                            bf16* v_ptr_t0 = &v_pack[(d / 16) * 1024];
                            bf16* v_ptr_t32 = &v_pack[(d / 16) * 1024 + 512];
                            _tile_loadd(6, v_ptr_t0, 64);   _tile_dpbf16ps(0, 4, 6);
                            _tile_loadd(7, v_ptr_t32, 64);  _tile_dpbf16ps(0, 5, 7);
                            alignas(64) float delta_o[16][16];
                            _tile_stored(0, delta_o, 64);
                            for(int h=0; h<16; h++) {
                                __m512 acc = _mm512_load_ps(&acc_local[h][d]);
                                __m512 d0  = _mm512_load_ps(&delta_o[h][0]);
                                _mm512_store_ps(&acc_local[h][d], _mm512_add_ps(acc, d0));
                            }
                        }
                    }

                    // ... [State saving IDENTICAL] ...
                    int state_idx_base = (((b * H_GROUPS) + h_g) * NUM_CHUNKS + chunk) * HEAD_GROUP;
                    for (int h = 0; h < 16; h++) {
                        global_m[state_idx_base + h] = m_local[h];
                        global_l[state_idx_base + h] = l_local[h];
                        for (int d = 0; d < 512; d++) {
                            global_acc[(state_idx_base + h) * Dv + d] = acc_local[h][d];
                        }
                    }
                } 
            } 
        } 
        t_map_total[tid] = omp_get_wtime() - map_start;

        // ... [Barrier and Reduce IDENTICAL to Phase-Only Version] ...
        double b_start = omp_get_wtime();
        #pragma omp barrier
        t_barrier[tid] = omp_get_wtime() - b_start;

        double r_start = omp_get_wtime();
        #pragma omp for collapse(2) schedule(dynamic, 1)
        for (int b = 0; b < B; b++) {
            for (int h_g = 0; h_g < H_GROUPS; h_g++) {
                int h_start = h_g * HEAD_GROUP;
                for (int h = 0; h < 16; h++) {
                    float m_final = -INFINITY; float l_final = 0.0f; alignas(64) float acc_final[512] = {0};
                    for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {
                        int state_idx = (((b * H_GROUPS) + h_g) * NUM_CHUNKS + chunk) * HEAD_GROUP + h;
                        float m_chunk = global_m[state_idx]; float l_chunk = global_l[state_idx]; float* acc_chunk = &global_acc[state_idx * Dv];
                        if (m_chunk == -INFINITY) continue;
                        float m_new = std::max(m_final, m_chunk);
                        float alpha_final = std::exp(m_final - m_new); float alpha_chunk = std::exp(m_chunk - m_new);
                        __m512 a_f_v = _mm512_set1_ps(alpha_final); __m512 a_c_v = _mm512_set1_ps(alpha_chunk);
                        for (int d = 0; d < 512; d+=16) {
                            __m512 acc_f = _mm512_load_ps(&acc_final[d]); __m512 acc_c = _mm512_loadu_ps(&acc_chunk[d]);
                            acc_f = _mm512_mul_ps(acc_f, a_f_v); acc_f = _mm512_fmadd_ps(acc_c, a_c_v, acc_f); _mm512_store_ps(&acc_final[d], acc_f);
                        }
                        l_final = (l_final * alpha_final) + (l_chunk * alpha_chunk); m_final = m_new;
                    }
                    bf16* out_ptr = Out + (b * H * Dv) + ((h_start + h) * Dv);
                    float denom = (l_final == 0.0f) ? 1e-10f : l_final; __m512 denom_v = _mm512_set1_ps(denom);
                    for(int d=0; d<512; d+=16) {
                        __m512 acc = _mm512_load_ps(&acc_final[d]); store_fp32_as_bf16(out_ptr + d, _mm512_div_ps(acc, denom_v));
                    }
                }
            }
        }
        t_reduce[tid] = omp_get_wtime() - r_start;
    } 

    double max_kpack = 0, max_map = 0;
    for (int i = 0; i < max_threads; i++) {
        if (t_map_total[i] > 0) { 
            max_kpack = std::max(max_kpack, t_kpack[i]);
            max_map = std::max(max_map, t_map_total[i]);
        }
    }
    printf("\n========= K-PACKING ONLY ========\n");
    printf("1. K-Packing (L1 VNNI) | %8.2f ms\n", max_kpack*1000.0);
    printf("Total Map Phase        | %8.2f ms\n", max_map*1000.0);
    printf("=================================\n\n");
}

void dense_attention_amx_profile_qkt_only(
    const bf16* Q, 
    const bf16* KV_flat, 
    const int32_t* SeqLens,
    bf16* Out,
    int B, int H, int D, int BlockSize, float scale,
    size_t kv_stride 
) {
    int Dv = 512;
    int HEAD_GROUP = 16;
    int H_GROUPS = H / HEAD_GROUP;
    int NUM_CHUNKS = 12; 

    int num_states = B * H_GROUPS * NUM_CHUNKS * HEAD_GROUP;
    std::vector<float> global_m(num_states, -INFINITY);
    std::vector<float> global_l(num_states, 0.0f);
    std::vector<float> global_acc(num_states * Dv, 0.0f);

    int max_threads = omp_get_max_threads();
    
    // --- TIMERS ---
    std::vector<double> t_qkt(max_threads, 0.0);
    std::vector<double> t_map_total(max_threads, 0.0);
    std::vector<double> t_barrier(max_threads, 0.0);
    std::vector<double> t_reduce(max_threads, 0.0);

    #pragma omp parallel 
    {
        int tid = omp_get_thread_num();
        init_amx(); 

        static tile_config_t cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.palette_id = 1;
        cfg.start_row = 0;
        for(int i=0; i<8; i++) {
            cfg.rows[i] = 16; 
            cfg.colsb[i] = 64;
        }
        _tile_loadconfig(&cfg);

        alignas(64) bf16 k_pack[64 * 576];
        alignas(64) bf16 v_pack[64 * 512]; 

        // ⏱️ TIMER: MAP PHASE START
        double map_start = omp_get_wtime();

        #pragma omp for collapse(3) schedule(dynamic, 1) nowait
        for (int b = 0; b < B; b++) {
            for (int h_g = 0; h_g < H_GROUPS; h_g++) {
                for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {
                    
                    int h_start = h_g * HEAD_GROUP;
                    const bf16* q_ptr_base = Q + (b * H * D) + (h_start * D);
                    int seq_len = SeqLens[b];
                    int num_blocks = (seq_len + BlockSize - 1) / BlockSize;

                    int blocks_per_chunk = (num_blocks + NUM_CHUNKS - 1) / NUM_CHUNKS;
                    int block_start = chunk * blocks_per_chunk;
                    int block_end = std::min(block_start + blocks_per_chunk, num_blocks);

                    float m_local[16];
                    float l_local[16];
                    alignas(64) float acc_local[16][512]; 

                    for(int h=0; h<16; h++) {
                        m_local[h] = -INFINITY;
                        l_local[h] = 0.0f;
                        for(int d=0; d<512; d+=16) _mm512_store_ps(&acc_local[h][d], _mm512_setzero_ps());
                    }

                    const bf16* kv_batch_start = KV_flat + ((size_t)b * kv_stride * D);

                    for (int i = block_start; i < block_end; i++) {
                        const bf16* kv_block_ptr = kv_batch_start + ((size_t)i * BlockSize * D);
                        int valid_tokens = (i < num_blocks - 1) ? 64 : (seq_len - i*64);

                        // 1. K-PACKING (No Timer)
                        for(int t=0; t<64; t+=16) {
                            for(int d=0; d<576; d+=32) {
                                pack_k_chunk_amx(kv_block_ptr + t*D + d, &k_pack[t*576 + d*16], D);
                            }
                        }

                        // ⏱️ TIMER: Q * K^T (AMX) ONLY
                        double t1 = omp_get_wtime();
                        
                        _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3); 
                        for (int k=0; k<576; k+=32) {
                            _tile_loadd(4, q_ptr_base + k, 1152);
                            _tile_loadd(5, &k_pack[0*576 + k*16], 64);   _tile_dpbf16ps(0, 4, 5);
                            _tile_loadd(6, &k_pack[16*576 + k*16], 64);  _tile_dpbf16ps(1, 4, 6);
                            _tile_loadd(7, &k_pack[32*576 + k*16], 64);  _tile_dpbf16ps(2, 4, 7);
                            _tile_loadd(5, &k_pack[48*576 + k*16], 64);  _tile_dpbf16ps(3, 4, 5);
                        }
                        alignas(64) float scores[16][64];
                        _tile_stored(0, &scores[0][0],  256);
                        _tile_stored(1, &scores[0][16], 256);
                        _tile_stored(2, &scores[0][32], 256);
                        _tile_stored(3, &scores[0][48], 256);
                        
                        t_qkt[tid] += (omp_get_wtime() - t1);

                        // 3. SOFTMAX + CAST (No Timer)
                        alignas(64) bf16 scores_bf16[16][64];
                        for(int h=0; h<16; h++) {
                            float m_block = -INFINITY;
                            for(int tok=0; tok<valid_tokens; tok++) {
                                scores[h][tok] *= scale;
                                if (scores[h][tok] > m_block) m_block = scores[h][tok];
                            }
                            
                            float m_new = std::max(m_local[h], m_block);
                            if(m_new == -INFINITY) m_new = 0.0f;
                            float alpha = std::exp(m_local[h] - m_new);
                            float beta = std::exp(m_block - m_new);
                            
                            __m512 alpha_v = _mm512_set1_ps(alpha);
                            for(int d=0; d<512; d+=16) 
                                _mm512_store_ps(&acc_local[h][d], _mm512_mul_ps(_mm512_load_ps(&acc_local[h][d]), alpha_v));

                            float l_block = 0.0f;
                            
                            for(int tok=0; tok<64; tok+=16) {
                                __m512 prob_v = _mm512_setzero_ps();
                                if (tok < valid_tokens) {
                                    __m512 s_vec = _mm512_load_ps(&scores[h][tok]);
                                    __m512 m_vec = _mm512_set1_ps(m_block);
                                    __m512 b_vec = _mm512_set1_ps(beta);
                                    prob_v = _mm512_mul_ps(fast_exp_512(_mm512_sub_ps(s_vec, m_vec)), b_vec);
                                    l_block += _mm512_reduce_add_ps(prob_v);
                                }
                                
                                __m512i int_val = _mm512_castps_si512(prob_v);
                                __m512i shifted = _mm512_srli_epi32(int_val, 16);
                                __m256i bf16_vals = _mm512_cvtepi32_epi16(shifted);
                                _mm256_storeu_si256((__m256i*)&scores_bf16[h][tok], bf16_vals);
                            }
                            m_local[h] = m_new;
                            l_local[h] = (l_local[h] * alpha) + l_block;
                        }

                        // 4. V-PACKING (No Timer)
                        for (int d_blk = 0; d_blk < 512; d_blk += 16) {
                            for (int t_blk = 0; t_blk < 64; t_blk += 32) {
                                bf16* dst = &v_pack[(d_blk / 16) * 1024 + (t_blk / 32) * 512];
                                for (int r = 0; r < 16; r++) {
                                    int tok = t_blk + r * 2;
                                    for (int c = 0; c < 16; c++) {
                                        bf16 val0 = (tok < valid_tokens) ? kv_block_ptr[tok * D + d_blk + c] : 0;
                                        bf16 val1 = (tok + 1 < valid_tokens) ? kv_block_ptr[(tok + 1) * D + d_blk + c] : 0;
                                        dst[r * 32 + c * 2 + 0] = val0;
                                        dst[r * 32 + c * 2 + 1] = val1;
                                    }
                                }
                            }
                        }

                        // 5. P * V (AMX) (No Timer)
                        _tile_loadd(4, &scores_bf16[0][0], 128); 
                        _tile_loadd(5, &scores_bf16[0][32], 128);
                        
                        for(int d=0; d<512; d+=16) {
                            _tile_zero(0);
                            bf16* v_ptr_t0 = &v_pack[(d / 16) * 1024];
                            bf16* v_ptr_t32 = &v_pack[(d / 16) * 1024 + 512];
                            
                            _tile_loadd(6, v_ptr_t0, 64);   _tile_dpbf16ps(0, 4, 6);
                            _tile_loadd(7, v_ptr_t32, 64);  _tile_dpbf16ps(0, 5, 7);
                            
                            alignas(64) float delta_o[16][16];
                            _tile_stored(0, delta_o, 64);
                            
                            for(int h=0; h<16; h++) {
                                __m512 acc = _mm512_load_ps(&acc_local[h][d]);
                                __m512 d0  = _mm512_load_ps(&delta_o[h][0]);
                                _mm512_store_ps(&acc_local[h][d], _mm512_add_ps(acc, d0));
                            }
                        }
                    }

                    int state_idx_base = (((b * H_GROUPS) + h_g) * NUM_CHUNKS + chunk) * HEAD_GROUP;
                    for (int h = 0; h < 16; h++) {
                        global_m[state_idx_base + h] = m_local[h];
                        global_l[state_idx_base + h] = l_local[h];
                        for (int d = 0; d < 512; d++) {
                            global_acc[(state_idx_base + h) * Dv + d] = acc_local[h][d];
                        }
                    }
                } 
            } 
        } 
        t_map_total[tid] = omp_get_wtime() - map_start;

        // ⏱️ TIMER: BARRIER
        // double b_start = omp_get_wtime();
        #pragma omp barrier
        // t_barrier[tid] = omp_get_wtime() - b_start;

        // ⏱️ TIMER: REDUCE PHASE
        // double r_start = omp_get_wtime();
        #pragma omp for collapse(2) schedule(dynamic, 1)
        for (int b = 0; b < B; b++) {
            for (int h_g = 0; h_g < H_GROUPS; h_g++) {
                int h_start = h_g * HEAD_GROUP;
                for (int h = 0; h < 16; h++) {
                    float m_final = -INFINITY;
                    float l_final = 0.0f;
                    alignas(64) float acc_final[512] = {0};

                    for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {
                        int state_idx = (((b * H_GROUPS) + h_g) * NUM_CHUNKS + chunk) * HEAD_GROUP + h;
                        float m_chunk = global_m[state_idx];
                        float l_chunk = global_l[state_idx];
                        float* acc_chunk = &global_acc[state_idx * Dv];

                        if (m_chunk == -INFINITY) continue;

                        float m_new = std::max(m_final, m_chunk);
                        float alpha_final = std::exp(m_final - m_new);
                        float alpha_chunk = std::exp(m_chunk - m_new);

                        __m512 a_f_v = _mm512_set1_ps(alpha_final);
                        __m512 a_c_v = _mm512_set1_ps(alpha_chunk);

                        for (int d = 0; d < 512; d+=16) {
                            __m512 acc_f = _mm512_load_ps(&acc_final[d]);
                            __m512 acc_c = _mm512_loadu_ps(&acc_chunk[d]);
                            acc_f = _mm512_mul_ps(acc_f, a_f_v);
                            acc_f = _mm512_fmadd_ps(acc_c, a_c_v, acc_f);
                            _mm512_store_ps(&acc_final[d], acc_f);
                        }

                        l_final = (l_final * alpha_final) + (l_chunk * alpha_chunk);
                        m_final = m_new;
                    }

                    bf16* out_ptr = Out + (b * H * Dv) + ((h_start + h) * Dv);
                    float denom = (l_final == 0.0f) ? 1e-10f : l_final;
                    __m512 denom_v = _mm512_set1_ps(denom);
                    
                    for(int d=0; d<512; d+=16) {
                        __m512 acc = _mm512_load_ps(&acc_final[d]);
                        store_fp32_as_bf16(out_ptr + d, _mm512_div_ps(acc, denom_v));
                    }
                }
            }
        }
        // t_reduce[tid] = omp_get_wtime() - r_start;
    } 

    double max_qkt = 0, max_map = 0, max_bar = 0, max_red = 0;
    int active_threads = 0;
    
    for (int i = 0; i < max_threads; i++) {
        if (t_map_total[i] > 0) { 
            active_threads++;
            max_qkt = std::max(max_qkt, t_qkt[i]);
            max_map = std::max(max_map, t_map_total[i]);
        }
    }

    if (active_threads > 0) {
        printf("\n========= Q*K^T (AMX) ONLY ========\n");
        printf("2. AMX (Q * K^T)       | %8.2f ms\n", max_qkt*1000.0);
        printf("-----------------------------------\n");
        printf("Total Map Phase        | %8.2f ms\n", max_map*1000.0);
        printf("===================================\n\n");
    }
}

void dense_attention_amx_profile_soft_vpack_pv_sum(
    const bf16* Q, 
    const bf16* KV_flat, 
    const int32_t* SeqLens,
    bf16* Out,
    int B, int H, int D, int BlockSize, float scale,
    size_t kv_stride 
) {
    int Dv = 512; int HEAD_GROUP = 16; int H_GROUPS = H / HEAD_GROUP; int NUM_CHUNKS = 12; 
    int num_states = B * H_GROUPS * NUM_CHUNKS * HEAD_GROUP;
    std::vector<float> global_m(num_states, -INFINITY);
    std::vector<float> global_l(num_states, 0.0f);
    std::vector<float> global_acc(num_states * Dv, 0.0f);

    int max_threads = omp_get_max_threads();
    
    // --- TIMERS ---
    std::vector<double> t_combined(max_threads, 0.0); // Softmax + V-Pack + P*V
    std::vector<double> t_map_total(max_threads, 0.0);
    std::vector<double> t_barrier(max_threads, 0.0);
    std::vector<double> t_reduce(max_threads, 0.0);

    #pragma omp parallel 
    {
        int tid = omp_get_thread_num();
        init_amx(); 
        static tile_config_t cfg; memset(&cfg, 0, sizeof(cfg)); cfg.palette_id = 1; cfg.start_row = 0;
        for(int i=0; i<8; i++) { cfg.rows[i] = 16; cfg.colsb[i] = 64; }
        _tile_loadconfig(&cfg);

        alignas(64) bf16 k_pack[64 * 576];
        alignas(64) bf16 v_pack[64 * 512]; 

        double map_start = omp_get_wtime();

        #pragma omp for collapse(3) schedule(dynamic, 1) nowait
        for (int b = 0; b < B; b++) {
            for (int h_g = 0; h_g < H_GROUPS; h_g++) {
                for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {
                    
                    int h_start = h_g * HEAD_GROUP;
                    const bf16* q_ptr_base = Q + (b * H * D) + (h_start * D);
                    int seq_len = SeqLens[b];
                    int num_blocks = (seq_len + BlockSize - 1) / BlockSize;
                    int blocks_per_chunk = (num_blocks + NUM_CHUNKS - 1) / NUM_CHUNKS;
                    int block_start = chunk * blocks_per_chunk;
                    int block_end = std::min(block_start + blocks_per_chunk, num_blocks);

                    float m_local[16]; float l_local[16]; alignas(64) float acc_local[16][512]; 
                    for(int h=0; h<16; h++) { m_local[h] = -INFINITY; l_local[h] = 0.0f; for(int d=0; d<512; d+=16) _mm512_store_ps(&acc_local[h][d], _mm512_setzero_ps()); }

                    const bf16* kv_batch_start = KV_flat + ((size_t)b * kv_stride * D);

                    for (int i = block_start; i < block_end; i++) {
                        const bf16* kv_block_ptr = kv_batch_start + ((size_t)i * BlockSize * D);
                        int valid_tokens = (i < num_blocks - 1) ? 64 : (seq_len - i*64);

                        // 1. K-PACKING (No Timer)
                        for(int t=0; t<64; t+=16) {
                            for(int d=0; d<576; d+=32) {
                                pack_k_chunk_amx(kv_block_ptr + t*D + d, &k_pack[t*576 + d*16], D);
                            }
                        }

                        // 2. Q * K^T (No Timer)
                        _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3); 
                        for (int k=0; k<576; k+=32) {
                            _tile_loadd(4, q_ptr_base + k, 1152);
                            _tile_loadd(5, &k_pack[0*576 + k*16], 64);   _tile_dpbf16ps(0, 4, 5);
                            _tile_loadd(6, &k_pack[16*576 + k*16], 64);  _tile_dpbf16ps(1, 4, 6);
                            _tile_loadd(7, &k_pack[32*576 + k*16], 64);  _tile_dpbf16ps(2, 4, 7);
                            _tile_loadd(5, &k_pack[48*576 + k*16], 64);  _tile_dpbf16ps(3, 4, 5);
                        }
                        alignas(64) float scores[16][64];
                        _tile_stored(0, &scores[0][0],  256);
                        _tile_stored(1, &scores[0][16], 256);
                        _tile_stored(2, &scores[0][32], 256);
                        _tile_stored(3, &scores[0][48], 256);

                        // ⏱️ TIMER: COMBINED SOFTMAX + V-PACK + P*V
                        double t_comb = omp_get_wtime();

                        // 3. SOFTMAX + CAST 
                        alignas(64) bf16 scores_bf16[16][64];
                        for(int h=0; h<16; h++) {
                            float m_block = -INFINITY;
                            for(int tok=0; tok<valid_tokens; tok++) {
                                scores[h][tok] *= scale;
                                if (scores[h][tok] > m_block) m_block = scores[h][tok];
                            }
                            
                            float m_new = std::max(m_local[h], m_block);
                            if(m_new == -INFINITY) m_new = 0.0f;
                            float alpha = std::exp(m_local[h] - m_new);
                            float beta = std::exp(m_block - m_new);
                            
                            __m512 alpha_v = _mm512_set1_ps(alpha);
                            for(int d=0; d<512; d+=16) 
                                _mm512_store_ps(&acc_local[h][d], _mm512_mul_ps(_mm512_load_ps(&acc_local[h][d]), alpha_v));

                            float l_block = 0.0f;
                            for(int tok=0; tok<64; tok+=16) {
                                __m512 prob_v = _mm512_setzero_ps();
                                if (tok < valid_tokens) {
                                    __m512 s_vec = _mm512_load_ps(&scores[h][tok]);
                                    __m512 m_vec = _mm512_set1_ps(m_block);
                                    __m512 b_vec = _mm512_set1_ps(beta);
                                    prob_v = _mm512_mul_ps(fast_exp_512(_mm512_sub_ps(s_vec, m_vec)), b_vec);
                                    l_block += _mm512_reduce_add_ps(prob_v);
                                }
                                __m512i int_val = _mm512_castps_si512(prob_v);
                                __m512i shifted = _mm512_srli_epi32(int_val, 16);
                                __m256i bf16_vals = _mm512_cvtepi32_epi16(shifted);
                                _mm256_storeu_si256((__m256i*)&scores_bf16[h][tok], bf16_vals);
                            }
                            m_local[h] = m_new;
                            l_local[h] = (l_local[h] * alpha) + l_block;
                        }

                        // 4. V-PACKING 
                        for (int d_blk = 0; d_blk < 512; d_blk += 16) {
                            for (int t_blk = 0; t_blk < 64; t_blk += 32) {
                                bf16* dst = &v_pack[(d_blk / 16) * 1024 + (t_blk / 32) * 512];
                                for (int r = 0; r < 16; r++) {
                                    int tok = t_blk + r * 2;
                                    for (int c = 0; c < 16; c++) {
                                        bf16 val0 = (tok < valid_tokens) ? kv_block_ptr[tok * D + d_blk + c] : 0;
                                        bf16 val1 = (tok + 1 < valid_tokens) ? kv_block_ptr[(tok + 1) * D + d_blk + c] : 0;
                                        dst[r * 32 + c * 2 + 0] = val0;
                                        dst[r * 32 + c * 2 + 1] = val1;
                                    }
                                }
                            }
                        }

                        // 5. P * V (AMX) 
                        _tile_loadd(4, &scores_bf16[0][0], 128); 
                        _tile_loadd(5, &scores_bf16[0][32], 128);
                        for(int d=0; d<512; d+=16) {
                            _tile_zero(0);
                            bf16* v_ptr_t0 = &v_pack[(d / 16) * 1024];
                            bf16* v_ptr_t32 = &v_pack[(d / 16) * 1024 + 512];
                            _tile_loadd(6, v_ptr_t0, 64);   _tile_dpbf16ps(0, 4, 6);
                            _tile_loadd(7, v_ptr_t32, 64);  _tile_dpbf16ps(0, 5, 7);
                            alignas(64) float delta_o[16][16];
                            _tile_stored(0, delta_o, 64);
                            for(int h=0; h<16; h++) {
                                __m512 acc = _mm512_load_ps(&acc_local[h][d]);
                                __m512 d0  = _mm512_load_ps(&delta_o[h][0]);
                                _mm512_store_ps(&acc_local[h][d], _mm512_add_ps(acc, d0));
                            }
                        }

                        // End combined timer
                        t_combined[tid] += (omp_get_wtime() - t_comb);
                    }

                    int state_idx_base = (((b * H_GROUPS) + h_g) * NUM_CHUNKS + chunk) * HEAD_GROUP;
                    for (int h = 0; h < 16; h++) {
                        global_m[state_idx_base + h] = m_local[h]; global_l[state_idx_base + h] = l_local[h];
                        for (int d = 0; d < 512; d++) global_acc[(state_idx_base + h) * Dv + d] = acc_local[h][d];
                    }
                } 
            } 
        } 
        t_map_total[tid] = omp_get_wtime() - map_start;

        double b_start = omp_get_wtime();
        #pragma omp barrier
        t_barrier[tid] = omp_get_wtime() - b_start;

        double r_start = omp_get_wtime();
        #pragma omp for collapse(2) schedule(dynamic, 1)
        // ... [Reduce phase logic omitted for brevity, identically maintained] ...
        for (int b = 0; b < B; b++) {
            for (int h_g = 0; h_g < H_GROUPS; h_g++) {
                int h_start = h_g * HEAD_GROUP;
                for (int h = 0; h < 16; h++) {
                    float m_final = -INFINITY; float l_final = 0.0f; alignas(64) float acc_final[512] = {0};
                    for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {
                        int state_idx = (((b * H_GROUPS) + h_g) * NUM_CHUNKS + chunk) * HEAD_GROUP + h;
                        float m_chunk = global_m[state_idx]; float l_chunk = global_l[state_idx]; float* acc_chunk = &global_acc[state_idx * Dv];
                        if (m_chunk == -INFINITY) continue;
                        float m_new = std::max(m_final, m_chunk);
                        float alpha_final = std::exp(m_final - m_new); float alpha_chunk = std::exp(m_chunk - m_new);
                        __m512 a_f_v = _mm512_set1_ps(alpha_final); __m512 a_c_v = _mm512_set1_ps(alpha_chunk);
                        for (int d = 0; d < 512; d+=16) {
                            __m512 acc_f = _mm512_load_ps(&acc_final[d]); __m512 acc_c = _mm512_loadu_ps(&acc_chunk[d]);
                            acc_f = _mm512_mul_ps(acc_f, a_f_v); acc_f = _mm512_fmadd_ps(acc_c, a_c_v, acc_f); _mm512_store_ps(&acc_final[d], acc_f);
                        }
                        l_final = (l_final * alpha_final) + (l_chunk * alpha_chunk); m_final = m_new;
                    }
                    bf16* out_ptr = Out + (b * H * Dv) + ((h_start + h) * Dv);
                    float denom = (l_final == 0.0f) ? 1e-10f : l_final; __m512 denom_v = _mm512_set1_ps(denom);
                    for(int d=0; d<512; d+=16) {
                        __m512 acc = _mm512_load_ps(&acc_final[d]); store_fp32_as_bf16(out_ptr + d, _mm512_div_ps(acc, denom_v));
                    }
                }
            }
        }
        t_reduce[tid] = omp_get_wtime() - r_start;
    } 

    double max_comb = 0, max_map = 0;
    for (int i = 0; i < max_threads; i++) {
        if (t_map_total[i] > 0) { 
            max_comb = std::max(max_comb, t_combined[i]);
            max_map = std::max(max_map, t_map_total[i]);
        }
    }
    printf("\n====== SOFTMAX + V-PACK + P*V =======\n");
    printf("Combined Time          | %8.2f ms\n", max_comb*1000.0);
    printf("-------------------------------------\n");
    printf("Total Map Phase        | %8.2f ms\n", max_map*1000.0);
    printf("=====================================\n\n");
}

void dense_attention_amx_profile_softmax_only(
    const bf16* Q, 
    const bf16* KV_flat, 
    const int32_t* SeqLens,
    bf16* Out,
    int B, int H, int D, int BlockSize, float scale,
    size_t kv_stride 
) {
    int Dv = 512; int HEAD_GROUP = 16; int H_GROUPS = H / HEAD_GROUP; int NUM_CHUNKS = 12; 
    int num_states = B * H_GROUPS * NUM_CHUNKS * HEAD_GROUP;
    std::vector<float> global_m(num_states, -INFINITY); std::vector<float> global_l(num_states, 0.0f); std::vector<float> global_acc(num_states * Dv, 0.0f);
    int max_threads = omp_get_max_threads();
    
    // --- TIMERS ---
    std::vector<double> t_soft(max_threads, 0.0);
    std::vector<double> t_map_total(max_threads, 0.0);
    std::vector<double> t_barrier(max_threads, 0.0);
    std::vector<double> t_reduce(max_threads, 0.0);

    #pragma omp parallel 
    {
        int tid = omp_get_thread_num();
        init_amx(); 
        static tile_config_t cfg; memset(&cfg, 0, sizeof(cfg)); cfg.palette_id = 1; cfg.start_row = 0;
        for(int i=0; i<8; i++) { cfg.rows[i] = 16; cfg.colsb[i] = 64; }
        _tile_loadconfig(&cfg);

        alignas(64) bf16 k_pack[64 * 576]; alignas(64) bf16 v_pack[64 * 512]; 
        double map_start = omp_get_wtime();

        #pragma omp for collapse(3) schedule(dynamic, 1) nowait
        for (int b = 0; b < B; b++) {
            for (int h_g = 0; h_g < H_GROUPS; h_g++) {
                for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {
                    
                    int h_start = h_g * HEAD_GROUP;
                    const bf16* q_ptr_base = Q + (b * H * D) + (h_start * D);
                    int seq_len = SeqLens[b];
                    int num_blocks = (seq_len + BlockSize - 1) / BlockSize;
                    int blocks_per_chunk = (num_blocks + NUM_CHUNKS - 1) / NUM_CHUNKS;
                    int block_start = chunk * blocks_per_chunk;
                    int block_end = std::min(block_start + blocks_per_chunk, num_blocks);

                    float m_local[16]; float l_local[16]; alignas(64) float acc_local[16][512]; 
                    for(int h=0; h<16; h++) { m_local[h] = -INFINITY; l_local[h] = 0.0f; for(int d=0; d<512; d+=16) _mm512_store_ps(&acc_local[h][d], _mm512_setzero_ps()); }
                    const bf16* kv_batch_start = KV_flat + ((size_t)b * kv_stride * D);

                    for (int i = block_start; i < block_end; i++) {
                        const bf16* kv_block_ptr = kv_batch_start + ((size_t)i * BlockSize * D);
                        int valid_tokens = (i < num_blocks - 1) ? 64 : (seq_len - i*64);

                        // 1. K-PACKING (No Timer)
                        for(int t=0; t<64; t+=16) {
                            for(int d=0; d<576; d+=32) { pack_k_chunk_amx(kv_block_ptr + t*D + d, &k_pack[t*576 + d*16], D); }
                        }

                        // 2. Q * K^T (No Timer)
                        _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3); 
                        for (int k=0; k<576; k+=32) {
                            _tile_loadd(4, q_ptr_base + k, 1152);
                            _tile_loadd(5, &k_pack[0*576 + k*16], 64);   _tile_dpbf16ps(0, 4, 5);
                            _tile_loadd(6, &k_pack[16*576 + k*16], 64);  _tile_dpbf16ps(1, 4, 6);
                            _tile_loadd(7, &k_pack[32*576 + k*16], 64);  _tile_dpbf16ps(2, 4, 7);
                            _tile_loadd(5, &k_pack[48*576 + k*16], 64);  _tile_dpbf16ps(3, 4, 5);
                        }
                        alignas(64) float scores[16][64];
                        _tile_stored(0, &scores[0][0],  256); _tile_stored(1, &scores[0][16], 256);
                        _tile_stored(2, &scores[0][32], 256); _tile_stored(3, &scores[0][48], 256);

                        // ⏱️ TIMER: SOFTMAX ONLY
                        double t_soft_start = omp_get_wtime();
                        
                        alignas(64) bf16 scores_bf16[16][64];
                        for(int h=0; h<16; h++) {
                            float m_block = -INFINITY;
                            for(int tok=0; tok<valid_tokens; tok++) {
                                scores[h][tok] *= scale;
                                if (scores[h][tok] > m_block) m_block = scores[h][tok];
                            }
                            
                            float m_new = std::max(m_local[h], m_block);
                            if(m_new == -INFINITY) m_new = 0.0f;
                            float alpha = std::exp(m_local[h] - m_new);
                            float beta = std::exp(m_block - m_new);
                            
                            __m512 alpha_v = _mm512_set1_ps(alpha);
                            for(int d=0; d<512; d+=16) 
                                _mm512_store_ps(&acc_local[h][d], _mm512_mul_ps(_mm512_load_ps(&acc_local[h][d]), alpha_v));

                            float l_block = 0.0f;
                            for(int tok=0; tok<64; tok+=16) {
                                __m512 prob_v = _mm512_setzero_ps();
                                if (tok < valid_tokens) {
                                    __m512 s_vec = _mm512_load_ps(&scores[h][tok]);
                                    __m512 m_vec = _mm512_set1_ps(m_block);
                                    __m512 b_vec = _mm512_set1_ps(beta);
                                    prob_v = _mm512_mul_ps(fast_exp_512(_mm512_sub_ps(s_vec, m_vec)), b_vec);
                                    l_block += _mm512_reduce_add_ps(prob_v);
                                }
                                __m512i int_val = _mm512_castps_si512(prob_v);
                                __m512i shifted = _mm512_srli_epi32(int_val, 16);
                                __m256i bf16_vals = _mm512_cvtepi32_epi16(shifted);
                                _mm256_storeu_si256((__m256i*)&scores_bf16[h][tok], bf16_vals);
                            }
                            m_local[h] = m_new;
                            l_local[h] = (l_local[h] * alpha) + l_block;
                        }

                        t_soft[tid] += (omp_get_wtime() - t_soft_start);

                        // 4. V-PACKING (No Timer)
                        for (int d_blk = 0; d_blk < 512; d_blk += 16) {
                            for (int t_blk = 0; t_blk < 64; t_blk += 32) {
                                bf16* dst = &v_pack[(d_blk / 16) * 1024 + (t_blk / 32) * 512];
                                for (int r = 0; r < 16; r++) {
                                    int tok = t_blk + r * 2;
                                    for (int c = 0; c < 16; c++) {
                                        bf16 val0 = (tok < valid_tokens) ? kv_block_ptr[tok * D + d_blk + c] : 0;
                                        bf16 val1 = (tok + 1 < valid_tokens) ? kv_block_ptr[(tok + 1) * D + d_blk + c] : 0;
                                        dst[r * 32 + c * 2 + 0] = val0; dst[r * 32 + c * 2 + 1] = val1;
                                    }
                                }
                            }
                        }

                        // 5. P * V (AMX) (No Timer)
                        _tile_loadd(4, &scores_bf16[0][0], 128); _tile_loadd(5, &scores_bf16[0][32], 128);
                        for(int d=0; d<512; d+=16) {
                            _tile_zero(0);
                            bf16* v_ptr_t0 = &v_pack[(d / 16) * 1024]; bf16* v_ptr_t32 = &v_pack[(d / 16) * 1024 + 512];
                            _tile_loadd(6, v_ptr_t0, 64);   _tile_dpbf16ps(0, 4, 6);
                            _tile_loadd(7, v_ptr_t32, 64);  _tile_dpbf16ps(0, 5, 7);
                            alignas(64) float delta_o[16][16]; _tile_stored(0, delta_o, 64);
                            for(int h=0; h<16; h++) {
                                __m512 acc = _mm512_load_ps(&acc_local[h][d]); __m512 d0  = _mm512_load_ps(&delta_o[h][0]);
                                _mm512_store_ps(&acc_local[h][d], _mm512_add_ps(acc, d0));
                            }
                        }
                    }

                    int state_idx_base = (((b * H_GROUPS) + h_g) * NUM_CHUNKS + chunk) * HEAD_GROUP;
                    for (int h = 0; h < 16; h++) {
                        global_m[state_idx_base + h] = m_local[h]; global_l[state_idx_base + h] = l_local[h];
                        for (int d = 0; d < 512; d++) global_acc[(state_idx_base + h) * Dv + d] = acc_local[h][d];
                    }
                } 
            } 
        } 
        t_map_total[tid] = omp_get_wtime() - map_start;

        double b_start = omp_get_wtime();
        #pragma omp barrier
        t_barrier[tid] = omp_get_wtime() - b_start;

        double r_start = omp_get_wtime();
        #pragma omp for collapse(2) schedule(dynamic, 1)
        for (int b = 0; b < B; b++) {
            for (int h_g = 0; h_g < H_GROUPS; h_g++) {
                int h_start = h_g * HEAD_GROUP;
                for (int h = 0; h < 16; h++) {
                    float m_final = -INFINITY; float l_final = 0.0f; alignas(64) float acc_final[512] = {0};
                    for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {
                        int state_idx = (((b * H_GROUPS) + h_g) * NUM_CHUNKS + chunk) * HEAD_GROUP + h;
                        float m_chunk = global_m[state_idx]; float l_chunk = global_l[state_idx]; float* acc_chunk = &global_acc[state_idx * Dv];
                        if (m_chunk == -INFINITY) continue;
                        float m_new = std::max(m_final, m_chunk);
                        float alpha_final = std::exp(m_final - m_new); float alpha_chunk = std::exp(m_chunk - m_new);
                        __m512 a_f_v = _mm512_set1_ps(alpha_final); __m512 a_c_v = _mm512_set1_ps(alpha_chunk);
                        for (int d = 0; d < 512; d+=16) {
                            __m512 acc_f = _mm512_load_ps(&acc_final[d]); __m512 acc_c = _mm512_loadu_ps(&acc_chunk[d]);
                            acc_f = _mm512_mul_ps(acc_f, a_f_v); acc_f = _mm512_fmadd_ps(acc_c, a_c_v, acc_f); _mm512_store_ps(&acc_final[d], acc_f);
                        }
                        l_final = (l_final * alpha_final) + (l_chunk * alpha_chunk); m_final = m_new;
                    }
                    bf16* out_ptr = Out + (b * H * Dv) + ((h_start + h) * Dv);
                    float denom = (l_final == 0.0f) ? 1e-10f : l_final; __m512 denom_v = _mm512_set1_ps(denom);
                    for(int d=0; d<512; d+=16) {
                        __m512 acc = _mm512_load_ps(&acc_final[d]); store_fp32_as_bf16(out_ptr + d, _mm512_div_ps(acc, denom_v));
                    }
                }
            }
        }
        t_reduce[tid] = omp_get_wtime() - r_start;
    } 

    double max_soft = 0, max_map = 0;
    for (int i = 0; i < max_threads; i++) {
        if (t_map_total[i] > 0) { 
            max_soft = std::max(max_soft, t_soft[i]);
            max_map = std::max(max_map, t_map_total[i]);
        }
    }
    printf("\n========= SOFTMAX ONLY ========\n");
    printf("3. Softmax+AVX+Cast    | %8.2f ms\n", max_soft*1000.0);
    printf("-------------------------------\n");
    printf("Total Map Phase        | %8.2f ms\n", max_map*1000.0);
    printf("===============================\n\n");
}

void dense_attention_amx_profile_vpacking_only(
    const bf16* Q, 
    const bf16* KV_flat, 
    const int32_t* SeqLens,
    bf16* Out,
    int B, int H, int D, int BlockSize, float scale,
    size_t kv_stride 
) {
    int Dv = 512; int HEAD_GROUP = 16; int H_GROUPS = H / HEAD_GROUP; int NUM_CHUNKS = 12; 
    int num_states = B * H_GROUPS * NUM_CHUNKS * HEAD_GROUP;
    std::vector<float> global_m(num_states, -INFINITY); std::vector<float> global_l(num_states, 0.0f); std::vector<float> global_acc(num_states * Dv, 0.0f);
    int max_threads = omp_get_max_threads();
    
    // --- TIMERS ---
    std::vector<double> t_soft(max_threads, 0.0);
    std::vector<double> t_map_total(max_threads, 0.0);
    std::vector<double> t_barrier(max_threads, 0.0);
    std::vector<double> t_reduce(max_threads, 0.0);

    #pragma omp parallel 
    {
        int tid = omp_get_thread_num();
        init_amx(); 
        static tile_config_t cfg; memset(&cfg, 0, sizeof(cfg)); cfg.palette_id = 1; cfg.start_row = 0;
        for(int i=0; i<8; i++) { cfg.rows[i] = 16; cfg.colsb[i] = 64; }
        _tile_loadconfig(&cfg);

        alignas(64) bf16 k_pack[64 * 576]; alignas(64) bf16 v_pack[64 * 512]; 
        double map_start = omp_get_wtime();

        #pragma omp for collapse(3) schedule(dynamic, 1) nowait
        for (int b = 0; b < B; b++) {
            for (int h_g = 0; h_g < H_GROUPS; h_g++) {
                for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {
                    
                    int h_start = h_g * HEAD_GROUP;
                    const bf16* q_ptr_base = Q + (b * H * D) + (h_start * D);
                    int seq_len = SeqLens[b];
                    int num_blocks = (seq_len + BlockSize - 1) / BlockSize;
                    int blocks_per_chunk = (num_blocks + NUM_CHUNKS - 1) / NUM_CHUNKS;
                    int block_start = chunk * blocks_per_chunk;
                    int block_end = std::min(block_start + blocks_per_chunk, num_blocks);

                    float m_local[16]; float l_local[16]; alignas(64) float acc_local[16][512]; 
                    for(int h=0; h<16; h++) { m_local[h] = -INFINITY; l_local[h] = 0.0f; for(int d=0; d<512; d+=16) _mm512_store_ps(&acc_local[h][d], _mm512_setzero_ps()); }
                    const bf16* kv_batch_start = KV_flat + ((size_t)b * kv_stride * D);

                    for (int i = block_start; i < block_end; i++) {
                        const bf16* kv_block_ptr = kv_batch_start + ((size_t)i * BlockSize * D);
                        int valid_tokens = (i < num_blocks - 1) ? 64 : (seq_len - i*64);

                        // 1. K-PACKING (No Timer)
                        for(int t=0; t<64; t+=16) {
                            for(int d=0; d<576; d+=32) { pack_k_chunk_amx(kv_block_ptr + t*D + d, &k_pack[t*576 + d*16], D); }
                        }

                        // 2. Q * K^T (No Timer)
                        _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3); 
                        for (int k=0; k<576; k+=32) {
                            _tile_loadd(4, q_ptr_base + k, 1152);
                            _tile_loadd(5, &k_pack[0*576 + k*16], 64);   _tile_dpbf16ps(0, 4, 5);
                            _tile_loadd(6, &k_pack[16*576 + k*16], 64);  _tile_dpbf16ps(1, 4, 6);
                            _tile_loadd(7, &k_pack[32*576 + k*16], 64);  _tile_dpbf16ps(2, 4, 7);
                            _tile_loadd(5, &k_pack[48*576 + k*16], 64);  _tile_dpbf16ps(3, 4, 5);
                        }
                        alignas(64) float scores[16][64];
                        _tile_stored(0, &scores[0][0],  256); _tile_stored(1, &scores[0][16], 256);
                        _tile_stored(2, &scores[0][32], 256); _tile_stored(3, &scores[0][48], 256);

                        
                        alignas(64) bf16 scores_bf16[16][64];
                        for(int h=0; h<16; h++) {
                            float m_block = -INFINITY;
                            for(int tok=0; tok<valid_tokens; tok++) {
                                scores[h][tok] *= scale;
                                if (scores[h][tok] > m_block) m_block = scores[h][tok];
                            }
                            
                            float m_new = std::max(m_local[h], m_block);
                            if(m_new == -INFINITY) m_new = 0.0f;
                            float alpha = std::exp(m_local[h] - m_new);
                            float beta = std::exp(m_block - m_new);
                            
                            __m512 alpha_v = _mm512_set1_ps(alpha);
                            for(int d=0; d<512; d+=16) 
                                _mm512_store_ps(&acc_local[h][d], _mm512_mul_ps(_mm512_load_ps(&acc_local[h][d]), alpha_v));

                            float l_block = 0.0f;
                            for(int tok=0; tok<64; tok+=16) {
                                __m512 prob_v = _mm512_setzero_ps();
                                if (tok < valid_tokens) {
                                    __m512 s_vec = _mm512_load_ps(&scores[h][tok]);
                                    __m512 m_vec = _mm512_set1_ps(m_block);
                                    __m512 b_vec = _mm512_set1_ps(beta);
                                    prob_v = _mm512_mul_ps(fast_exp_512(_mm512_sub_ps(s_vec, m_vec)), b_vec);
                                    l_block += _mm512_reduce_add_ps(prob_v);
                                }
                                __m512i int_val = _mm512_castps_si512(prob_v);
                                __m512i shifted = _mm512_srli_epi32(int_val, 16);
                                __m256i bf16_vals = _mm512_cvtepi32_epi16(shifted);
                                _mm256_storeu_si256((__m256i*)&scores_bf16[h][tok], bf16_vals);
                            }
                            m_local[h] = m_new;
                            l_local[h] = (l_local[h] * alpha) + l_block;
                        }

                        // ⏱️ TIMER: SOFTMAX ONLY
                        double t_soft_start = omp_get_wtime();

                        // 4. V-PACKING (No Timer)
                        for (int d_blk = 0; d_blk < 512; d_blk += 16) {
                            for (int t_blk = 0; t_blk < 64; t_blk += 32) {
                                bf16* dst = &v_pack[(d_blk / 16) * 1024 + (t_blk / 32) * 512];
                                for (int r = 0; r < 16; r++) {
                                    int tok = t_blk + r * 2;
                                    for (int c = 0; c < 16; c++) {
                                        bf16 val0 = (tok < valid_tokens) ? kv_block_ptr[tok * D + d_blk + c] : 0;
                                        bf16 val1 = (tok + 1 < valid_tokens) ? kv_block_ptr[(tok + 1) * D + d_blk + c] : 0;
                                        dst[r * 32 + c * 2 + 0] = val0; dst[r * 32 + c * 2 + 1] = val1;
                                    }
                                }
                            }
                        }
                        t_soft[tid] += (omp_get_wtime() - t_soft_start);

                        // 5. P * V (AMX) (No Timer)
                        _tile_loadd(4, &scores_bf16[0][0], 128); _tile_loadd(5, &scores_bf16[0][32], 128);
                        for(int d=0; d<512; d+=16) {
                            _tile_zero(0);
                            bf16* v_ptr_t0 = &v_pack[(d / 16) * 1024]; bf16* v_ptr_t32 = &v_pack[(d / 16) * 1024 + 512];
                            _tile_loadd(6, v_ptr_t0, 64);   _tile_dpbf16ps(0, 4, 6);
                            _tile_loadd(7, v_ptr_t32, 64);  _tile_dpbf16ps(0, 5, 7);
                            alignas(64) float delta_o[16][16]; _tile_stored(0, delta_o, 64);
                            for(int h=0; h<16; h++) {
                                __m512 acc = _mm512_load_ps(&acc_local[h][d]); __m512 d0  = _mm512_load_ps(&delta_o[h][0]);
                                _mm512_store_ps(&acc_local[h][d], _mm512_add_ps(acc, d0));
                            }
                        }
                    }

                    int state_idx_base = (((b * H_GROUPS) + h_g) * NUM_CHUNKS + chunk) * HEAD_GROUP;
                    for (int h = 0; h < 16; h++) {
                        global_m[state_idx_base + h] = m_local[h]; global_l[state_idx_base + h] = l_local[h];
                        for (int d = 0; d < 512; d++) global_acc[(state_idx_base + h) * Dv + d] = acc_local[h][d];
                    }
                } 
            } 
        } 
        t_map_total[tid] = omp_get_wtime() - map_start;

        double b_start = omp_get_wtime();
        #pragma omp barrier
        t_barrier[tid] = omp_get_wtime() - b_start;

        double r_start = omp_get_wtime();
        #pragma omp for collapse(2) schedule(dynamic, 1)
        for (int b = 0; b < B; b++) {
            for (int h_g = 0; h_g < H_GROUPS; h_g++) {
                int h_start = h_g * HEAD_GROUP;
                for (int h = 0; h < 16; h++) {
                    float m_final = -INFINITY; float l_final = 0.0f; alignas(64) float acc_final[512] = {0};
                    for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {
                        int state_idx = (((b * H_GROUPS) + h_g) * NUM_CHUNKS + chunk) * HEAD_GROUP + h;
                        float m_chunk = global_m[state_idx]; float l_chunk = global_l[state_idx]; float* acc_chunk = &global_acc[state_idx * Dv];
                        if (m_chunk == -INFINITY) continue;
                        float m_new = std::max(m_final, m_chunk);
                        float alpha_final = std::exp(m_final - m_new); float alpha_chunk = std::exp(m_chunk - m_new);
                        __m512 a_f_v = _mm512_set1_ps(alpha_final); __m512 a_c_v = _mm512_set1_ps(alpha_chunk);
                        for (int d = 0; d < 512; d+=16) {
                            __m512 acc_f = _mm512_load_ps(&acc_final[d]); __m512 acc_c = _mm512_loadu_ps(&acc_chunk[d]);
                            acc_f = _mm512_mul_ps(acc_f, a_f_v); acc_f = _mm512_fmadd_ps(acc_c, a_c_v, acc_f); _mm512_store_ps(&acc_final[d], acc_f);
                        }
                        l_final = (l_final * alpha_final) + (l_chunk * alpha_chunk); m_final = m_new;
                    }
                    bf16* out_ptr = Out + (b * H * Dv) + ((h_start + h) * Dv);
                    float denom = (l_final == 0.0f) ? 1e-10f : l_final; __m512 denom_v = _mm512_set1_ps(denom);
                    for(int d=0; d<512; d+=16) {
                        __m512 acc = _mm512_load_ps(&acc_final[d]); store_fp32_as_bf16(out_ptr + d, _mm512_div_ps(acc, denom_v));
                    }
                }
            }
        }
        t_reduce[tid] = omp_get_wtime() - r_start;
    } 

    double max_soft = 0, max_map = 0;
    for (int i = 0; i < max_threads; i++) {
        if (t_map_total[i] > 0) { 
            max_soft = std::max(max_soft, t_soft[i]);
            max_map = std::max(max_map, t_map_total[i]);
        }
    }
    printf("\n========= SOFTMAX ONLY ========\n");
    printf("3. V Packing    | %8.2f ms\n", max_soft*1000.0);
    printf("-------------------------------\n");
    printf("Total Map Phase        | %8.2f ms\n", max_map*1000.0);
    printf("===============================\n\n");
}

void dense_attention_amx_profile_pv_mul_only(
    const bf16* Q, 
    const bf16* KV_flat, 
    const int32_t* SeqLens,
    bf16* Out,
    int B, int H, int D, int BlockSize, float scale,
    size_t kv_stride 
) {
    int Dv = 512; int HEAD_GROUP = 16; int H_GROUPS = H / HEAD_GROUP; int NUM_CHUNKS = 12; 
    int num_states = B * H_GROUPS * NUM_CHUNKS * HEAD_GROUP;
    std::vector<float> global_m(num_states, -INFINITY); std::vector<float> global_l(num_states, 0.0f); std::vector<float> global_acc(num_states * Dv, 0.0f);
    int max_threads = omp_get_max_threads();
    
    // --- TIMERS ---
    std::vector<double> t_soft(max_threads, 0.0);
    std::vector<double> t_map_total(max_threads, 0.0);
    std::vector<double> t_barrier(max_threads, 0.0);
    std::vector<double> t_reduce(max_threads, 0.0);

    #pragma omp parallel 
    {
        int tid = omp_get_thread_num();
        init_amx(); 
        static tile_config_t cfg; memset(&cfg, 0, sizeof(cfg)); cfg.palette_id = 1; cfg.start_row = 0;
        for(int i=0; i<8; i++) { cfg.rows[i] = 16; cfg.colsb[i] = 64; }
        _tile_loadconfig(&cfg);

        alignas(64) bf16 k_pack[64 * 576]; alignas(64) bf16 v_pack[64 * 512]; 
        double map_start = omp_get_wtime();

        #pragma omp for collapse(3) schedule(dynamic, 1) nowait
        for (int b = 0; b < B; b++) {
            for (int h_g = 0; h_g < H_GROUPS; h_g++) {
                for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {
                    
                    int h_start = h_g * HEAD_GROUP;
                    const bf16* q_ptr_base = Q + (b * H * D) + (h_start * D);
                    int seq_len = SeqLens[b];
                    int num_blocks = (seq_len + BlockSize - 1) / BlockSize;
                    int blocks_per_chunk = (num_blocks + NUM_CHUNKS - 1) / NUM_CHUNKS;
                    int block_start = chunk * blocks_per_chunk;
                    int block_end = std::min(block_start + blocks_per_chunk, num_blocks);

                    float m_local[16]; float l_local[16]; alignas(64) float acc_local[16][512]; 
                    for(int h=0; h<16; h++) { m_local[h] = -INFINITY; l_local[h] = 0.0f; for(int d=0; d<512; d+=16) _mm512_store_ps(&acc_local[h][d], _mm512_setzero_ps()); }
                    const bf16* kv_batch_start = KV_flat + ((size_t)b * kv_stride * D);

                    for (int i = block_start; i < block_end; i++) {
                        const bf16* kv_block_ptr = kv_batch_start + ((size_t)i * BlockSize * D);
                        int valid_tokens = (i < num_blocks - 1) ? 64 : (seq_len - i*64);

                        // 1. K-PACKING (No Timer)
                        for(int t=0; t<64; t+=16) {
                            for(int d=0; d<576; d+=32) { pack_k_chunk_amx(kv_block_ptr + t*D + d, &k_pack[t*576 + d*16], D); }
                        }

                        // 2. Q * K^T (No Timer)
                        _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3); 
                        for (int k=0; k<576; k+=32) {
                            _tile_loadd(4, q_ptr_base + k, 1152);
                            _tile_loadd(5, &k_pack[0*576 + k*16], 64);   _tile_dpbf16ps(0, 4, 5);
                            _tile_loadd(6, &k_pack[16*576 + k*16], 64);  _tile_dpbf16ps(1, 4, 6);
                            _tile_loadd(7, &k_pack[32*576 + k*16], 64);  _tile_dpbf16ps(2, 4, 7);
                            _tile_loadd(5, &k_pack[48*576 + k*16], 64);  _tile_dpbf16ps(3, 4, 5);
                        }
                        alignas(64) float scores[16][64];
                        _tile_stored(0, &scores[0][0],  256); _tile_stored(1, &scores[0][16], 256);
                        _tile_stored(2, &scores[0][32], 256); _tile_stored(3, &scores[0][48], 256);

                        
                        alignas(64) bf16 scores_bf16[16][64];
                        for(int h=0; h<16; h++) {
                            float m_block = -INFINITY;
                            for(int tok=0; tok<valid_tokens; tok++) {
                                scores[h][tok] *= scale;
                                if (scores[h][tok] > m_block) m_block = scores[h][tok];
                            }
                            
                            float m_new = std::max(m_local[h], m_block);
                            if(m_new == -INFINITY) m_new = 0.0f;
                            float alpha = std::exp(m_local[h] - m_new);
                            float beta = std::exp(m_block - m_new);
                            
                            __m512 alpha_v = _mm512_set1_ps(alpha);
                            for(int d=0; d<512; d+=16) 
                                _mm512_store_ps(&acc_local[h][d], _mm512_mul_ps(_mm512_load_ps(&acc_local[h][d]), alpha_v));

                            float l_block = 0.0f;
                            for(int tok=0; tok<64; tok+=16) {
                                __m512 prob_v = _mm512_setzero_ps();
                                if (tok < valid_tokens) {
                                    __m512 s_vec = _mm512_load_ps(&scores[h][tok]);
                                    __m512 m_vec = _mm512_set1_ps(m_block);
                                    __m512 b_vec = _mm512_set1_ps(beta);
                                    prob_v = _mm512_mul_ps(fast_exp_512(_mm512_sub_ps(s_vec, m_vec)), b_vec);
                                    l_block += _mm512_reduce_add_ps(prob_v);
                                }
                                __m512i int_val = _mm512_castps_si512(prob_v);
                                __m512i shifted = _mm512_srli_epi32(int_val, 16);
                                __m256i bf16_vals = _mm512_cvtepi32_epi16(shifted);
                                _mm256_storeu_si256((__m256i*)&scores_bf16[h][tok], bf16_vals);
                            }
                            m_local[h] = m_new;
                            l_local[h] = (l_local[h] * alpha) + l_block;
                        }


                        // 4. V-PACKING (No Timer)
                        for (int d_blk = 0; d_blk < 512; d_blk += 16) {
                            for (int t_blk = 0; t_blk < 64; t_blk += 32) {
                                bf16* dst = &v_pack[(d_blk / 16) * 1024 + (t_blk / 32) * 512];
                                for (int r = 0; r < 16; r++) {
                                    int tok = t_blk + r * 2;
                                    for (int c = 0; c < 16; c++) {
                                        bf16 val0 = (tok < valid_tokens) ? kv_block_ptr[tok * D + d_blk + c] : 0;
                                        bf16 val1 = (tok + 1 < valid_tokens) ? kv_block_ptr[(tok + 1) * D + d_blk + c] : 0;
                                        dst[r * 32 + c * 2 + 0] = val0; dst[r * 32 + c * 2 + 1] = val1;
                                    }
                                }
                            }
                        }
                        // ⏱️ TIMER: SOFTMAX ONLY
                        double t_soft_start = omp_get_wtime();

                        // 5. P * V (AMX) (No Timer)
                        _tile_loadd(4, &scores_bf16[0][0], 128); _tile_loadd(5, &scores_bf16[0][32], 128);
                        for(int d=0; d<512; d+=16) {
                            _tile_zero(0);
                            bf16* v_ptr_t0 = &v_pack[(d / 16) * 1024]; bf16* v_ptr_t32 = &v_pack[(d / 16) * 1024 + 512];
                            _tile_loadd(6, v_ptr_t0, 64);   _tile_dpbf16ps(0, 4, 6);
                            _tile_loadd(7, v_ptr_t32, 64);  _tile_dpbf16ps(0, 5, 7);
                            alignas(64) float delta_o[16][16]; _tile_stored(0, delta_o, 64);
                            for(int h=0; h<16; h++) {
                                __m512 acc = _mm512_load_ps(&acc_local[h][d]); __m512 d0  = _mm512_load_ps(&delta_o[h][0]);
                                _mm512_store_ps(&acc_local[h][d], _mm512_add_ps(acc, d0));
                            }
                        }
                        t_soft[tid] += (omp_get_wtime() - t_soft_start);
                    }

                    int state_idx_base = (((b * H_GROUPS) + h_g) * NUM_CHUNKS + chunk) * HEAD_GROUP;
                    for (int h = 0; h < 16; h++) {
                        global_m[state_idx_base + h] = m_local[h]; global_l[state_idx_base + h] = l_local[h];
                        for (int d = 0; d < 512; d++) global_acc[(state_idx_base + h) * Dv + d] = acc_local[h][d];
                    }
                } 
            } 
        } 
        t_map_total[tid] = omp_get_wtime() - map_start;

        double b_start = omp_get_wtime();
        #pragma omp barrier
        t_barrier[tid] = omp_get_wtime() - b_start;

        double r_start = omp_get_wtime();
        #pragma omp for collapse(2) schedule(dynamic, 1)
        for (int b = 0; b < B; b++) {
            for (int h_g = 0; h_g < H_GROUPS; h_g++) {
                int h_start = h_g * HEAD_GROUP;
                for (int h = 0; h < 16; h++) {
                    float m_final = -INFINITY; float l_final = 0.0f; alignas(64) float acc_final[512] = {0};
                    for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {
                        int state_idx = (((b * H_GROUPS) + h_g) * NUM_CHUNKS + chunk) * HEAD_GROUP + h;
                        float m_chunk = global_m[state_idx]; float l_chunk = global_l[state_idx]; float* acc_chunk = &global_acc[state_idx * Dv];
                        if (m_chunk == -INFINITY) continue;
                        float m_new = std::max(m_final, m_chunk);
                        float alpha_final = std::exp(m_final - m_new); float alpha_chunk = std::exp(m_chunk - m_new);
                        __m512 a_f_v = _mm512_set1_ps(alpha_final); __m512 a_c_v = _mm512_set1_ps(alpha_chunk);
                        for (int d = 0; d < 512; d+=16) {
                            __m512 acc_f = _mm512_load_ps(&acc_final[d]); __m512 acc_c = _mm512_loadu_ps(&acc_chunk[d]);
                            acc_f = _mm512_mul_ps(acc_f, a_f_v); acc_f = _mm512_fmadd_ps(acc_c, a_c_v, acc_f); _mm512_store_ps(&acc_final[d], acc_f);
                        }
                        l_final = (l_final * alpha_final) + (l_chunk * alpha_chunk); m_final = m_new;
                    }
                    bf16* out_ptr = Out + (b * H * Dv) + ((h_start + h) * Dv);
                    float denom = (l_final == 0.0f) ? 1e-10f : l_final; __m512 denom_v = _mm512_set1_ps(denom);
                    for(int d=0; d<512; d+=16) {
                        __m512 acc = _mm512_load_ps(&acc_final[d]); store_fp32_as_bf16(out_ptr + d, _mm512_div_ps(acc, denom_v));
                    }
                }
            }
        }
        t_reduce[tid] = omp_get_wtime() - r_start;
    } 

    double max_soft = 0, max_map = 0;
    for (int i = 0; i < max_threads; i++) {
        if (t_map_total[i] > 0) { 
            max_soft = std::max(max_soft, t_soft[i]);
            max_map = std::max(max_map, t_map_total[i]);
        }
    }
    printf("\n========= SOFTMAX ONLY ========\n");
    printf("3. PV Multiplication    | %8.2f ms\n", max_soft*1000.0);
    printf("-------------------------------\n");
    printf("Total Map Phase        | %8.2f ms\n", max_map*1000.0);
    printf("===============================\n\n");
}

int main() {
    std::cout << "⚡ LOADING DATA..." << std::endl;
    auto cfg_data = read_file("bins/config.bin");
    std::cout << "Config Data: " << cfg_data.data() << " bytes" << std::endl;
    Config* cfg = reinterpret_cast<Config*>(cfg_data.data());
    auto q_bin = read_file("bins/q.bin");
    auto k_bin = read_file("bins/k_cache.bin");
    auto bt_bin = read_file("bins/block_table.bin");
    auto sl_bin = read_file("bins/seqlens.bin");
    std::cout << "sl_bin Data: " << sl_bin.data() << " bytes" << std::endl;

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
    double ms_flat = std::chrono::duration<double, std::milli>(end_flat - start_flat).count();
    std::cout << "  Flattening time: " << ms_flat << " ms" << std::endl;

    std::cout << "🚀 RUNNING OPTIMIZED KERNEL (VECTORIZED EXP)..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    dense_attention_amx_profile_phases_only(Q, KV_flat.data(), SL, out_cpu.data(), cfg->B, cfg->H, cfg->D, cfg->BlockSize, cfg->scale, kv_stride);
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "✅ Compute Time: " << ms << " ms" << std::endl;

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

    std::cout << "🚀 RUNNING OPTIMIZED KERNEL (VECTORIZED EXP)..." << std::endl;
     start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 10; i++)
        dense_attention_amx_profile_phases_only(Q, KV_flat.data(), SL, out_cpu.data(), cfg->B, cfg->H, cfg->D, cfg->BlockSize, cfg->scale, kv_stride);
     end = std::chrono::high_resolution_clock::now();
     ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "✅ Compute Time: " << ms/10 << " ms" << std::endl;

    return 0;
}