#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <fstream>
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

// ============================================================================
// Step 5g — Adaptive (cores, batch_size) → (effective_threads, NUM_CHUNKS)
// ============================================================================
// Empirical 9-row sweep on SPR 8468 produced these optima:
//   (B=1, 32c) → chunks=4, 0.463 ms
//   (B=3, 32c) → chunks=4, 0.435 ms/batch
//   (B=3, 96c) → chunks=4, 0.294 ms/batch  (cross-NUMA pays off)
//   (B=1, 96c) → no good — falls below ~8 blocks/thread, cross-NUMA loses
// Decision rule: dual-socket only when total_blocks ≥ 8×96=768; else single-socket.
struct AttentionConfig {
    int effective_threads;
    int num_chunks;
    const char* rationale;
};

inline int amx_gcd(int a, int b) {
    while (b) { int t = b; b = a % b; a = t; }
    return a;
}

inline AttentionConfig pick_attention_config(
    int max_threads, int B, int H_GROUPS, int num_blocks)
{
    constexpr int SINGLE_SOCKET_THREADS = 32;
    constexpr int DUAL_SOCKET_THREADS   = 96;
    constexpr int MIN_BLOCKS_PER_THREAD = 8;
    constexpr int MIN_BLOCKS_PER_CHUNK  = 4;

    long long total_blocks = (long long)B * H_GROUPS * num_blocks;

    int effective;
    const char* rationale;
    if (max_threads >= DUAL_SOCKET_THREADS &&
        total_blocks >= (long long)MIN_BLOCKS_PER_THREAD * DUAL_SOCKET_THREADS) {
        effective = DUAL_SOCKET_THREADS;
        rationale = "dual-socket: enough work to amortize cross-NUMA";
    } else if (total_blocks >= (long long)MIN_BLOCKS_PER_THREAD * SINGLE_SOCKET_THREADS) {
        effective = std::min(max_threads, SINGLE_SOCKET_THREADS);
        rationale = "single-socket: insufficient work for cross-NUMA";
    } else {
        effective = std::max(1, (int)(total_blocks / MIN_BLOCKS_PER_THREAD));
        rationale = "tiny: downscaled to match work";
    }

    int g = amx_gcd(effective, B * H_GROUPS);
    int ideal_chunks = effective / g;

    int max_chunks_by_blocks = std::max(1, num_blocks / MIN_BLOCKS_PER_CHUNK);
    int num_chunks = std::max(1, std::min(ideal_chunks, max_chunks_by_blocks));

    return {effective, num_chunks, rationale};
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

    // Adaptive (cores, batch_size) → (effective_threads, NUM_CHUNKS).
    int max_threads = omp_get_max_threads();
    int num_blocks_for_dispatch = (SeqLens[0] + BlockSize - 1) / BlockSize;
    AttentionConfig cfg = pick_attention_config(max_threads, B, H_GROUPS, num_blocks_for_dispatch);
    // Env overrides for diagnostic sweeps (e.g. 48-core scaling check) —
    // dispatcher-chosen values still drive the default behavior.
    if (const char* e = getenv("FORCE_EFFECTIVE_THREADS")) {
        cfg.effective_threads = std::max(1, atoi(e));
        cfg.rationale = "env-forced effective_threads";
    }
    int NUM_CHUNKS = cfg.num_chunks;
    if (const char* e = getenv("FORCE_NUM_CHUNKS")) {
        NUM_CHUNKS = std::max(1, atoi(e));
    }

    static thread_local int print_once = 0;
    if (!print_once) {
        printf("[dispatcher] B=%d H_GROUPS=%d num_blocks=%d max_threads=%d -> "
               "effective_threads=%d num_chunks=%d (%s)\n",
               B, H_GROUPS, num_blocks_for_dispatch, max_threads,
               cfg.effective_threads, NUM_CHUNKS, cfg.rationale);
        print_once = 1;
    }

    int num_states = B * H_GROUPS * NUM_CHUNKS * HEAD_GROUP;
    std::vector<float> global_m(num_states, -INFINITY);
    std::vector<float> global_l(num_states, 0.0f);
    std::vector<float> global_acc(num_states * Dv, 0.0f);

    std::vector<double> t_kpack(max_threads, 0.0);
    std::vector<double> t_qkt(max_threads, 0.0);
    std::vector<double> t_soft_cast(max_threads, 0.0);
    std::vector<double> t_vpack(max_threads, 0.0);
    std::vector<double> t_amx_v(max_threads, 0.0);
    std::vector<double> t_map_total(max_threads, 0.0);
    std::vector<double> t_barrier(max_threads, 0.0);
    std::vector<double> t_reduce(max_threads, 0.0);

    #pragma omp parallel num_threads(cfg.effective_threads)
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

                        // ⏱️ 2. Q * K^T (AMX) — serial sub-chunks for L1 fit
                        // Per sub-chunk working set: K(18KB) + Q(18KB) = 36KB ≤ L1(48KB).
                        // Q stays L1-hot across sub-chunks 1-3. RAW dep on tile 0 is fine
                        // (per amx_compute_only: 1-per-16-cyc throughput regardless of dep).
                        double t1 = omp_get_wtime();
                        alignas(64) float scores[16][64];
                        for (int sub = 0; sub < 64; sub += 16) {
                            _tile_zero(0);
                            for (int k = 0; k < 576; k += 32) {
                                _tile_loadd(4, q_ptr_base + k, 1152);
                                _tile_loadd(5, &k_pack[sub * 576 + k * 16], 64);
                                _tile_dpbf16ps(0, 4, 5);
                            }
                            _tile_stored(0, &scores[0][sub], 256);
                        }
                        t_qkt[tid] += (omp_get_wtime() - t1);

                        // // ⏱️ 3. SOFTMAX + CAST
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

                        // // ⏱️ 4. V-PACKING (VNNI Format for Tokens)
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

                        // ⏱️ 5. P * V (AMX) — 4-way unroll on d for parallel AMX issue + amortized overhead
                        // 8 outer iters × 4 independent output tiles = 32 d-blocks (same as before).
                        // Benefit: 4× less per-iter overhead (tile_zero/store/accumulate),
                        // 4 independent output tiles let AMX scheduler keep the issue port full.
                        double t4 = omp_get_wtime();

                        _tile_loadd(4, &scores_bf16[0][0],  128);   // P[16h, tok 0-31]
                        _tile_loadd(5, &scores_bf16[0][32], 128);   // P[16h, tok 32-63]

                        for (int d = 0; d < 512; d += 64) {
                            _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);

                            bf16* v0_a = &v_pack[((d +  0)/16) * 1024];
                            bf16* v0_b = &v_pack[((d +  0)/16) * 1024 + 512];
                            bf16* v1_a = &v_pack[((d + 16)/16) * 1024];
                            bf16* v1_b = &v_pack[((d + 16)/16) * 1024 + 512];
                            bf16* v2_a = &v_pack[((d + 32)/16) * 1024];
                            bf16* v2_b = &v_pack[((d + 32)/16) * 1024 + 512];
                            bf16* v3_a = &v_pack[((d + 48)/16) * 1024];
                            bf16* v3_b = &v_pack[((d + 48)/16) * 1024 + 512];

                            _tile_loadd(6, v0_a, 64); _tile_dpbf16ps(0, 4, 6);
                            _tile_loadd(7, v0_b, 64); _tile_dpbf16ps(0, 5, 7);
                            _tile_loadd(6, v1_a, 64); _tile_dpbf16ps(1, 4, 6);
                            _tile_loadd(7, v1_b, 64); _tile_dpbf16ps(1, 5, 7);
                            _tile_loadd(6, v2_a, 64); _tile_dpbf16ps(2, 4, 6);
                            _tile_loadd(7, v2_b, 64); _tile_dpbf16ps(2, 5, 7);
                            _tile_loadd(6, v3_a, 64); _tile_dpbf16ps(3, 4, 6);
                            _tile_loadd(7, v3_b, 64); _tile_dpbf16ps(3, 5, 7);

                            alignas(64) float delta_o[4][16][16];
                            _tile_stored(0, &delta_o[0][0][0], 64);
                            _tile_stored(1, &delta_o[1][0][0], 64);
                            _tile_stored(2, &delta_o[2][0][0], 64);
                            _tile_stored(3, &delta_o[3][0][0], 64);

                            for (int b = 0; b < 4; b++) {
                                for (int h = 0; h < 16; h++) {
                                    __m512 acc = _mm512_load_ps(&acc_local[h][d + b * 16]);
                                    __m512 d0  = _mm512_load_ps(&delta_o[b][h][0]);
                                    acc = _mm512_add_ps(acc, d0);
                                    _mm512_store_ps(&acc_local[h][d + b * 16], acc);
                                }
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
        double b_start = omp_get_wtime();
        #pragma omp barrier
        t_barrier[tid] = omp_get_wtime() - b_start;
        double r_start = omp_get_wtime();

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
        t_reduce[tid] = omp_get_wtime() - r_start;
    }

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
        printf("-------------------------------------------------------\n");
        printf("Total Map Phase        |      -       | %8.2f ms\n", max_map*1000.0);
        printf("Barrier Wait Time      |      -       | %8.2f ms\n", max_bar*1000.0);
        printf("Phase 2 (Reduce)       |      -       | %8.2f ms\n", max_red*1000.0);
        printf("=======================================================\n\n");
    }
}

void sparse_attention_amx(
    const bf16* Q, int B, int D, int active_seq_len,
    const DeepSeekToken* K_flat_fp8,
    const int32_t* active_indices,
    bf16* DenseKV_scratch,
    int H, int BlockSize, float scale, bf16* Out
) {
    // Apply the same (cores, B) dispatcher decision to the dequant loop so that
    // first-touch of DenseKV_scratch happens on the same NUMA node(s) the dense
    // kernel will read from. Without this, with OMP_NUM_THREADS=96 the dequant
    // would spread across both sockets while the dense kernel uses only 32
    // threads on NUMA0 → cross-NUMA reads, ~5x slowdown observed in Step 5g.
    int H_GROUPS_disp = H / 16;
    int num_blocks_disp = (active_seq_len + BlockSize - 1) / BlockSize;
    AttentionConfig cfg = pick_attention_config(
        omp_get_max_threads(), B, H_GROUPS_disp, num_blocks_disp);
    if (const char* e = getenv("FORCE_EFFECTIVE_THREADS")) {
        cfg.effective_threads = std::max(1, atoi(e));
    }

    #pragma omp parallel for collapse(2) schedule(static) num_threads(cfg.effective_threads)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < active_seq_len; t++) {
            int physical_token_idx = active_indices[(size_t)b * active_seq_len + t];
            const DeepSeekToken* __restrict src_token = &K_flat_fp8[physical_token_idx];
            bf16* __restrict dst_token = DenseKV_scratch + ((size_t)b * active_seq_len * D) + ((size_t)t * D);

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

            // 3. AVX-512 RoPE Copy
            __m512i rope_0 = _mm512_loadu_si512((const void*)&src_token->rope[0]);
            __m512i rope_1 = _mm512_loadu_si512((const void*)&src_token->rope[32]);
            _mm512_storeu_si512((void*)&dst_token[512], rope_0);
            _mm512_storeu_si512((void*)&dst_token[544], rope_1);
        }
    }

    // Per-batch dummy seqlens (all = active_seq_len because every batch is replicated)
    std::vector<int32_t> dummy_seqlens(B, active_seq_len);
    dense_attention_amx_flat_timers7(
        Q, DenseKV_scratch, dummy_seqlens.data(), Out,
        B, H, D, 64, scale, active_seq_len
    );
}

int main() {
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

    int B_test = []{
        const char* e = getenv("BENCH_B");
        return e ? std::max(1, atoi(e)) : 1;
    }();

    // Replicate Q and indices B_test times (same input each batch)
    std::vector<bf16> Q_rep((size_t)B_test * cfg->H * cfg->D);
    std::vector<int32_t> idx_rep((size_t)B_test * cfg->NumTokens);
    for (int b = 0; b < B_test; b++) {
        memcpy(&Q_rep[(size_t)b * cfg->H * cfg->D],
               Q, (size_t)cfg->H * cfg->D * sizeof(bf16));
        memcpy(&idx_rep[(size_t)b * cfg->NumTokens],
               active_indices, (size_t)cfg->NumTokens * sizeof(int32_t));
    }
    Q = Q_rep.data();
    active_indices = idx_rep.data();
    std::cout << "Effective Batch Size: " << B_test << std::endl;

    int Dv = 512;
    std::vector<bf16> out_cpu((size_t)B_test * cfg->H * Dv);

    // Scratchpad to hold the dense bf16 tokens (B_test batches)
    std::vector<bf16> DenseKV_scratch((size_t)B_test * cfg->NumTokens * cfg->D, 0);

    std::cout << "🚀 RUNNING CUSTOM SPARSE KERNEL (WARMUP)..." << std::endl;
    sparse_attention_amx(Q, B_test, cfg->D, cfg->NumTokens, K_flat_fp8, active_indices, DenseKV_scratch.data(), 128, 64, cfg->scale, out_cpu.data());

    auto gold_bin = read_file("sparse_bins/out_golden.bin");
    float* gold = reinterpret_cast<float*>(gold_bin.data());
    size_t per_batch = (size_t)cfg->H * Dv;
    int mismatch_count = 0;
    double max_diff = 0.0;
    for (int b = 0; b < B_test; b++) {
        for (size_t i = 0; i < per_batch; i++) {
            float cpu = bf16_to_fp32(out_cpu[(size_t)b * per_batch + i]);
            if (std::isnan(cpu)) {
                std::cout << "❌ NaN at b=" << b << " i=" << i << "\n"; return 1;
            }
            double d = std::abs(cpu - gold[i]);
            if (d > max_diff) max_diff = d;
            if (d > 0.05 && mismatch_count < 10) {
                std::cout << "❌ b=" << b << " i=" << i << ": CPU=" << cpu << " GPU=" << gold[i] << "\n";
                mismatch_count++;
            }
        }
    }
    std::cout << "🔎 Max Diff (across " << B_test << " batches): " << max_diff
              << (max_diff < 0.1 ? " (PASS)" : " (FAIL)") << std::endl;

    int iterations = 50;
    std::cout << "⏱️  BENCHMARKING " << iterations << " ITERATIONS..." << std::endl;
    // // Measure End-to-End
    double total_start = omp_get_wtime();
    for(int i = 0; i < iterations; i++) {
    sparse_attention_amx(Q, B_test, cfg->D, cfg->NumTokens, K_flat_fp8, active_indices, DenseKV_scratch.data(), 128, 64, cfg->scale, out_cpu.data());
    }
    double total_end = (omp_get_wtime() - total_start);
    printf("\n========= CUSTOM SPARSE AMX PIPELINE ========\n");
    // Multiplied by 1000 to convert seconds to milliseconds
    printf("---------------------------------------------\n");
    printf("Total End-to-End Time   | %8.4f ms\n", (total_end / iterations) * 1000.0);
    printf("=============================================\n\n");

    return 0;
}