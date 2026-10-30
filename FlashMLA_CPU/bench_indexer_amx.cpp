// DeepSeek-V3.2 INDEXER (fp8_index) as an AMX kernel.
//
// WHY: measured at 0.22 TFLOP/s in-model, which alone accounts for ~9.1 h of a
// 9.2 h prefill at Lin=120k. The torch override (src/overrides/kernel.py fp8_index)
//     logits = einsum("bmhd,bnd->bmhn", q.float(), k.float())   # [M, 64, N] !!
//     logits.clamp_min_(0); logits *= q_s; logits = logits.sum(2); logits *= k_s
// materialises a PER-HEAD score tensor 64x larger than the result (17 GB at
// M=N=8192, 1.0 TB at M=32768/N=120k) and runs it in fp32, so no AMX. Profiling
// showed the GEMM is 1.4% of the call -- the other 98.6% is four elementwise
// passes over a 268 MB score matrix, 64 times over.
//
// WHAT THIS DOES: out[m,n] = ( sum_h relu(q[m,h,:] . k[n,:]) * q_s[m,h] ) * k_s[n]
// computed tile-by-tile so the 16x16 accumulator never leaves registers:
//   * k is packed to VNNI ONCE per call (2.1 MB at N=8192) and reused by every
//     m-tile and every head
//   * per (m-tile, 4 n-tiles): 4 AMX accumulators, Q loaded once per k-step
//   * relu + q_s scale + head accumulation happen on the 16x16 tile in AVX-512,
//     so the [M,N] score matrix is written exactly ONCE instead of 64 times
//
// Two instantiations, picked by kv_bf16 on the C ABI (KV_CACHE_DTYPE):
//   fp8   KEY_SCALE=true: q and k arrive as fp8 values widened to bf16 (exact:
//         3 mantissa bits into 8), and the result is multiplied by k_s[n].
//   bf16  KEY_SCALE=false: q and k are plain bf16, there is no k_s (KS may be
//         NULL) and the final multiply is compiled out.
//
// PREFILL only: tile rows are 16 QUERIES. At decode M=1 would waste 15/16 of each
// tile, so decode keeps the torch path (kernel.py) until it has its own kernel.
// Scores ALL N keys; the causal mask and top-k stay in torch (model.py).
//
// Standalone: main() below fills fp8-representable random inputs, checks sampled
// rows against a double-precision reference, and times the call (K pack included):
//   IDX_M=2048 IDX_N=4096 ./bench      (also IDX_B, IDX_H, IDX_D, BENCH_ITERS,
//                                       IDX_KV_BF16=1 for the bf16 instantiation)
// Built with -DFLASHMLA_SHARED_LIB (build_libs.sh) it is amx_prefill_indexer.so.
#include <immintrin.h>
#include <omp.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>

using bf16 = uint16_t;
#define XFEATURE_XTILEDATA 18
#define ARCH_GET_XCOMP_PERM 0x1022
#define ARCH_REQ_XCOMP_PERM 0x1023
static bool idx_init_amx() {
    unsigned long bm = 0;
    if (syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bm) != 0) return false;
    if (bm & (1UL << XFEATURE_XTILEDATA)) return true;
    return syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) == 0;
}
struct IdxTileCfg { uint8_t palette_id, start_row, reserved[14]; uint16_t colsb[16]; uint8_t rows[16]; };

// 16x16 dword transpose == AMX B-tile VNNI (verified element-exact, job 27925178)
static inline void idx_pack(const bf16* s, bf16* d, int stride) {
    __m512i a[16], t[16], u[16], v[16];
    for (int c = 0; c < 16; c++) a[c] = _mm512_loadu_si512((const void*)(s + (size_t)c * stride));
    for (int i = 0; i < 8; i++) {
        t[2*i]   = _mm512_unpacklo_epi32(a[2*i], a[2*i+1]);
        t[2*i+1] = _mm512_unpackhi_epi32(a[2*i], a[2*i+1]);
    }
    for (int i = 0; i < 4; i++) {
        u[4*i+0] = _mm512_unpacklo_epi64(t[4*i+0], t[4*i+2]);
        u[4*i+1] = _mm512_unpackhi_epi64(t[4*i+0], t[4*i+2]);
        u[4*i+2] = _mm512_unpacklo_epi64(t[4*i+1], t[4*i+3]);
        u[4*i+3] = _mm512_unpackhi_epi64(t[4*i+1], t[4*i+3]);
    }
    for (int g = 0; g < 2; g++) for (int i = 0; i < 4; i++) {
        int b = g*8+i;
        v[b]   = _mm512_shuffle_i32x4(u[b], u[b+4], 0x88);
        v[b+4] = _mm512_shuffle_i32x4(u[b], u[b+4], 0xdd);
    }
    for (int i = 0; i < 8; i++) {
        _mm512_storeu_si512((void*)(d + (size_t)i     * 32), _mm512_shuffle_i32x4(v[i], v[i+8], 0x88));
        _mm512_storeu_si512((void*)(d + (size_t)(i+8) * 32), _mm512_shuffle_i32x4(v[i], v[i+8], 0xdd));
    }
}

template <bool KEY_SCALE>
void indexer_amx(const bf16* Q, const bf16* K, const float* QS, const float* KS,
                 float* Out, int B, int M, int N, int H, int D) {
    const int NK = D / 32;                 // 4 k-steps for D=128
    const int NT = N / 16;                 // n-tiles
    const size_t q_stride = (size_t)H * D; // elements between consecutive queries

    // pack K once: [n-tile][k-step][16][32], reused by every m-tile and head
    static std::vector<bf16> kp;
    if (kp.size() < (size_t)B * N * D) kp.resize((size_t)B * N * D);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int b = 0; b < B; b++)
        for (int j = 0; j < NT; j++)
            for (int s = 0; s < NK; s++)
                idx_pack(K + ((size_t)b * N + (size_t)j * 16) * D + s * 32,
                         &kp[(((size_t)b * NT + j) * NK + s) * 512], D);

    #pragma omp parallel
    {
        IdxTileCfg cfg = {};
        cfg.palette_id = 1;
        for (int i = 0; i < 8; i++) { cfg.rows[i] = 16; cfg.colsb[i] = 64; }
        _tile_loadconfig(&cfg);
        alignas(64) float sc[4][16][16];
        alignas(64) float acc[4][16][16];

        // collapse(3), not (2): at DECODE M=1 is padded to 16, so M/16 == 1 and
        // B == 1 -- a collapse(2) nest yields exactly ONE iteration and the whole
        // indexer runs single-threaded while 63 cores idle. Including the jt loop
        // gives NT/4 iterations (1875 at N=120000) to spread. Nothing carries
        // across jt: acc is zeroed at the top of each one, so this is safe.
        #pragma omp for collapse(3) schedule(dynamic, 1)
        for (int b = 0; b < B; b++)
        for (int mt = 0; mt < M / 16; mt++) {
            for (int jt = 0; jt < NT; jt += 4) {
                const bf16* qbase = Q + ((size_t)b * M + (size_t)mt * 16) * q_stride;
                const float* qsb  = QS + ((size_t)b * M + (size_t)mt * 16) * H;
                for (int u = 0; u < 4; u++)
                    for (int i = 0; i < 16; i++)
                        _mm512_store_ps(&acc[u][i][0], _mm512_setzero_ps());

                for (int h = 0; h < H; h++) {
                    _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
                    for (int s = 0; s < NK; s++) {
                        _tile_loadd(4, qbase + (size_t)h * D + s * 32, q_stride * 2);
                        _tile_loadd(5, &kp[(((size_t)b*NT + jt+0)*NK + s)*512], 64); _tile_dpbf16ps(0, 4, 5);
                        _tile_loadd(6, &kp[(((size_t)b*NT + jt+1)*NK + s)*512], 64); _tile_dpbf16ps(1, 4, 6);
                        _tile_loadd(7, &kp[(((size_t)b*NT + jt+2)*NK + s)*512], 64); _tile_dpbf16ps(2, 4, 7);
                        _tile_loadd(5, &kp[(((size_t)b*NT + jt+3)*NK + s)*512], 64); _tile_dpbf16ps(3, 4, 5);
                    }
                    _tile_stored(0, &sc[0][0][0], 64); _tile_stored(1, &sc[1][0][0], 64);
                    _tile_stored(2, &sc[2][0][0], 64); _tile_stored(3, &sc[3][0][0], 64);
                    // relu, scale by q_s[m,h], accumulate over heads -- all in registers
                    for (int i = 0; i < 16; i++) {
                        __m512 qs = _mm512_set1_ps(qsb[(size_t)i * H + h]);
                        for (int u = 0; u < 4; u++) {
                            __m512 s0 = _mm512_max_ps(_mm512_load_ps(&sc[u][i][0]), _mm512_setzero_ps());
                            _mm512_store_ps(&acc[u][i][0],
                                _mm512_fmadd_ps(s0, qs, _mm512_load_ps(&acc[u][i][0])));
                        }
                    }
                }
                // fp8: scale by k_s[n]; bf16: no key scale. Write the score matrix ONCE.
                for (int u = 0; u < 4; u++) {
                    __m512 ks;
                    if constexpr (KEY_SCALE) ks = _mm512_loadu_ps(KS + (size_t)b * N + (size_t)(jt+u) * 16);
                    for (int i = 0; i < 16; i++) {
                        __m512 v = _mm512_load_ps(&acc[u][i][0]);
                        if constexpr (KEY_SCALE) v = _mm512_mul_ps(v, ks);
                        _mm512_storeu_ps(Out + ((size_t)b * M + (size_t)mt*16 + i) * N + (size_t)(jt+u)*16, v);
                    }
                }
            }
        }
    }
}

extern "C" int flashmla_indexer(const uint16_t* Q, const uint16_t* K,
                                const float* QS, const float* KS, float* Out,
                                int B, int M, int N, int H, int D, int kv_bf16) {
    static bool ok = idx_init_amx();
    if (!ok) return -1;
    if (N % 64 || M % 16 || D % 32) return -2;   // caller pads; see the Python side
    if (kv_bf16) indexer_amx<false>((const bf16*)Q, (const bf16*)K, QS, nullptr, Out, B, M, N, H, D);
    else         indexer_amx<true >((const bf16*)Q, (const bf16*)K, QS, KS,      Out, B, M, N, H, D);
    return 0;
}

#ifndef FLASHMLA_SHARED_LIB
#include <chrono>
#include <cstdlib>
#include <random>
static int env_int(const char* k, int d) { const char* v = getenv(k); return v && *v ? atoi(v) : d; }
// float -> bf16 bits, keeping 3 mantissa bits so every value is fp8_e4m3-representable
static bf16 to_fp8ish(float f) { uint32_t u; memcpy(&u, &f, 4); return (bf16)((u >> 16) & 0xFFF0); }
// float -> bf16 bits, all 7 mantissa bits (truncation; any bf16 value is a valid input)
static bf16 to_bf16(float f) { uint32_t u; memcpy(&u, &f, 4); return (bf16)(u >> 16); }
static float bf2f(bf16 b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }

int main() {
    const int B = env_int("IDX_B", 1), M = env_int("IDX_M", 2048), N = env_int("IDX_N", 4096);
    const int H = env_int("IDX_H", 64), D = env_int("IDX_D", 128), iters = env_int("BENCH_ITERS", 3);
    const int kv_bf16 = env_int("IDX_KV_BF16", 0);
    printf("B=%d M=%d N=%d H=%d D=%d threads=%d kv=%s\n", B, M, N, H, D, omp_get_max_threads(),
           kv_bf16 ? "bf16" : "fp8");
    auto cvt = kv_bf16 ? to_bf16 : to_fp8ish;
    std::mt19937 rng(0);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<bf16> Q((size_t)B * M * H * D), K((size_t)B * N * D);
    std::vector<float> QS((size_t)B * M * H), KS((size_t)B * N), Out((size_t)B * M * N);
    for (auto& x : Q) x = cvt(nd(rng));
    for (auto& x : K) x = cvt(nd(rng));
    for (auto& x : QS) x = 0.01f * nd(rng);               // weights: signed
    for (auto& x : KS) x = kv_bf16 ? 1.f : 0.5f + 0.01f * std::abs(nd(rng)); // k scales: positive
    const float* ksp = kv_bf16 ? nullptr : KS.data();   // bf16: the kernel must not read KS
    int rc = flashmla_indexer(Q.data(), K.data(), QS.data(), ksp, Out.data(), B, M, N, H, D, kv_bf16);
    if (rc != 0) { printf("FATAL: rc=%d\n", rc); return 1; }

    // check 64 sampled rows (all N columns) against a double reference
    double max_diff = 0, max_ref = 0;
    #pragma omp parallel for reduction(max:max_diff, max_ref)
    for (int r = 0; r < 64; r++) {
        int b = r % B, m = (int)(((long long)r * 7919) % M);
        for (int n = 0; n < N; n++) {
            double acc = 0;
            for (int h = 0; h < H; h++) {
                double dot = 0;
                const bf16* q = &Q[(((size_t)b * M + m) * H + h) * D];
                const bf16* k = &K[((size_t)b * N + n) * D];
                for (int d = 0; d < D; d++) dot += (double)bf2f(q[d]) * bf2f(k[d]);
                acc += std::max(dot, 0.0) * QS[((size_t)b * M + m) * H + h];
            }
            acc *= KS[(size_t)b * N + n];
            max_diff = std::max(max_diff, std::abs(acc - Out[((size_t)b * M + m) * N + n]));
            max_ref = std::max(max_ref, std::abs(acc));
        }
    }
    double rel = max_diff / (max_ref + 1e-30);
    printf("Max Diff %.3e (max |ref| %.3e, rel %.2e)  %s\n", max_diff, max_ref, rel, rel < 1e-5 ? "PASS" : "FAIL");

    double best = 1e30;
    for (int i = 0; i < iters; i++) {
        auto t0 = std::chrono::steady_clock::now();
        flashmla_indexer(Q.data(), K.data(), QS.data(), ksp, Out.data(), B, M, N, H, D, kv_bf16);
        best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    double flop = 2.0 * B * M * N * H * D;
    printf("time %.2f ms  %.2f TFLOP/s (best of %d, K pack included)\n", best * 1e3, flop / best / 1e12, iters);
    return rel < 1e-5 ? 0 : 1;
}
#endif  // FLASHMLA_SHARED_LIB
