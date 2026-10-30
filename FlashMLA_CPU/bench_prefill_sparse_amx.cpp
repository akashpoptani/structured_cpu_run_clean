// SPARSE prefill attention, custom AMX, over the ABSORBED (MQA) latent.
//
// Shape follows src/overrides/mla_absorbed.py, which puts prefill on the
// absorbed path: Q[B][Sq][H][512 + 64], cache rows [B][Skv][512 + 64] SHARED by
// all heads, V = the first 512 dims, per-query top-K indices.
//
// AMX rows = 16 HEADS of ONE query, the same layout as the decode kernel
// (bench_sparse_amx.cpp). Every head of a query attends to the same top-K
// tokens, so a 16-head tile has one key set and nothing is masked. This is also
// how DeepSeek's FlashMLA tiles sparse prefill on GPU (64 heads of one query per
// block). One task = one (batch row, query); H/16 head tiles share each block.
//
// Per query: drop future indices (causal), then walk the kept tokens 64 at a
// time. Each 64-token block is gathered from the cache and packed for AMX just
// before use (136 KB, stays in L2), consumed by every head tile, then replaced.
//
// Measured against the previous union-of-16-queries layout on real 4K and
// synthetic 32K top-k (H=64, Sq=2048, K=2048): 2.2-3.8x faster at B=1 and
// 1.8-2.8x at B=4, 17-19 TFLOP/s at 88 threads.
#include <immintrin.h>
#include <omp.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using bf16 = uint16_t;
#define XFEATURE_XTILEDATA 18
#define ARCH_GET_XCOMP_PERM 0x1022
#define ARCH_REQ_XCOMP_PERM 0x1023
static bool init_amx() {
    unsigned long bm = 0;
    if (syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bm) != 0) return false;
    if (bm & (1UL << XFEATURE_XTILEDATA)) return true;
    return syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) == 0;
}
struct TileConfig { uint8_t palette_id, start_row, reserved[14]; uint16_t colsb[16]; uint8_t rows[16]; };
struct SparseCfg { int B, H, Sq, Skv, Dlat, K; float scale; int start_pos; };

static std::vector<char> read_file(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) { std::cerr << "FATAL: cannot open " << p << "\n"; exit(1); }
    std::streamsize n = f.tellg(); f.seekg(0, std::ios::beg);
    std::vector<char> b(n); f.read(b.data(), n); return b;
}
static std::string dir_() { const char* e = getenv("SPARSE_DIR");
    return std::string(e ? e : "prefill_sparse_bins") + "/"; }
static bool exists_(const std::string& p) { std::ifstream f(p); return (bool)f; }

static inline __m512 fast_exp_512(__m512 x) {
    x = _mm512_min_ps(_mm512_max_ps(x, _mm512_set1_ps(-88.f)), _mm512_set1_ps(88.f));
    __m512 t = _mm512_mul_ps(x, _mm512_set1_ps(1.442695f));
    __m512 r = _mm512_roundscale_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 f = _mm512_sub_ps(t, r), p = _mm512_set1_ps(0.00000198f);
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(0.0001398f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(0.0083334f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(0.0416657f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(0.1666666f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(0.5f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.0f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.0f));
    return _mm512_scalef_ps(p, _mm512_cvtepi32_ps(_mm512_cvtps_epi32(r)));
}
// 16x16 dword transpose == AMX B-tile VNNI. Verified element-exact (job 27925178).
static inline void pack_k_chunk_amx(const bf16* s, bf16* d, int stride) {
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

// CAUSALITY: query s sits at absolute position start_pos + s and
// may attend only to t <= that. topk alone does NOT guarantee it: upstream takes
// topk(min(2048, end_pos)) AFTER adding the causal mask, so any query with fewer
// than 2048 past positions gets its set padded with FUTURE positions (scored -inf
// but still selected). Upstream and mla_absorbed.attention_torch remove them by adding the
// causal mask back; this kernel has no mask, so it must drop them here. The
// unsigned compare also rejects t < 0.
//
// OPERANDS -- each is the tensor mla_absorbed.forward_prefill already holds, so the
// Python side passes pointers and copies nothing:
//   Qn   [B][Sq][H][512]  bf16   absorbed q_nope      (was cat'ed with q_pe: 2.4 GB/layer)
//   Qp   [B][Sq][H][64]   bf16   q_pe
//   KV   [B][kv_stride][512]     the kv_cache slab; rows >= Skv are never read
//   PE   [B][kv_stride][64]      the pe_cache slab  (was sliced+copied at B>1)
//   topk [B][Sq][K]       int64  straight from torch.topk (was cast to int32)
//   Out  [B][Sq][H][512]  bf16   what the next einsum consumes (was fp32 + cast)
void prefill_sparse_amx(const bf16* Qn, const bf16* Qp, const bf16* KV, const bf16* PE,
                        const int64_t* topk, bf16* Out, int B, int H, int Sq, int Skv,
                        int kv_stride, int K, float scale, int start_pos) {
    const int Dlat = 576, DV = 512, HT = H / 16;
    (void)Skv;

    #pragma omp parallel
    {
        TileConfig cfg = {};
        cfg.palette_id = 1;
        for (int i = 0; i < 8; i++) { cfg.rows[i] = 16; cfg.colsb[i] = 64; }
        _tile_loadconfig(&cfg);

        std::vector<int32_t> keep(K);                                  // this query's kept indices
        std::vector<bf16> gath((size_t)64 * Dlat), k_pack((size_t)64 * Dlat), v_pack((size_t)32 * 1024);
        std::vector<float> accbuf((size_t)H * DV), mbuf(H), lbuf(H);
        alignas(64) float scores[16][64];
        alignas(64) bf16  probs[16][64];

        #pragma omp for collapse(2) schedule(dynamic, 4)
        for (int b = 0; b < B; b++)
        for (int s = 0; s < Sq; s++) {
            const int64_t* ti = topk + ((size_t)b * Sq + s) * K;
            const uint64_t qpos = (uint64_t)(start_pos + s);
            int n = 0;
            for (int j = 0; j < K; j++)
                if ((uint64_t)ti[j] <= qpos) keep[n++] = (int32_t)ti[j];   // causal filter

            std::fill(accbuf.begin(), accbuf.end(), 0.f);
            for (int h = 0; h < H; h++) { mbuf[h] = -INFINITY; lbuf[h] = 0.f; }
            const bf16* kvb = KV + (size_t)b * kv_stride * 512;
            const bf16* peb = PE + (size_t)b * kv_stride * 64;
            const size_t qrow = ((size_t)b * Sq + s) * H;               // row of head 0

            for (int t0 = 0; t0 < n; t0 += 64) {
                const int valid = std::min(64, n - t0);
                // ---- gather + pack THIS block only (all head tiles reuse it)
                for (int u = 0; u < 64; u++) {
                    bf16* dst = &gath[(size_t)u * Dlat];
                    if (u < valid) {
                        std::memcpy(dst,       kvb + (size_t)keep[t0 + u] * 512, 512 * sizeof(bf16));
                        std::memcpy(dst + 512, peb + (size_t)keep[t0 + u] *  64,  64 * sizeof(bf16));
                    } else std::memset(dst, 0, Dlat * sizeof(bf16));
                }
                for (int t = 0; t < 64; t += 16)
                    for (int d = 0; d < Dlat; d += 32)
                        pack_k_chunk_amx(&gath[(size_t)t * Dlat + d], &k_pack[(size_t)t * Dlat + d * 16], Dlat);
                for (int d_blk = 0; d_blk < DV; d_blk += 16)
                    for (int t_blk = 0; t_blk < 64; t_blk += 32) {
                        bf16* dst = &v_pack[(d_blk / 16) * 1024 + (t_blk / 32) * 512];
                        for (int r = 0; r < 16; r++) {
                            const bf16* s0 = &gath[(size_t)(t_blk + r * 2) * Dlat + d_blk];
                            __m256i a0 = _mm256_loadu_si256((const __m256i*)s0);
                            __m256i a1 = _mm256_loadu_si256((const __m256i*)(s0 + Dlat));
                            _mm512_storeu_si512((void*)(dst + r * 32),
                                _mm512_or_si512(_mm512_cvtepu16_epi32(a0),
                                                _mm512_slli_epi32(_mm512_cvtepu16_epi32(a1), 16)));
                        }
                    }

                for (int ht = 0; ht < HT; ht++) {
                    const int h0 = ht * 16;
                    float* acc = &accbuf[(size_t)h0 * DV];
                    float* m_run = &mbuf[h0];
                    float* l_run = &lbuf[h0];
                    // QK: A tile = 16 heads x 32 dims, straight from the model's Q
                    // (heads are adjacent rows: 1 KB apart in Qn, 128 B in Qp)
                    _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
                    for (int k = 0; k < Dlat; k += 32) {
                        if (k < 512) _tile_loadd(4, Qn + (qrow + h0) * 512 + k, 1024);
                        else         _tile_loadd(4, Qp + (qrow + h0) * 64 + (k - 512), 128);
                        _tile_loadd(5, &k_pack[(size_t) 0 * Dlat + k * 16], 64); _tile_dpbf16ps(0, 4, 5);
                        _tile_loadd(6, &k_pack[(size_t)16 * Dlat + k * 16], 64); _tile_dpbf16ps(1, 4, 6);
                        _tile_loadd(7, &k_pack[(size_t)32 * Dlat + k * 16], 64); _tile_dpbf16ps(2, 4, 7);
                        _tile_loadd(5, &k_pack[(size_t)48 * Dlat + k * 16], 64); _tile_dpbf16ps(3, 4, 5);
                    }
                    _tile_stored(0, &scores[0][ 0], 256); _tile_stored(1, &scores[0][16], 256);
                    _tile_stored(2, &scores[0][32], 256); _tile_stored(3, &scores[0][48], 256);

                    for (int i = 0; i < 16; i++) {                  // i = head within the tile
                        float mb = -INFINITY;
                        for (int j = 0; j < 64; j++) {
                            float sv = (j < valid) ? scores[i][j] * scale : -INFINITY;
                            scores[i][j] = sv;
                            if (sv > mb) mb = sv;
                        }
                        float mn = std::max(m_run[i], mb);
                        float al = (m_run[i] == -INFINITY) ? 0.f : std::exp(m_run[i] - mn);
                        float lb = 0.f; __m512 mv = _mm512_set1_ps(mn);
                        for (int j = 0; j < 64; j += 16) {
                            __m512 pv = fast_exp_512(_mm512_sub_ps(_mm512_load_ps(&scores[i][j]), mv));
                            lb += _mm512_reduce_add_ps(pv);
                            _mm256_storeu_si256((__m256i*)&probs[i][j], (__m256i)_mm512_cvtneps_pbh(pv));
                        }
                        if (al != 1.f) {
                            __m512 av = _mm512_set1_ps(al);
                            for (int d = 0; d < DV; d += 16)
                                _mm512_storeu_ps(&acc[i*DV + d], _mm512_mul_ps(_mm512_loadu_ps(&acc[i*DV + d]), av));
                        }
                        m_run[i] = mn; l_run[i] = l_run[i] * al + lb;
                    }

                    _tile_loadd(4, &probs[0][0], 128); _tile_loadd(5, &probs[0][32], 128);
                    for (int d = 0; d < DV; d += 64) {
                        _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
                        const bf16* v0 = &v_pack[((d +  0)/16) * 1024];
                        const bf16* v1 = &v_pack[((d + 16)/16) * 1024];
                        const bf16* v2 = &v_pack[((d + 32)/16) * 1024];
                        const bf16* v3 = &v_pack[((d + 48)/16) * 1024];
                        _tile_loadd(6, v0,       64); _tile_dpbf16ps(0, 4, 6);
                        _tile_loadd(7, v0 + 512, 64); _tile_dpbf16ps(0, 5, 7);
                        _tile_loadd(6, v1,       64); _tile_dpbf16ps(1, 4, 6);
                        _tile_loadd(7, v1 + 512, 64); _tile_dpbf16ps(1, 5, 7);
                        _tile_loadd(6, v2,       64); _tile_dpbf16ps(2, 4, 6);
                        _tile_loadd(7, v2 + 512, 64); _tile_dpbf16ps(2, 5, 7);
                        _tile_loadd(6, v3,       64); _tile_dpbf16ps(3, 4, 6);
                        _tile_loadd(7, v3 + 512, 64); _tile_dpbf16ps(3, 5, 7);
                        alignas(64) float dl[4][16][16];
                        _tile_stored(0, &dl[0][0][0], 64); _tile_stored(1, &dl[1][0][0], 64);
                        _tile_stored(2, &dl[2][0][0], 64); _tile_stored(3, &dl[3][0][0], 64);
                        for (int u = 0; u < 4; u++) for (int i = 0; i < 16; i++)
                            _mm512_storeu_ps(&acc[i*DV + d + u*16],
                                _mm512_add_ps(_mm512_loadu_ps(&acc[i*DV + d + u*16]),
                                              _mm512_load_ps(&dl[u][i][0])));
                    }
                }   // head tiles
            }       // blocks
            for (int h = 0; h < H; h++) {
                bf16* o = Out + (qrow + h) * DV;
                const float* acc = &accbuf[(size_t)h * DV];
                __m512 inv = _mm512_set1_ps(lbuf[h] > 0.f ? 1.f / lbuf[h] : 0.f);
                for (int d = 0; d < DV; d += 16)
                    _mm256_storeu_si256((__m256i*)(o + d), (__m256i)_mm512_cvtneps_pbh(
                        _mm512_mul_ps(_mm512_loadu_ps(&acc[d]), inv)));
            }
        }
    }
}

// ---- C ABI for the Python side (src/overrides/amx_attention.py). Operands as
// documented on prefill_sparse_amx; Skv = end_pos, kv_stride = the cache's max_seq_len.
extern "C" int flashmla_prefill_sparse(
    const uint16_t* Qn, const uint16_t* Qp, const uint16_t* KV, const uint16_t* PE,
    const int64_t* topk, uint16_t* Out, int B, int H, int Sq, int Skv, int kv_stride,
    int K, float scale, int start_pos) {
    if (start_pos < 0 || start_pos + Sq > Skv || Skv > kv_stride) return -2;
    if (H % 16) return -3;                     // head tiles are 16 rows
    static bool ok = init_amx();
    if (!ok) return -1;
    prefill_sparse_amx((const bf16*)Qn, (const bf16*)Qp, (const bf16*)KV, (const bf16*)PE,
                       topk, (bf16*)Out, B, H, Sq, Skv, kv_stride, K, scale, start_pos);
    return 0;
}

#ifndef FLASHMLA_SHARED_LIB
int main() {
    if (!init_amx()) { std::cerr << "FATAL: AMX permission denied\n"; return 1; }
    auto cr = read_file(dir_() + "config.bin");
    SparseCfg c{};
    std::memcpy(&c, cr.data(), std::min(cr.size(), sizeof(c)));
    if (cr.size() < sizeof(c)) {   // config.bin from before start_pos existed
        c.start_pos = c.Skv - c.Sq;
        printf("  (old config.bin: no start_pos, assuming Skv-Sq=%d; regenerate the bins)\n", c.start_pos);
    }
    printf("PREFILL SPARSE AMX | B=%d H=%d Sq=%d Skv=%d Dlat=%d K=%d scale=%.8f start_pos=%d\n",
           c.B, c.H, c.Sq, c.Skv, c.Dlat, c.K, c.scale, c.start_pos);
    if (c.H % 16) { std::cerr << "FATAL: H=" << c.H << " is not a multiple of 16\n"; return 1; }
    auto qr = read_file(dir_() + "q.bin");
    auto kr = read_file(dir_() + "kv.bin");
    auto tr = read_file(dir_() + "topk.bin");
    // The bins hold packed Q[..][576], KV[Skv][576] and int32 topk; the kernel takes
    // the model's layout (q_nope / q_pe, kv / pe, int64), so split them once here.
    const bf16* Qcat = (const bf16*)qr.data(); const bf16* KVcat = (const bf16*)kr.data();
    const size_t nq = (size_t)c.B * c.Sq * c.H, nkv = (size_t)c.B * c.Skv;
    std::vector<bf16> qn(nq * 512), qp(nq * 64), kvn(nkv * 512), pe(nkv * 64);
    for (size_t r = 0; r < nq; r++) {
        std::memcpy(&qn[r * 512], Qcat + r * 576,       512 * sizeof(bf16));
        std::memcpy(&qp[r * 64],  Qcat + r * 576 + 512,  64 * sizeof(bf16));
    }
    for (size_t r = 0; r < nkv; r++) {
        std::memcpy(&kvn[r * 512], KVcat + r * 576,       512 * sizeof(bf16));
        std::memcpy(&pe[r * 64],   KVcat + r * 576 + 512,  64 * sizeof(bf16));
    }
    const int32_t* tk32 = (const int32_t*)tr.data();
    std::vector<int64_t> tk(tk32, tk32 + (size_t)c.B * c.Sq * c.K);
    std::vector<bf16> out(nq * 512);
    auto run = [&] {
        prefill_sparse_amx(qn.data(), qp.data(), kvn.data(), pe.data(), tk.data(), out.data(),
                           c.B, c.H, c.Sq, c.Skv, c.Skv, c.K, c.scale, c.start_pos);
    };

    run();
    if (exists_(dir_() + "out_golden.bin")) {
        auto gr = read_file(dir_() + "out_golden.bin");
        const float* g = (const float*)gr.data();
        auto f32 = [](bf16 v) { uint32_t u = (uint32_t)v << 16; float f; std::memcpy(&f, &u, 4); return f; };
        double md = 0; for (size_t i = 0; i < out.size(); i++) md = std::max(md, (double)std::fabs(f32(out[i])-g[i]));
        printf("  Max Diff: %.8f (%s)\n", md, md < 0.02 ? "PASS" : "FAIL");
    } else printf("  (no golden -- timing only)\n");

    int iters = getenv("BENCH_ITERS") ? atoi(getenv("BENCH_ITERS")) : 5;
    double t0 = omp_get_wtime();
    for (int i = 0; i < iters; i++) run();
    double ms = (omp_get_wtime() - t0) / iters * 1000.0;
    double fl = 2.0 * c.B * c.H * (double)c.Sq * c.K * (c.Dlat + 512);
    printf("Total End-to-End Time   | %8.4f ms\n", ms);
    printf("  FLOPs %.2f G | %.2f TFLOP/s\n", fl/1e9, fl/(ms*1e-3)/1e12);
    return 0;
}
#endif  // FLASHMLA_SHARED_LIB
