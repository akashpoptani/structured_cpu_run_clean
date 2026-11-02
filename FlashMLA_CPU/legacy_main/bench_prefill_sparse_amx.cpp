// SPARSE prefill attention, custom AMX, union-gather over the ABSORBED latent.
//
// Shape follows src/overrides/chunked_prefill.py, which puts prefill on the
// absorbed (MQA) path: Q[B][Sq][H][576], KV[B][Skv][576] SHARED across heads,
// V = first 512 dims, per-query top-K indices. Same 576/512 geometry as the
// decode kernel, so the AMX inner loop carries over -- what changes is that M
// is 16 QUERIES rather than 16 heads, and each query has its own index set.
//
// Structure chosen from MEASURED data (job 27925600, real indexer dump,
// analyze_topk_overlap.py), not assumed:
//     adjacent-query overlap  0.91 - 0.94
//     tile union factor       1.20 - 1.35   (p90 1.59)
//     block touch rate        1.000
// The last number kills mask-over-dense outright: every causally reachable
// 64-token block holds at least one selected token, so block-skipping saves
// nothing. The first two say a 16-query tile can gather the UNION of its 16
// index sets and pay only ~20-35% over a single query's 2048.
//
// The union is gathered ONCE per query tile and reused by all 128 heads,
// because the absorbed latent is head-independent. That is the whole reason to
// prefer this form: in the per-head MHA form the same gather would be repeated
// 128 times, and the K/V it gathers would be 10.7 GB per layer at 128k.
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
struct SparseCfg { int B, H, Sq, Skv, Dlat, K; float scale; };

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

void prefill_sparse_amx(const bf16* Q, const bf16* KV, const bf16* PE, const int32_t* topk, float* Out,
                        int B, int H, int Sq, int Skv, int Dlat, int K, float scale) {
    const int DV = 512, QT = (Sq + 15) / 16;
    const size_t q_row = (size_t)H * Dlat;        // stride between query positions

    #pragma omp parallel
    {
        TileConfig cfg = {};
        cfg.palette_id = 1;
        for (int i = 0; i < 8; i++) { cfg.rows[i] = 16; cfg.colsb[i] = 64; }
        _tile_loadconfig(&cfg);

        std::vector<uint8_t>  bits((Skv + 7) / 8);
        std::vector<int32_t>  pos_of(Skv);
        std::vector<int32_t>  uni; uni.reserve((size_t)K * 2);
        std::vector<bf16>     gath((size_t)K * 2 * Dlat);        // union latent, shared by all heads
        std::vector<uint64_t> rowmask((size_t)16 * (((size_t)K * 2 + 63) / 64));
        // Packed K and V for EVERY union block, built once per query tile and
        // reused by all 128 heads. Both packs depend only on the gathered block,
        // never on h, so leaving them in the head loop packed each block 128x.
        std::vector<bf16> k_all, v_all;
        alignas(64) float scores[16][64];
        alignas(64) bf16  probs[16][64];
        // Head-GROUP state. The head loop used to walk the whole 5.71 MB of
        // packed k_all+v_all once per head, 128 times per query tile -- 365 MB
        // per tile, and at 64 threads that is 365 MB against ~105 MB of L3.
        // Measured consequence: 748 GB/layer/chunk at 189 GB/s, i.e. exactly
        // DRAM bandwidth. Processing HG heads per pass over the blocks cuts
        // that by HG. HG=8 keeps acc (262 KB) + one block (140 KB) inside L2.
        const int HG = []{ const char* e = getenv("FLASHMLA_HEAD_GROUP");
                           return e ? std::max(1, atoi(e)) : 8; }();
        std::vector<float> accbuf((size_t)HG * 16 * DV);
        std::vector<float> mbuf((size_t)HG * 16), lbuf((size_t)HG * 16);

        #pragma omp for collapse(2) schedule(dynamic, 1)
        for (int b = 0; b < B; b++)
        for (int qt = 0; qt < QT; qt++) {
            const int q0 = qt * 16, qr = std::min(16, Sq - q0);

            // ---- union of this tile's 16 index sets, gathered ONCE for all heads
            std::fill(bits.begin(), bits.end(), 0);
            for (int i = 0; i < qr; i++) {
                const int32_t* ti = topk + ((size_t)b * Sq + q0 + i) * K;
                for (int j = 0; j < K; j++) { int t = ti[j]; bits[t >> 3] |= uint8_t(1u << (t & 7)); }
            }
            uni.clear();
            for (int t = 0; t < Skv; t++)
                if (bits[t >> 3] & (1u << (t & 7))) { pos_of[t] = (int32_t)uni.size(); uni.push_back(t); }
            const int U = (int)uni.size();
            const int UB = (U + 63) / 64;                     // 64-token blocks
            const size_t mw = (size_t)(UB * 64 + 63) / 64;    // mask words per row
            if (gath.size() < (size_t)UB * 64 * Dlat) gath.resize((size_t)UB * 64 * Dlat);
            if (rowmask.size() < 16 * mw) rowmask.resize(16 * mw);
            std::fill(rowmask.begin(), rowmask.begin() + 16 * mw, 0ULL);
            for (int i = 0; i < qr; i++) {
                const int32_t* ti = topk + ((size_t)b * Sq + q0 + i) * K;
                uint64_t* rm = &rowmask[(size_t)i * mw];
                for (int j = 0; j < K; j++) { int p = pos_of[ti[j]]; rm[p >> 6] |= 1ULL << (p & 63); }
            }
            const bf16* kvb = KV + (size_t)b * Skv * 512;
            const bf16* peb = PE ? PE + (size_t)b * Skv * 64 : nullptr;
            for (int u = 0; u < U; u++) {
                bf16* dst = &gath[(size_t)u * Dlat];
                if (peb) {   // split caches: concatenate during the gather
                    std::memcpy(dst,       kvb + (size_t)uni[u] * 512, 512 * sizeof(bf16));
                    std::memcpy(dst + 512, peb + (size_t)uni[u] *  64,  64 * sizeof(bf16));
                } else {     // pre-concatenated [Skv][576]
                    std::memcpy(dst, KV + ((size_t)b * Skv + uni[u]) * Dlat, Dlat * sizeof(bf16));
                }
            }
            for (int u = U; u < UB * 64; u++)                 // pad tail block with zeros
                std::memset(&gath[(size_t)u * Dlat], 0, Dlat * sizeof(bf16));

            // ---- pack every block ONCE for this tile; all heads reuse it
            if (k_all.size() < (size_t)UB * 64 * Dlat) k_all.resize((size_t)UB * 64 * Dlat);
            if (v_all.size() < (size_t)UB * 32 * 1024) v_all.resize((size_t)UB * 32 * 1024);
            for (int ub = 0; ub < UB; ub++) {
                const bf16* blk = &gath[(size_t)ub * 64 * Dlat];
                const int valid = std::min(64, U - ub * 64);
                bf16* kp = &k_all[(size_t)ub * 64 * Dlat];
                for (int t = 0; t < 64; t += 16)
                    for (int d = 0; d < Dlat; d += 32)
                        pack_k_chunk_amx(blk + (size_t)t * Dlat + d,
                                         &kp[(size_t)t * Dlat + d * 16], Dlat);
                bf16* vp = &v_all[(size_t)ub * 32 * 1024];
                for (int d_blk = 0; d_blk < DV; d_blk += 16)
                    for (int t_blk = 0; t_blk < 64; t_blk += 32) {
                        bf16* dst = &vp[(d_blk / 16) * 1024 + (t_blk / 32) * 512];
                        for (int r = 0; r < 16; r++) {
                            int tok = t_blk + r * 2;
                            const bf16* s0 = blk + (size_t)tok * Dlat + d_blk;
                            __m256i a0 = (tok     < valid) ? _mm256_loadu_si256((const __m256i*)s0)        : _mm256_setzero_si256();
                            __m256i a1 = (tok + 1 < valid) ? _mm256_loadu_si256((const __m256i*)(s0+Dlat)) : _mm256_setzero_si256();
                            _mm512_storeu_si512((void*)(dst + r * 32),
                                _mm512_or_si512(_mm512_cvtepu16_epi32(a0),
                                                _mm512_slli_epi32(_mm512_cvtepu16_epi32(a1), 16)));
                        }
                    }
            }

            for (int hg = 0; hg < H; hg += HG) {
              const int hn = std::min(HG, H - hg);
              std::fill(accbuf.begin(), accbuf.begin() + (size_t)hn * 16 * DV, 0.f);
              for (int t = 0; t < hn * 16; t++) { mbuf[t] = -INFINITY; lbuf[t] = 0.f; }

              // blocks OUTER, heads INNER: each packed block is read once and
              // consumed by all hn heads while it is still in L2
              for (int ub = 0; ub < UB; ub++) {
                const int valid = std::min(64, U - ub * 64);
                const bf16* k_pack = &k_all[(size_t)ub * 64 * Dlat];
                const bf16* v_pack = &v_all[(size_t)ub * 32 * 1024];

                for (int hh = 0; hh < hn; hh++) {
                    const int h = hg + hh;
                    const bf16* q_ptr = Q + ((size_t)(b * Sq + q0) * H + h) * Dlat;
                    float* acc   = &accbuf[(size_t)hh * 16 * DV];
                    float* m_run = &mbuf[(size_t)hh * 16];
                    float* l_run = &lbuf[(size_t)hh * 16];

                    _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
                    for (int k = 0; k < Dlat; k += 32) {
                        _tile_loadd(4, q_ptr + k, q_row * 2);
                        _tile_loadd(5, &k_pack[(size_t) 0 * Dlat + k * 16], 64); _tile_dpbf16ps(0, 4, 5);
                        _tile_loadd(6, &k_pack[(size_t)16 * Dlat + k * 16], 64); _tile_dpbf16ps(1, 4, 6);
                        _tile_loadd(7, &k_pack[(size_t)32 * Dlat + k * 16], 64); _tile_dpbf16ps(2, 4, 7);
                        _tile_loadd(5, &k_pack[(size_t)48 * Dlat + k * 16], 64); _tile_dpbf16ps(3, 4, 5);
                    }
                    _tile_stored(0, &scores[0][ 0], 256); _tile_stored(1, &scores[0][16], 256);
                    _tile_stored(2, &scores[0][32], 256); _tile_stored(3, &scores[0][48], 256);

                    bool any = false;
                    for (int i = 0; i < qr; i++) {
                        const uint64_t* rm = &rowmask[(size_t)i * mw];
                        float mb = -INFINITY;
                        for (int j = 0; j < 64; j++) {
                            int p = ub * 64 + j;
                            bool on = (j < valid) && (rm[p >> 6] >> (p & 63)) & 1ULL;
                            float sv = on ? scores[i][j] * scale : -INFINITY;
                            scores[i][j] = sv;
                            if (sv > mb) mb = sv;
                        }
                        if (mb == -INFINITY) { std::memset(&probs[i][0], 0, 64 * sizeof(bf16)); continue; }
                        any = true;
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
                                _mm512_storeu_ps(&acc[i*DV + d],
                                    _mm512_mul_ps(_mm512_loadu_ps(&acc[i*DV + d]), av));
                        }
                        m_run[i] = mn; l_run[i] = l_run[i] * al + lb;
                    }
                    for (int i = qr; i < 16; i++) std::memset(&probs[i][0], 0, 64 * sizeof(bf16));
                    if (!any) continue;


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
                }   // heads within the block
              }     // blocks
              for (int hh = 0; hh < hn; hh++) {
                const int h = hg + hh;
                const float* acc = &accbuf[(size_t)hh * 16 * DV];
                const float* l_run = &lbuf[(size_t)hh * 16];
                for (int i = 0; i < qr; i++) {
                    float* o = Out + ((size_t)(b * Sq + q0 + i) * H + h) * DV;
                    __m512 inv = _mm512_set1_ps(l_run[i] > 0.f ? 1.f / l_run[i] : 0.f);
                    for (int d = 0; d < DV; d += 16)
                        _mm512_storeu_ps(o + d, _mm512_mul_ps(_mm512_loadu_ps(&acc[i*DV + d]), inv));
                }
              }
            }     // head groups
        }
    }
}

// ---- C ABI for the Python side (src/overrides/flash_attn_cpu.py).
// The absorbed override already produces exactly these operands:
//   Q  = cat(q_nope_absorbed[512], q_pe[64])        -> [B][Sq][H][576]
//   KV = cat(kv_cache[512], pe_cache[64])           -> [B][Skv][576]
// so no repacking is needed on the Python side beyond those two cats.
extern "C" int flashmla_prefill_sparse(
    const uint16_t* Q, const uint16_t* KV, const uint16_t* PE, const int32_t* topk,
    float* Out, int B, int H, int Sq, int Skv, int Dlat, int K, float scale) {
    static bool ok = init_amx();
    if (!ok) return -1;
    prefill_sparse_amx((const bf16*)Q, (const bf16*)KV, (const bf16*)PE, topk, Out,
                       B, H, Sq, Skv, Dlat, K, scale);
    return 0;
}

#ifndef FLASHMLA_SHARED_LIB
int main() {
    if (!init_amx()) { std::cerr << "FATAL: AMX permission denied\n"; return 1; }
    auto cr = read_file(dir_() + "config.bin");
    SparseCfg c = *reinterpret_cast<SparseCfg*>(cr.data());
    printf("PREFILL SPARSE AMX | B=%d H=%d Sq=%d Skv=%d Dlat=%d K=%d scale=%.8f\n",
           c.B, c.H, c.Sq, c.Skv, c.Dlat, c.K, c.scale);
    auto qr = read_file(dir_() + "q.bin");
    auto kr = read_file(dir_() + "kv.bin");
    auto tr = read_file(dir_() + "topk.bin");
    std::vector<float> out((size_t)c.B * c.Sq * c.H * 512);
    const bf16* Q = (const bf16*)qr.data(); const bf16* KV = (const bf16*)kr.data();
    const int32_t* TK = (const int32_t*)tr.data();

    prefill_sparse_amx(Q, KV, nullptr, TK, out.data(), c.B, c.H, c.Sq, c.Skv, c.Dlat, c.K, c.scale);
    if (exists_(dir_() + "out_golden.bin")) {
        auto gr = read_file(dir_() + "out_golden.bin");
        const float* g = (const float*)gr.data();
        double md = 0; for (size_t i = 0; i < out.size(); i++) md = std::max(md, (double)std::fabs(out[i]-g[i]));
        printf("  Max Diff: %.8f (%s)\n", md, md < 0.02 ? "PASS" : "FAIL");
    } else printf("  (no golden -- timing only)\n");

    int iters = getenv("BENCH_ITERS") ? atoi(getenv("BENCH_ITERS")) : 5;
    double t0 = omp_get_wtime();
    for (int i = 0; i < iters; i++)
        prefill_sparse_amx(Q, KV, nullptr, TK, out.data(), c.B, c.H, c.Sq, c.Skv, c.Dlat, c.K, c.scale);
    double ms = (omp_get_wtime() - t0) / iters * 1000.0;
    double fl = 2.0 * c.B * c.H * (double)c.Sq * c.K * (c.Dlat + 512);
    printf("Total End-to-End Time   | %8.4f ms\n", ms);
    printf("  FLOPs %.2f G | %.2f TFLOP/s\n", fl/1e9, fl/(ms*1e-3)/1e12);
    return 0;
}
#endif  // FLASHMLA_SHARED_LIB
