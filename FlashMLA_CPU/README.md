# FlashMLA_CPU — AMX and OneDNN attention kernels

Four benchmarked AMX / OneDNN attention kernels, self-contained: sources, the `.pt` inputs they
derive from, and the generators that convert them. All numbers below are **measured**; the full
optimisation log they are drawn from is [FlashMLAOptimisationsAndCheck.md](FlashMLAOptimisationsAndCheck.md).

| File | Path | Library | Origin |
|---|---|---|---|
| `bench_sparse_amx.cpp` | sparse | custom AMX intrinsics | `bench_sparse_amx.cpp` |
| `bench_sparse_onednn.cpp` | sparse | OneDNN | `bench_sparse_onednn_correctway.cpp` |
| `bench_dense_amx.cpp` | dense | custom AMX intrinsics | `bench_dense_amx.cpp` |
| `bench_dense_onednn.cpp` | dense | OneDNN | `bench_dense_onednn_correctway.cpp` |

The 15 exploratory `bench_sparse_amx_v*.cpp` variants are **not** vendored; the optimisation ladder
they represent is recorded below.

## Running these benchmarks

Everything needed is in this directory. The chain:

```
{dense,sparse}_{input,output}.pt     captured from an instrumented GPU FlashMLA run
  -> {dense,sparse}_pt_to_bin.py     here
  -> bins/ and sparse_bins/          raw binary, generated
  -> the benchmark binary            reads them cwd-relative
```

The four `.pt` files are present but **untracked** (~259 MB, see [.gitignore](.gitignore)); they are
the source of truth and are not regenerable without a GPU. `bins/` and `sparse_bins/` are derived
and can be rebuilt at any time by the step below.

### 1. Generate the inputs

Run in this directory:

```bash
python dense_pt_to_bin.py     # -> bins/         194 MB (k_cache.bin alone is 169 MB)
python sparse_pt_to_bin.py    # -> sparse_bins/  ~8.5 MB
```

### 2. Build

The **default login-node toolchain cannot build these.** `g++` is GCC 8.5.0, which rejects
`-mamx-tile` / `-mamx-int8` / `-mamx-bf16` outright, and the system assembler is binutils 2.30,
which does not know `tileloadd` / `tdpbf16ps` / `tilestored`. Both must be upgraded:

```bash
module load gcc/13.2.0 binutils/2.45

g++ -O3 -fopenmp -march=sapphirerapids bench_sparse_amx.cpp -o bench_sparse_amx -lstdc++
g++ -O3 -fopenmp -march=sapphirerapids bench_dense_amx.cpp  -o bench_dense_amx  -lstdc++
```

`-march=sapphirerapids` is used instead of listing ISA flags by hand: the explicit list in the
original build line omits `-mavx512bf16`, so `_mm512_cvtneps_pbh` (`bench_sparse_amx.cpp:574`)
fails with `inlining failed in call to always_inline ... target specific option mismatch`.

**Verified:** `bench_sparse_amx` builds clean with the above in ~2.7 s.

The OneDNN pair additionally needs `<dnnl.hpp>` on the include path and `-ldnnl`:

```bash
g++ -O3 -fopenmp -march=sapphirerapids bench_sparse_onednn.cpp -o bench_sparse_onednn -ldnnl -lstdc++
g++ -O3 -fopenmp -march=sapphirerapids bench_dense_onednn.cpp  -o bench_dense_onednn  -ldnnl -lstdc++
```

**Unverified:** no oneDNN module is exposed on this cluster, so the include/link paths for these two
still need to be worked out.

### 3. Run

**Run on a Sapphire Rapids compute node** — the login node reports no `amx_bf16` / `amx_tile`, so
the AMX benchmarks cannot execute there even once built.

**Run from the directory that contains `bins/` and `sparse_bins/`.** Every path is hardcoded and
cwd-relative; none of the four parses `argv`.

| Binary | Reads |
|---|---|
| `bench_dense_amx`, `bench_dense_onednn` | `bins/{config,q,k_cache,block_table,seqlens,out_golden}.bin` |
| `bench_sparse_amx`, `bench_sparse_onednn` | `sparse_bins/{config,q,k_cache,indices,out_golden}.bin` |

`config.bin` is 20 bytes, `struct.pack("iiiif", ...)`:

| | field 1-3 | field 4 | field 5 |
|---|---|---|---|
| dense | `B`, `H`, `D` | `BlockSize` = 64 | `softmax_scale` (float) |
| sparse | `B`, `H`, `D` | `NumIndices` = 2048 | `softmax_scale` (float, 1.0 if fused) |

### Outputs

**stdout only — nothing is written to disk.** Despite the name, `out_golden.bin` is an *input*: the
GPU reference each run is checked against. Each binary prints a per-phase timing breakdown, an
end-to-end figure, and a correctness line:

```
🔎 Max Diff: <value> (PASS)      # PASS when < 0.1
```

Timed loop: 50 iterations in `bench_sparse_amx`, 10 in both OneDNN benchmarks.

### Environment knobs

`bench_sparse_amx.cpp` only — the other three read no environment variables.

| Variable | Effect | Default |
|---|---|---|
| `BENCH_B` | Batch size; Q and indices are replicated B times (same input per batch) | `1` |
| `FORCE_EFFECTIVE_THREADS` | Overrides the `pick_attention_config` thread count | dispatcher's choice |
| `FORCE_NUM_CHUNKS` | Overrides `NUM_CHUNKS` | dispatcher's choice |

These two reproduce the section-4 sweep cells, e.g. cell F (B=3, 96 cores):

```bash
BENCH_B=3 FORCE_EFFECTIVE_THREADS=96 ./bench_sparse_amx
```

> **Untracked but present:** the four `.pt` files (~259 MB). **Untracked and generated:** `bins/`,
> `sparse_bins/` (~203 MB) and the compiled binaries. All are covered by [.gitignore](.gitignore).

## Hardware and workload

Intel Xeon Platinum 8468 (Sapphire Rapids): 2 sockets x 48 cores = 96, **no SMT**, 2 NUMA nodes
(502.9 / 504.0 GiB), `amx_bf16` + `amx_tile` + `avx512_bf16` + `avx512_vnni`.

- **Sparse** input: 2048 active tokens selected from a **147,200-token** KV pool (DeepSeek Sparse
  Attention, `index_topk=2048`) — token-indexed gather.
- **Dense** input: block-table KV cache, 8192 tokens, 64-token blocks.
- Goldens captured from a GPU FlashMLA run and carried here as `{dense,sparse}_output.pt`.
  Correctness gate: `Max Diff = 0.00708 < 0.1`.

## 1. Custom AMX vs OneDNN — the headline

| Path | Cores | Custom AMX | OneDNN | Winner | Margin |
|---|---:|---:|---:|---|---|
| Dense | 96 | ~8.5 ms | **6.13 ms** | **OneDNN** | 28% |
| Sparse | 32 | **0.554 ms** | ~0.83 ms | **custom AMX** | 33% |

OneDNN wins dense because the custom path's K/V packing (~8 ms Map phase) dominates, and
`brg_matmul:amx` avoids manual VNNI staging. Custom AMX wins sparse because OneDNN cannot fold the
Dequant+RoPE step into the GEMM as efficiently as a hand-written kernel.

## 2. Sparse AMX optimisation ladder (32 cores, B=1, static-input path)

| # | Variant | Median ms | Speedup | Delta |
|--:|---|---:|---:|---|
| 0 | baseline | 0.4559 | 1.00x | — |
| 3 | cached K/V packs | 0.2109 | 2.16x | -54% |
| 4 | + cached gather+dequant | 0.1803 | 2.53x | -15% |
| 7 | + static allocs (kill 1 MB/iter) | 0.1165 | 3.91x | -35% |
| 8 | + vectorized softmax max | 0.0957 | 4.76x | -18% |
| **9** | **+ NUM_CHUNKS=8** | **0.0944** | **4.83x** | -1.4% |

**Caveat:** v3/v4/v9 cache packs across calls, which is only valid for a static-input benchmark.
**Production top-k indices change every call**, so these wins do not transfer. The production-path
number is the Step 5e result below.

## 3. L1 working-set fix (Step 5e — applies to the production no-caching path)

L1d is **48 KB/core**; the original per-thread working set was **K 73 KB + Q 18 KB = 91 KB, 1.9x
over L1**, so every `tile_loadd` was an L2 hit. Fixed by sub-chunking Q*K^T into 4 sub-chunks of 16
tokens (36 KB per sub-chunk, fits L1d) plus a 4-way unroll on the P*V `d` dimension.

| Component | Before | After 5e |
|---|---:|---:|
| AMX(Q*K^T) | ~0.04 ms | 0.02 ms (2x) |
| AMX(P*V) | ~0.04 ms | 0.02 ms (2x) |
| **End-to-end** | ~0.55 ms | **0.46 ms (1.20x)** |

Q*K^T fell short of the 3-4x predicted by load-cycle math: the HW L2 prefetcher was already covering
much of the L2-hit cost. Software `_mm_prefetch` (5c v3) gave **0%** for the same reason.

**This avenue is now closed.** The matmuls are ~4.3% of E2E (0.020 of 0.460 ms), so even an infinite
matmul speedup buys at most 1.045x.

## 4. Core / batch scaling (Step 5h, 6-cell sweep, all PASS)

| Test | B | cores | E2E (ms) | ms/batch | Q*K^T (ms) | Q*K^T TFLOPs | % of peak | pack+V-pack (ms) | pack % E2E |
|---|--:|--:|---:|---:|---:|---:|---:|---:|---:|
| **A** | 1 | 32 | 0.460 | 0.460 | 0.020 | **15.10** | 18.4% | 0.205 | 44.6% |
| B | 3 | 32 | 1.304 | 0.435 | 0.150 | 6.04 | 7.4% | 0.625 | 47.9% |
| C | 1 | 48 | 0.484 | 0.484 | 0.050 | 6.04 | 4.9% | 0.210 | 43.4% |
| D | 3 | 48 | 1.042 | **0.347** | 0.090 | 10.07 | 8.2% | 0.420 | 40.3% |
| E | 6 | 48 | 1.877 | 0.313 | 0.185 | 9.79 | 8.0% | 0.865 | 46.1% |
| **F** | 3 | 96 | 0.876 | **0.292** | 0.060 | **15.10** | 6.1% | 0.280 | 32.0% |

Per-call FLOPs: `N x 570 MFLOPs` (302 Q*K^T + 268 P*V), verified from kernel source.
AMX ceiling at 2.5 GHz: 32c ~82 TFLOPs, 48c ~123, 96c ~246. A bare back-to-back `tdpbf16ps` micro
reaches **77.8 TFLOPs at 32c = 95% of theoretical** (6.74 ns/instr = 16.85 cycles, matching Intel's
spec), so the gap is entirely in the surrounding kernel, not the AMX unit.

### The 15 TFLOPs ceiling

Tests A (B=1, 32c) and F (B=3, 96c) hit **exactly 15.10 TFLOPs** aggregate Q*K^T. Tripling cores
*and* tripling work delivers the same absolute rate. Per-core efficiency collapses: 472 GF/core at
B=1/32c (18.4% of the 2.56 TF/core peak) down to 157 GF/core at B=3/96c (6.1%).

**Why:** pack is 32-48% of E2E in every cell and is memory-bandwidth bound (~73 KB of K/V per
block); `TDPBF16PS` issues 1-per-16-cycles and the port stays under-fed because pack stalls leave
the FU idle; per-core L1d is fully consumed by the L1-fit Q-tile + K-pack.

**To break it:** stream AMX directly from the KV cache without VNNI staging (would remove ~40% of
E2E), or fuse FP8 dequant into K-pack so the only pass over KV is the AMX load.

## 5. Core-count guidance

| Operating point | 32c | 48c | speedup |
|---|---:|---:|---:|
| B=1 | **0.460 ms** (A) | 0.484 ms (C) | 0.95x — **regression** |
| B=3 | 1.304 ms (B) | **1.042 ms** (D) | 1.25x |

**96 cores is 64% SLOWER than 32 at B=1** regardless of `NUM_CHUNKS` (0.1553 / 0.1535 / 0.1548 for
8 / 12 / 24). Causes: cross-NUMA K/V traffic (96 cores span both sockets, so half the threads cross
UPI), barrier cost growing with team size, AMX-active frequency throttling, and per-thread work
shrinking below the point where overheads amortise.

**At B>=3, 96 cores wins** (F: 0.292 ms/batch) because per-call overhead is roughly fixed and
amortises as B grows.

`pick_attention_config(max_threads, B, H_GROUPS, num_blocks)` in `bench_sparse_amx.cpp` self-tunes
`(effective_threads, NUM_CHUNKS)` at call time. Its current rule caps single-socket threads at 32,
leaving **~25% on the table at B>=2** where 48 would win.

**Independent corroboration:** the clean lane's end-to-end TP2 thread sweep found the same 32-thread
optimum, with 96 threads 4.3x worse on decode. Two unrelated methodologies, same conclusion.

## 6. GPU comparison — not available

**No GPU latency numbers exist.** The `.pt` captures carried here hold correctness tensors only —
reference outputs, not timings. Nothing in the optimisation log records a GPU ms figure.

Producing a real CPU-vs-GPU comparison requires running the upstream GPU benchmarks on an H100 at
these shapes. Until then, any GPU number quoted here would be fabricated. The GPU kernels to compare
against are `sparse_decode_fwd`, `sparse_prefill_fwd`, `dense_decode_fwd` and `dense_prefill_fwd`
(SM90 for all but dense prefill, which is SM100).

Note the CPU port has **no sparse prefill kernel**; the GPU does. See
`../PARALLELISM_PLAN_FINAL.md` item 8.

## 7. Open items

- **Sparse prefill kernel** — restructure the outer loop from head-groups to query positions,
  mirroring the GPU's `blockIdx.x -> (s_q_idx, head_group)` decomposition. Item 8 of the plan.
- **Break the pack ceiling** — see section 4.
- **Raise the dispatcher's single-socket cap** from 32 to 48 for B>=2.
- **Sparsity is worthless below 2048 tokens.** `index_topk=2048`, so at `s <= 2048` every key is
  selected and sparse == dense. Every benchmark here uses a 147k pool, which is the regime where it
  pays; do not extrapolate to short context.
