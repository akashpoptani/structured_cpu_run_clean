# PARALLELISM_PLAN_FINAL.md

Ordered work plan. Items 1-4 are cleanups that unblock everything else; 5-9 are the build;
10-11 turn measurements into a dispatcher; 12 is parked.

---

## 1. Remove dead config fields

`INFERENCE_ARCHITECTURE`, `STREAMING`, `FAST_LINEAR`, `BATCHED_MOE`, `AMX_ENABLED`.
All validate in `parse_config.sh` and propagate to the resolved env; **none is read by any Python**.
Delete from `_baseline.env`, the parser's required list, and the emit list. Revisit when real.

## 2. Change the thread-placement defaults

`scripts/run_native_distributed.sh:98-99` forces `OMP_PROC_BIND=close` / `OMP_PLACES=cores`.
Measured **5.5x slower on decode at 8 threads, 9.4x at 32**. Set `OMP_PROC_BIND=false` and default
`OMP_NUM_THREADS=32`. Every config currently has to override this by hand.

## 3. Remove the indexer broadcast (`SKIP_INDEXER_BROADCAST`)

`Indexer.forward:485` — unguarded, fires 61x per forward. It is a **debug assertion, not a
computation**: every rank already computes the full `index_score` independently (the indexer is
replicated — `wq_b`/`wk` carry dim `None` in `convert.py`'s mapping), and the broadcast only
verifies the ranks agree. Legacy tagged removal token-exact safe.
**Mandatory before item 7**: under sharded batches ranks legitimately hold different tokens, so the
assert would fail.

## 4. Wire or assert `EP_SIZE`

The EP path is gated solely on `SHARDING_MODE == "dp2_epon"`. `EP_SIZE` is read by nothing, so
`EP_SIZE=1` would silently still split experts. Either drive the split from it, or assert
`EP_SIZE == world_size` in `install_ep_moe`.

## 5. Prompt corpus (replaces seed repetition)

`build_exact_prompt` repeats one pangram to reach Lin. That produces **degenerate expert routing** —
the same few experts fire for every token — so any EP load-balance or throughput measurement is
unrealistic. Needed even at batch=1.

Requirements: offline (no downloads), deterministic (seeded), exact-token-length, diverse across
batch slots. Candidate source: text already on disk (repo docs, the DeepSeek papers under
`../DeepSeek-V3.2-Exp/`), chunked into a corpus file, sampled by seeded index. Fall back to the
existing `SEEDS` cycle if no corpus is available.

Record per-expert token counts per layer, to quantify routing skew.

## 6. batch > 1 (prerequisite, blocks everything)

| File | Change |
|---|---|
| `src/clean_inference/prompting.py` | drop `NotImplementedError` for `batch_size != 1`; use the item-5 corpus |
| `src/clean_inference/generation.py` | `torch.full((1, T))` -> `(B, T)`; `next_token.item()` -> `.tolist()`; per-row termination |
| `scripts/native_run.py` | per-sequence result reporting |

Do this **on TP2 first** — smallest change, and it yields the batch-scaling curve we lack.

## 7. True DP (sequence-sharded attention)

Rank *r* owns `batch[r::world_size]`, computes attention only on its own sequences, holds only their
KV.

Requires: per-rank batch assignment; `max_batch_size = ceil(B/world_size)` in
`build_modelargs_for_case` (**this is what actually sheds the KV** — without it the buffers are
still allocated full size); item 3; `B_local > 0` guards for B=1, which means overriding
`Block.forward`/`Transformer.forward`.

### MoE dispatch/combine

Today `_ep_moe_forward` assumes `x` is identical on every rank — true only because both compute the
same attention. With sharded batches it is not:

```python
x_all   = all_gather(x_local)        # (a) dense gather; gate is replicated so routing is consistent
w, idx  = gate(x_all)                # never communicated -- derived identically on every rank
y_all   = local_experts(x_all)       # experts [r*128, (r+1)*128)
all_reduce(y_all)                    # or reduce_scatter straight to the local slice
y_local = y_all[my_slice] + shared_experts(x_local)
```

| Design | Moves | Verdict |
|---|---|---|
| (a) `all_gather` x | every token to every rank | **start here** — at world_size=2 with top-8-of-256, ~half of each token's experts are remote anyway, so (b) saves little |
| (b) `all_to_all` dispatch | only tokens needing remote experts | wins at larger world_size; needs variable-size all-to-all + routing metadata |

`shared_experts(x_local)` depends only on local data, so it can overlap an `async_op=True` gather.
`gate(x_all)` cannot.

**Unverified:** gloo's `reduce_scatter` / `all_to_all` at world_size >= 2. A world_size=1 test passes
but is meaningless (collectives no-op there). `all_gather` and `all_reduce` are certain. **Test
before designing around them.**

Payoff: **2x batch capacity** at long context (KV is 9.56 GiB/sequence at 128k; ~360 GiB headroom
gives 37 sequences replicated vs 75 sharded), and prefill attention work divided by world_size.

## 7b. TODO — vendor a clean `FlashMLA_CPU/` into this repo

Before item 8 (which prototypes inside it), bring the kernel work in-tree as
`structured_cpu_run_clean/FlashMLA_CPU/`, containing only the four benchmarked kernels:

| File | Path | Library |
|---|---|---|
| `bench_sparse_amx.cpp` | sparse | custom AMX intrinsics |
| `bench_sparse_onednn.cpp` | sparse | OneDNN |
| `bench_dense_amx.cpp` | dense | custom AMX intrinsics |
| `bench_dense_onednn.cpp` | dense | OneDNN |

Plus a `README.md` carrying every measured number (see that file). Drop the 15 exploratory
`bench_sparse_amx_v*.cpp` variants — the ladder they represent is recorded in the README.

### L1 working-set question — confirmed, and already fixed

The hypothesis is right: Sapphire Rapids L1d is **48 KB/core**, and the original per-thread working
set was **K 73 KB + Q 18 KB = 91 KB — 1.9x over L1**, so every `tile_loadd` was an L2 hit.

**Step 5e already fixed it** by sub-chunking Q*K^T into 4 sub-chunks of 16 tokens
(per-sub-chunk K 18 KB + Q 18 KB = 36 KB, fits L1d) plus a 4-way unroll on P*V:

| Component | Before | After 5e |
|---|---|---|
| AMX(Q*K^T) | ~0.04 ms | 0.02 ms (2x) |
| AMX(P*V) | ~0.04 ms | 0.02 ms (2x) |
| End-to-end | ~0.55 ms | **0.46 ms (1.20x)** |

**Do not spend more effort here.** The matmuls are now ~4.3% of E2E (0.020 ms of 0.460 ms), so even
an infinite matmul speedup buys at most 1.045x. Multi-level tiling is already done; software
prefetch was tested in 5c v3 and gave **0%** (the HW prefetcher already covers the L2 hits); and
`tile_loadd` cannot be told to target a specific cache level.

**The real bottleneck is the pack phase — 32-48% of E2E in every measured cell**, and it is what
caps aggregate Q*K^T at 15.10 TFLOPs regardless of core count. Attack that instead:
stream AMX directly from the KV cache without VNNI staging, or fuse FP8 dequant into K-pack so the
only pass over KV is the AMX load. Pack caching (v3/v4/v9, 4.83x) does **not** apply — production
top-k indices change every call.

## 8. Chunked prefill + indexer fusion — but read this first

Current hard ceiling: unchunked prefill OOMs at ~22-23k. Measured: **Lin=20480 passes** at 891.8 GiB
of 950 (prefill 4697 s); **Lin=26624 fails** allocating 169 GiB for the FP32 indexer einsum.

### Can FlashMLA_CPU do sparse *prefill* directly? Probably yes — and the GPU does exactly that.

**The GPU has a dedicated sparse prefill kernel**: `csrc/sm90/prefill/sparse/fwd.cu`, exported as
`sparse_prefill_fwd` (alongside `sparse_decode_fwd`, `dense_prefill_fwd`, `dense_decode_fwd`). So
all four combinations exist upstream; our CPU port has sparse decode and dense, not sparse prefill.

**How the GPU solves the per-query top-k problem** (`phase1.cuh:47`):
```cuda
const int s_q_idx = blockIdx.x / (params.h_q/B_H);        // one CUDA block per (query pos, head grp)
const int topk_length   = params.topk_length[s_q_idx];    // each query has its OWN top-k list
const int num_topk_blocks = ceil_div(topk_length, B_TOPK);
```
It does **not** try to share gathered K across query positions. The M dimension of each tile is
**heads**, exactly as in decode; query positions are parallelised across blocks.

**Implication for us: the existing sparse decode kernel already has the right inner shape.** What
changes is the outer loop — parallelise over query positions with OpenMP instead of over head
groups. `pick_attention_config()` already picks thread count from total work units, so it
generalises. This is restructuring, not a new kernel.

**The catch:** zero K reuse across queries. Each query gathers its own 2048 keys (2048 x 576 x 2 B =
2.36 MB); at 2048 queries that is ~4.8 GB of gather traffic per layer. Pack was already **32-48% of
E2E** in the decode benchmarks and is the measured 15 TFLOPs ceiling — here it gets worse.
Mitigation to test: adjacent queries likely share many top-k keys, so take the union over a block of
queries and gather once.

**Sparsity buys nothing below 2048 tokens.** `index_topk=2048`, so for `s <= 2048` every key is
selected and sparse == dense. The win starts above 2048 and grows: at 120k it is 2048/120000 =
**1.7% of the work, ~59x**. All our benchmarks so far are at s=2048, i.e. exactly the point where
sparse attention is worthless — do not conclude anything about it from them.

### DECISION: prototype sparse prefill BEFORE writing chunked prefill

Do **not** start chunked prefill until the M-over-queries prototype has been run. If it works we get
chunking *and* sparsity from one kernel, and the legacy chunked-prefill port becomes unnecessary.

Prototype (standalone bench in `FlashMLA_CPU`, no integration, no clean-lane code):
1. Take `bench_sparse_amx_v9.cpp` and change the outer OpenMP loop from head-groups to **query
   positions**, keeping the inner AMX tiling (M = heads) exactly as is — this mirrors
   `phase1.cuh:47`'s `blockIdx.x -> (s_q_idx, head_group)` decomposition.
2. Shapes: `s_q` in {512, 2048, 8192}, topk 2048, 147k-token KV pool, 32 threads.
3. Measure ms and the **pack fraction** of E2E (already instrumented in the v-series).

| Outcome | Next step |
|---|---|
| pack fraction stays near the decode path's 32-48% | proceed — build the real sparse prefill kernel, skip chunked prefill |
| pack fraction dominates (gather traffic kills it) | try the union-over-query-block gather; if that fails, fall back to chunked prefill (legacy implementation exists, ran at 32k) |

Indexer fusion is worth doing either way — see below.

Indexer fusion is independent and worth doing either way: `fp8_index` materialises `(b, s, 64, s)`
FP32 then reduces heads; the GPU's `T.reduce_sum` fuses that reduction inside the tile loop and
never materialises it — **32x memory traffic**, and it is the tensor that actually OOMs.

## 9. Custom AMX sparse attention

Attach FlashMLA_CPU's sparse kernel (0.0944 ms, 4.83x optimised, already shaped for top-2048 of a
147k-token pool). **Decode first** — the kernel exists and needs only a PyTorch extension wrapper
(precedent: `../structured_cpu_run/without_vllm/overrides/amx_kernels/` + `fast_linear.py`, which
already monkey-patches `model.linear`). **Prefill second**, per item 8.

Keep the dense path as a correctness fallback and verify token-exact against `case_0001.json` at
every step.

## 10. Profiling support + crossover experiments

Build the profiler first (`PROFILER.md`): one file, config-driven, module hooks +
`LLM_INSTRUMENT`, inclusive *and* exclusive time, memory via `VmHWM`. Without it every core-count
and layout question is unanswerable.

Then map where each layout wins:

| Axis | Values |
|---|---|
| batch | 1, 2, 4, 8 |
| Lin/Lout | 128/20, 2048/2048, 8192/2048 |
| threads | 16, 32, 48 |
| layout | TP2, DP2_EPON |

Record TTFT, TPOT, MaxRSS, tokens/s/rank, and per-expert routing counts.

Open questions: does DP's prefill win grow with batch (expected: yes, linearly)? Does TP's decode
win survive batch > 1? Where does max batch cross over (DP should win at long context via KV)?

## 11. Adaptive dispatcher

`pick_layout(batch, lin, lout, cores) -> (TP|DP, EP, threads, chunk)`, modelled on FlashMLA_CPU's
existing `pick_attention_config()`. Claims must come from item 10's data, not from priors.

Note on why this is cheap: **TP2 is already TP+EP** — upstream `MoE.__init__` splits experts
`n_routed_experts // world_size` and `convert.py` filters them by rank, so both modes hold an
**identical** 128-expert shard. Only the non-expert 2.7% differs (sharded vs replicated), measured
as an 18 GB/rank premium for DP. Store experts once + non-expert unsharded, slice at load for TP.
One checkpoint, either layout, no expert reshuffling.

## 12. Parked ideas

- **Hot-expert replication.** Replicate the top-N% most-frequently-routed experts on both ranks to
  cut traffic, trading maximum batch size for lower communication. Composes with both layouts since
  EP is always on. Needs the routing histogram from item 5 first.
- **EP load balancing.** Routing skew means one rank finishes early and waits at the all_reduce.
  Measure skew before optimising.
- **Per-phase thread counts.** Prefill (compute-bound) and decode (bandwidth-bound) likely want
  different `OMP_NUM_THREADS`; `torch.set_num_threads()` is settable at runtime.
