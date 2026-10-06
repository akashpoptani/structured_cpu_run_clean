# PARALLELISM_PLAN.md — adaptive TP / DP / EP with custom AMX kernels

Goal: one framework that picks the parallelism layout from (batch size, Lin, Lout), runs sparse
attention on custom AMX kernels, and is optimised for **both** batched throughput and single-stream
latency. This file is the roadmap; `DP_IMPLEMENTATION_PLAN.md` covers DP2_EPON specifically.

## 1. Measured baseline (2026-09-23)

TP2 vs DP2_EPON, Lin=2048/Lout=2048, 32 threads, `OMP_PROC_BIND=false`, batch=1,
**same nodes lh[0900,0909], back-to-back** (jobs 27756417 / 27756418):

| | TTFT | TPOT | MaxRSS/rank |
|---|---|---|---|
| **DP2_EPON** | **227.8 s** | 1.229 s/tok | 750.0 GiB |
| **TP2** | 268.5 s | **1.153 s/tok** | 641.7 GiB |
| | **DP 15% faster** | **TP 6% faster** | TP 108 GiB lighter |

**Prefill favours DP, decode favours TP — even at batch=1.** That split is the central fact this
plan is built around, and it is why a single fixed layout is the wrong answer.

Explanation: prefill moves large payloads, so TP2's **185 collectives/forward** hurt; DP2_EPON has
**119** (58 after removing the indexer broadcast). Decode payloads are tiny and latency-bound, so
collective count matters less, and DP's **2x duplicated attention weights** (18 GB more streamed per
token) dominates instead.

## 2. Communication inventory (measured from upstream `model.py`)

| Site | Op | Guard | TP2 | DP2_EPON |
|---|---|---|---|---|
| `ParallelEmbedding:130` | all_reduce | `world_size>1` | 1 | 0 |
| `RowParallelLinear:266` | all_reduce | `reduce_output and world_size>1` | **64** | 0 |
| `Indexer:485` | **broadcast** | **NONE** | **61** | **61** |
| `MoE:803` | all_reduce | `world_size>1` | 58 | 58 (our override makes it unconditional) |
| `Transformer:911` | all_gather | `world_size>1` | 1 | 0 |
| **total** | | | **185** | **119** |

`reduce_scatter` and `all_to_all` are used **nowhere** today — they are proposals in section 5.

## 3. The shared-checkpoint insight (validated)

```
routed experts : 653.9B  (97.3%)
everything else:  18.0B  ( 2.7%)     <- attention, dense MLP, shared experts, embed, head

tp2       experts/2 + nonexp/2 = 671.9 GB   (measured 673.8)
dp2_epon  experts/2 + nonexp   = 689.8 GB   (measured 689.9)
DP premium = 18.0 GB = 2.7%
```

**Expert shards are byte-identical between TP2 and DP2_EPON** (both split 128/128 by rank). Only
the 2.7% non-expert portion differs. So: store experts once in EP layout + non-expert weights
**unsharded**, and choose TP or DP **at load time** by slicing (or not) that 18B. One checkpoint,
either mode, no expert reshuffling ever. The option premium is 18 GB/rank.

## 4. Phases

### Phase 1 — batch > 1 (prerequisite, blocks everything)
| File | Change |
|---|---|
| `src/clean_inference/prompting.py` | drop `NotImplementedError` for `batch_size != 1`; real prompt corpus (section 6) |
| `src/clean_inference/generation.py` | `torch.full((1, T))` -> `(B, T)`; `next_token.item()` -> `.tolist()`; per-row termination |
| `scripts/native_run.py` | per-sequence result reporting |
Do this **on TP2 first** — smallest change, and it yields the batch-scaling curve we lack.

### Phase 2 — true DP (sequence-sharded attention)
Rank *r* owns `batch[r::world_size]`, computes attention only on its own sequences, holds only their
KV. Requires: per-rank batch assignment; `max_batch_size = ceil(B/world_size)` in
`build_modelargs_for_case` (this is what actually sheds the KV); MoE gather/combine (section 5);
**removal of the indexer broadcast** (mandatory — see section 7); `B_local > 0` guards for B=1,
which means overriding `Block.forward`/`Transformer.forward`.

Payoff: **2x batch capacity** at long context (KV is 9.56 GiB/sequence at 128k; ~360 GiB headroom
gives 37 sequences replicated vs 75 sharded), and prefill attention work divided by world_size.

### Phase 3 — chunked prefill + indexer fusion
Independent of parallelism, and currently the hard ceiling: unchunked prefill OOMs at ~22-23k
(measured: Lin=20480 passes at 891.8 GiB of 950; Lin=26624 fails allocating 169 GiB for the FP32
indexer einsum). See README TODO.

### Phase 4 — custom AMX sparse attention
Attach `FlashMLA_CPU`'s sparse kernel (0.0944 ms, 4.83x optimised, already shaped for top-2048 of a
147k pool). Decode first (kernel exists), then prefill (needs M = query tokens tiling). Exploits the
DSA sparsity we currently compute and throw away — ~59x on prefill attention FLOPs at 120k.

### Phase 5 — adaptive dispatcher
`pick_layout(batch, lin, lout, cores) -> (TP|DP, EP, threads, chunk)`, modelled on FlashMLA_CPU's
existing `pick_attention_config()`. Needs the crossover map from section 8.

## 5. MoE dispatch/combine for true DP

Today `_ep_moe_forward` assumes `x` is identical on every rank (true only because both compute the
same attention). With sharded batches it is not:

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
but is meaningless (collectives no-op). `all_gather` and `all_reduce` are certain. **Test before
designing around them.**

## 6. Prompt corpus (replaces seed repetition)

`build_exact_prompt` repeats one pangram to reach Lin. That produces **degenerate expert routing** —
the same few experts fire for every token — so any EP load-balance or throughput measurement is
unrealistic. Needed even at batch=1.

Requirements: offline (no downloads), deterministic (seeded, reproducible), exact-token-length,
diverse across batch slots. Candidate source: text already on disk (repo docs, DeepSeek papers under
`../DeepSeek-V3.2-Exp/`), chunked into a corpus file, sampled by seeded index. Fall back to the
existing `SEEDS` cycle only if no corpus is available.

Output to record: per-expert token counts per layer, to quantify routing skew.

## 7. Cleanups (do with Phase 1)

- **Remove dead config fields**: `INFERENCE_ARCHITECTURE`, `STREAMING`, `FAST_LINEAR`, `BATCHED_MOE`,
  `AMX_ENABLED`. All validate and propagate; **none is read by any Python**. Revisit when real.
- **Change `OMP_PROC_BIND` default to `false`** in `scripts/run_native_distributed.sh:98`. Measured
  5.5x slower on decode at 8 threads, **9.4x at 32**. Also default `OMP_NUM_THREADS=32`.
- **Remove the indexer broadcast** (`SKIP_INDEXER_BROADCAST`). It is a debug assertion, not a
  computation: every rank already computes the full index_score independently (the indexer is
  replicated — `wq_b`/`wk` carry dim `None` in convert.py's mapping). The broadcast only verifies
  ranks agree. Legacy tagged it token-exact safe. Under sharded batches it would **fail**, since
  ranks legitimately hold different tokens.
- **Wire or assert `EP_SIZE`**: the EP path is gated solely on `SHARDING_MODE == "dp2_epon"`;
  `EP_SIZE` is read by nothing, so `EP_SIZE=1` would silently still split experts.

## 8. Crossover experiments

Map where each layout wins. Axes: batch {1, 2, 4, 8}, Lin/Lout {128/20, 2048/2048, 8192/2048},
threads {16, 32, 48}, layout {TP2, DP2_EPON}. Record TTFT, TPOT, MaxRSS, tokens/s/rank.
Open questions: does DP's prefill win grow with batch (expected: yes, linearly)? Does TP's decode
win survive batch > 1? Where does max batch cross over (DP should win at long context via KV)?

## 9. Parked ideas

- **Hot-expert replication.** Replicate the top-N% most-frequently-routed experts on both ranks to
  cut all-to-all traffic, trading maximum batch size for lower communication. Needs the routing
  histogram from section 6 first.
- **EP load balancing.** Routing skew means one rank finishes early and waits at the all_reduce.
  Measure skew before optimising.
- **Per-phase thread counts.** Prefill (compute-bound) and decode (bandwidth-bound) likely want
  different `OMP_NUM_THREADS`; `torch.set_num_threads()` is settable at runtime.
