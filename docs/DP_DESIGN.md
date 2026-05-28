# DP2 / DP2_EPON — Design Notes (structured_cpu_run_dp)

This lane (`structured_cpu_run_dp`) implements data-parallel CPU inference for
DeepSeek-V3.2 on top of the TP2-proven clean runner. `structured_cpu_run_clean`
is the separate profiling lane and is not touched here.

Modes:

- **DP2** — full model replicated on each of 2 ranks. Two independent
  single-rank processes; no collectives. Used as a regression baseline.
- **DP2_EPON** — dense layers replicated, routed experts partitioned 50/50,
  one `all_reduce` per MoE layer. Production-relevant mode.

This note currently covers the first open design question: the **Indexer
cross-rank broadcast**. It will be extended as the other stages land.

## Indexer.forward — what actually happens today (TP2)

Source: `../DeepSeek-V3.2/inference/model.py`, `class Indexer` (lines 435–487).

Facts established by inspection:

1. The Indexer's projections (`wq_b`, `wk`, `weights_proj`) are plain `Linear`
   — **replicated full-size on every rank**, never Column/Row-parallel.
2. `self.n_local_heads = args.index_n_heads // world_size` is computed in
   `__init__` but **never referenced in `forward`**. The forward computes the
   full `self.n_heads` heads. So the Indexer is **not head-partitioned at
   runtime** under any mode — every rank computes the complete `topk_indices`
   independently and identically.
3. The tail of `forward`:
   ```python
   topk_indices  = index_score.topk(...)[1]
   topk_indices_ = topk_indices.clone()
   dist.broadcast(topk_indices_, src=0)
   assert torch.all(topk_indices == topk_indices_)
   return topk_indices      # <- returns the LOCAL value, not the broadcast
   ```
   The broadcast pulls rank-0's indices into `topk_indices_`; the assert checks
   the local indices equal rank-0's; the broadcast result is then **discarded**.

So the broadcast + assert is a **cross-rank determinism sanity check on a
replicated computation**. It is *not* an all_gather of partitioned heads and
*not* an all_reduce. It exchanges no data that the model consumes.

It does, however, call `dist.broadcast`, which **requires
`dist.is_initialized()`** — it crashes if the process group is absent.

## Answers to the design questions

1. **Current TP2 Indexer behavior:** broadcasts rank-0's `topk_indices` to a
   scratch tensor and asserts every rank matches. Returns the locally-computed
   `topk_indices`. No all_gather, no all_reduce. It is a determinism check.

2. **Does TP2 *need* Indexer communication?** No, not for correctness. The
   Indexer is replicated (full heads on each rank), so each rank already has
   the correct `topk_indices`. The broadcast/assert only verifies the ranks
   agree; it is a safety net (cheap relative to TP's real collectives), not a
   data dependency. TP2 is the known-good path, so we leave it ON there.

3. **Should DP2 skip Indexer communication?** **Yes — mandatory.** DP2 runs as
   two independent processes with `dist` *not* initialized, so `dist.broadcast`
   would crash. The Indexer is fully replicated, so the local `topk_indices`
   is already correct; skipping the check is safe. Each DP2 rank produces a
   complete, self-consistent output (rank 0's result is the one we keep).

4. **Should DP2_EPON skip Indexer communication?** **Yes — recommended.**
   `dist` *is* initialized in DP2_EPON (for the MoE `all_reduce`), so the
   broadcast would not crash, but the dense/attention path including the
   Indexer is replicated and deterministic, so the per-layer broadcast is pure
   overhead (61 broadcasts/forward). The legacy lane ran DP2_EPON with the
   broadcast skipped and stayed token-exact, which empirically confirms the
   replicated Indexer is deterministic across ranks on this hardware.

5. **Predicate that should control this:** a dedicated `ShardingPlan` flag,
   not `heads_partitioned` and not a raw `sharding_mode` string in the override.
   - `heads_partitioned` is **semantically wrong**: the Indexer is not
     head-partitioned in forward under any mode (fact #2). It happens to give
     the right tp2-vs-dp answer, but it's a footgun — reject it.
   - A raw `sharding_mode` string check inside the override couples the
     override to mode names; the plan is meant to be that abstraction.
   - `dist.is_initialized()` alone is correct for *crash-safety* (skip when no
     process group) but leaves the broadcast ON in DP2_EPON where it's just
     overhead.
   Proposed: `ShardingPlan.indexer_cross_rank_check: bool`, decided per mode,
   AND-ed defensively with `dist.is_initialized()` at the call site so a
   misconfiguration can never crash.

   | mode | indexer_cross_rank_check | effective (AND dist.is_initialized) |
   |---|---|---|
   | tp2 | True | True (keep — known-good) |
   | dp2 | False | False (skip — dist not initialized anyway) |
   | dp2_epon | False | False (skip — perf; replicated + deterministic) |

6. **Where the behavior should eventually live:** the override
   `src/overrides/model.py` (shadows upstream `model`) owns the patched
   `Indexer.forward`. It reads the decision from `src/overrides/sharding_plan.py`
   (`plan.indexer_cross_rank_check`) set once per process via `set_plan(...)`.
   `scripts/native_run.py` only constructs the plan from `SHARDING_MODE` + the
   rank/world env and calls `set_plan(...)` before importing/constructing the
   model. native_run does not contain the predicate itself.

7. **Risks of removing the Indexer broadcast in DP modes:**
   - The broadcast/assert is the only cross-rank determinism guard. If the
     replicated Indexer were ever non-deterministic across ranks (FP8 rounding,
     differing BLAS reduction order per rank), removing it would hide the
     divergence.
     - **DP2:** low risk — ranks are independent; only rank 0's output is kept,
       and it is internally self-consistent regardless of rank 1.
     - **DP2_EPON:** higher (but still low) risk — the MoE `all_reduce` couples
       ranks, so a topk divergence upstream could silently corrupt the summed
       expert output. Mitigated by: (a) the legacy lane's token-exact DP2_EPON
       results with the broadcast skipped, and (b) keeping the guard
       re-enable-able via the plan flag for a one-time confirmation run.
   - Mitigation to add at implementation time: at first forward, assert
     `dist.is_initialized() == plan.needs_dist` so a mode/plan mismatch fails
     loudly instead of silently skipping a needed collective.

## Status

Design/inspection only. No runtime code, checkpoint-prep, or configs added in
this step. The `ShardingPlan` dataclass and `src/overrides/model.py` override
are the next stages.
