"""Sorted/batched expert dispatch for MoE.

THE PROBLEM (inherited from upstream, not introduced by the B>1 work --
model.py:49-54 has the identical loop; ep_moe.py only changed which expert
range a rank owns):

    counts = torch.bincount(indices.flatten(), ...)
    for i in range(start, end):
        idx, top = torch.where(indices == i)       # <-- full scan, PER EXPERT
        y[idx] += self.experts[i](x[idx]) * weights[idx, top, None]

Per prefill chunk (2048 tokens, top-8 of 256 experts, EP2 -> 128 local, 58 MoE
layers) that is:
    * 7,424 expert calls
    * 7,424 full scans of the [2048, 8] index tensor -- 128 rescans per layer
    * 7,424 fancy-index gathers x[idx] with scattered idx
    * every GEMM shaped [~64 x 7168] @ [7168 x 2048] -- M=64, the thin-M case
      that caps AMX hard

WHAT THIS CHANGES

  1. ONE argsort per layer instead of 128 scans. Sorting the flattened
     (token, slot) -> expert assignment once gives every expert a CONTIGUOUS
     run, found by searchsorted rather than by rescanning.
  2. Contiguous slices instead of scattered gathers. x[idx] with sorted idx is
     a sequential read; the previous form was a random gather per expert.
  3. index_add_ instead of y[idx] += ..., which avoids the read-modify-write
     hazard of duplicate indices and is a single fused scatter-add.

WHAT THIS DOES *NOT* CHANGE: the GEMM shape. Each expert still sees ~64 tokens.
Making M large needs the weights stacked into [n_local, dim, inter] for a true
batched GEMM, which duplicates weight storage (600 GB/rank resident) and is a
separate, riskier change. This module is the cheap, safe half.

STATUS: the claim that MoE dominates prefill is inferred from chunk 1 of job
27926568 taking 149 s at end_pos=2048, where attention and the indexer have
almost no work. Job 27929837 (PFPROF) measures it directly. The changes here
are correctness-preserving regardless of how that lands.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional


ENABLED = False     # set by install_batched_moe(); ep_moe falls back when False

# Sorted dispatch wins on prefill and LOSES on decode -- measured, 128 local
# experts, job 27930326:
#     decode B=1     0.95x   decode B=32   0.87x     <- slower
#     chunk 2048     1.68x   chunk 8192    1.86x     <- faster
#     chunk 32768    1.26x
# With 8192 decode steps a 0.9x decode penalty costs far more than the prefill
# win, so switch on token count rather than running it everywhere.
MIN_TOKENS = int(__import__("os").environ.get("FLASHMLA_MOE_MIN_TOKENS", "512"))
# Off by default: a barrier per MoE layer is a real cost, wanted only when
# attributing the all_reduce. See the comment at the barrier site.
_PROBE = __import__("os").environ.get("MOE_BARRIER_PROBE", "0") == "1"


def run_experts_sorted(moe, x, weights, indices, ep_start: int, ep_end: int):
    """Local experts over x, one sort instead of a scan per expert.

    Drop-in for the loop in ep_moe._ep_moe_forward / _true_dp_moe_forward.
    Returns y (float32, shaped like x) WITHOUT the shared expert or any
    collective -- the caller owns those, since tp2 / dp2_epon / dp2_true differ
    in exactly that part.
    """
    import torch

    topk = indices.shape[1]
    y = torch.zeros_like(x, dtype=torch.float32)
    self = moe

    if x.shape[0] < MIN_TOKENS:          # decode-sized: the plain loop is faster
        counts = torch.bincount(indices.flatten(),
                                minlength=self.n_routed_experts).tolist()
        for i in range(ep_start, ep_end):
            if counts[i] == 0:
                continue
            idx, top = torch.where(indices == i)
            y[idx] += self.experts[i](x[idx]) * weights[idx, top, None]
        return y

    # one pass: flatten (token, slot) -> expert, then sort by expert so each
    # expert owns a contiguous run. Replaces `torch.where(indices == i)` for
    # every i, which rescanned the whole tensor 128 times per layer.
    flat_e = indices.reshape(-1)
    order = torch.argsort(flat_e)                    # stable enough; ties don't matter
    sorted_e = flat_e[order]
    tok_of = torch.div(order, topk, rounding_mode="floor")
    slot_of = order % topk

    # Run boundaries for our experts, by binary search instead of scanning.
    # .tolist() ONCE: indexing bounds[i] per expert would be 2 tensor->scalar
    # syncs x 128 experts x 58 layers = ~14,800 syncs per forward, which at
    # decode (B=1, only 8 assignments total) would cost more than the scan it
    # replaces. The loop this supersedes had the same reason for counts.tolist().
    bounds = torch.searchsorted(
        sorted_e, torch.arange(ep_start, ep_end + 1, device=sorted_e.device)).tolist()

    for i in range(ep_start, ep_end):
        lo = bounds[i - ep_start]
        hi = bounds[i - ep_start + 1]
        if hi <= lo:
            continue
        tok = tok_of[lo:hi]                          # contiguous run
        slot = slot_of[lo:hi]
        out = self.experts[i](x[tok]) * weights[tok, slot, None]
        y.index_add_(0, tok, out.to(torch.float32))  # fused scatter-add
    return y


def _make_upstream_forward(model_module):
    """Upstream MoE.forward (model.py:786-798) with sorted dispatch.

    Needed because install_ep_moe only runs for dp2_epon / dp2_true. Under tp2
    the MoE keeps upstream's own forward, which has the identical per-expert
    scan over its TP range (experts_start_idx..experts_end_idx) -- so without
    this, USE_BATCHED_MOE would be a no-op on exactly the prefill runs we want
    to measure.
    """
    import torch
    import torch.distributed as dist

    def forward(self, x):
        # Timed into chunked_prefill.STAGE_MS so the decode-only breakdown in
        # ctx_tile covers MoE too. Measured at BS=1/ctx=4096 (job 27933571),
        # the five MLA stages summed to 776 ms of a 1980 ms step -- the other
        # ~1204 ms (61%) was MoE/MLP and completely uninstrumented, i.e. the
        # largest cost in a decode step was the one nobody could see.
        import time as _time
        try:
            from chunked_prefill import STAGE_MS as _SM
        except Exception:
            _SM = None
        _t0 = _time.perf_counter()
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        _tg = _time.perf_counter()
        y = run_experts_sorted(self, x, weights, indices,
                               self.experts_start_idx, self.experts_end_idx)
        if _SM is not None:
            _SM["moe_gate"] = _SM.get("moe_gate", 0.0) + (_tg - _t0) * 1e3
            _SM["moe_experts"] = _SM.get("moe_experts", 0.0) + (_time.perf_counter() - _tg) * 1e3
        _t1 = _time.perf_counter()
        # ORDER IS LOAD-BEARING: upstream (model.py:801-803) adds the shared
        # expert BEFORE the all_reduce, so under TP2 the replicated shared
        # contribution is summed across ranks and counted TWICE. Adding it
        # after the reduce counts it once -- different maths, different logits.
        # An earlier version of this file did exactly that and produced token
        # 2496 where the reference path gives 10575 (jobs 27930256/27930397).
        y += self.shared_experts(x)
        _t2 = _time.perf_counter()
        if dist.is_initialized() and dist.get_world_size() > 1:
            # MOE_BARRIER_PROBE splits "waiting for the other rank" from "actual
            # collective". Measured at BS=1/ctx=4096 (job 27933704), this
            # all_reduce cost 484.64 ms/step -- 42x the 11.6 ms that 58 layers
            # of a standalone [1,7168] bf16 all_reduce (0.20 ms each) predict.
            # Hypothesis: TP2 splits 256 experts 128/128, a token routes to 8
            # that do not divide evenly, and the rank drawing fewer idles in
            # all_reduce until the other finishes -- so the collective timer is
            # billing imbalance. A barrier immediately before absorbs that wait:
            # if moe_imbalance is large and moe_allreduce collapses toward
            # ~0.2 ms/layer, the cost is idle time, not network.
            if _PROBE:
                dist.barrier()
                if _SM is not None:
                    _SM["moe_imbalance"] = _SM.get("moe_imbalance", 0.0) + (_time.perf_counter() - _t2) * 1e3
                # _t3, NOT _t2: reassigning _t2 made moe_shared (_t2 - _t1)
                # swallow the barrier as well, reporting 437.80 ms against a
                # true 76.70 (job 27934004 vs 27933704).
                _t3 = _time.perf_counter()
            else:
                _t3 = _t2
            dist.all_reduce(y)
        if _SM is not None:
            _SM["moe_shared"] = _SM.get("moe_shared", 0.0) + (_t2 - _t1) * 1e3
            _SM["moe_allreduce"] = _SM.get("moe_allreduce", 0.0) + (_time.perf_counter() - _t3) * 1e3
        return y.type_as(x).view(shape)

    return forward


def install_batched_moe(transformer: Any,
                        model_module: Any = None,
                        log_fn: Callable[[str], None] = print) -> Dict[str, Any]:
    """Enable sorted expert dispatch.

    Two cases, because the MoE forward in play depends on sharding mode:
      * dp2_epon / dp2_true -- ep_moe already rebound forward; just flip the
        flag it consults, leaving its EP range and collective untouched.
      * tp2 -- no EP install happened, so rebind upstream MoE.forward over its
        own experts_start_idx..experts_end_idx range.
    """
    global ENABLED
    ENABLED = True
    ep = sum(1 for l in transformer.layers
             if hasattr(getattr(l, "ffn", None), "ep_start_idx"))
    rebound = 0
    if ep == 0 and model_module is not None:
        model_module.MoE.forward = _make_upstream_forward(model_module)
        rebound = sum(1 for l in transformer.layers
                      if hasattr(getattr(l, "ffn", None), "experts_start_idx"))
    log_fn(f"[moe-batched] sorted dispatch ENABLED -- ep_layers={ep} "
           f"upstream_rebound={rebound} (1 argsort/layer replaces ~128 index scans)")
    return {"installed": True, "ep_layers": ep, "upstream_rebound": rebound}
