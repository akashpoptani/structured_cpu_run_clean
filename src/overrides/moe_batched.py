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

    # one pass: flatten (token, slot) -> expert, then sort by expert so each
    # expert owns a contiguous run. Replaces `torch.where(indices == i)` for
    # every i, which rescanned the whole tensor 128 times per layer.
    flat_e = indices.reshape(-1)
    order = torch.argsort(flat_e)                    # stable enough; ties don't matter
    sorted_e = flat_e[order]
    tok_of = torch.div(order, topk, rounding_mode="floor")
    slot_of = order % topk

    # run boundaries for our experts, by binary search instead of scanning
    bounds = torch.searchsorted(
        sorted_e, torch.arange(ep_start, ep_end + 1, device=sorted_e.device))

    for i in range(ep_start, ep_end):
        lo = int(bounds[i - ep_start])
        hi = int(bounds[i - ep_start + 1])
        if hi <= lo:
            continue
        tok = tok_of[lo:hi]                          # contiguous run
        slot = slot_of[lo:hi]
        out = self.experts[i](x[tok]) * weights[tok, slot, None]
        y.index_add_(0, tok, out.to(torch.float32))  # fused scatter-add
    return y


def install_batched_moe(transformer: Any,
                        log_fn: Callable[[str], None] = print) -> Dict[str, Any]:
    """Enable sorted dispatch inside the already-installed EP/DP MoE forwards.

    Must run AFTER install_ep_moe / install_true_dp_moe: it does not rebind any
    forward, it only flips the flag those forwards consult, so the EP range and
    the collective they chose stay exactly as they were.
    """
    global ENABLED
    ENABLED = True
    n = sum(1 for l in transformer.layers
            if hasattr(getattr(l, "ffn", None), "ep_start_idx"))
    log_fn(f"[moe-batched] sorted expert dispatch ENABLED on {n} MoE layers "
           f"(1 argsort/layer replaces ~128 full index scans)")
    return {"installed": True, "layers": n}
