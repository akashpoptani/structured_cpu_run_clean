"""Expert-parallel MoE forward for SHARDING_MODE=dp2_epon.

In DP the model is REPLICATED, so `dist.init_process_group` runs AFTER
`Transformer(args)` and the module-global `model.world_size` stays 1 for the
whole run. That breaks upstream `MoE.forward` in three separate ways, each of
which this module fixes. See DP_IMPLEMENTATION_PLAN.md section 4.

Upstream (model.py:790-800):

    for i in range(self.experts_start_idx, self.experts_end_idx):   # (A)
        if counts[i] == 0: continue
        y[idx] += self.experts[i](x[idx]) * weights[idx, top, None]
    y += self.shared_experts(x)                                     # (B)
    if world_size > 1:                                              # (C)
        dist.all_reduce(y)

(A) `experts_start_idx`/`experts_end_idx` are computed in `MoE.__init__` from
    `rank * (n_routed_experts // world_size)`. With world_size==1 at
    construction that range is 0..255 on BOTH ranks, so every rank would run
    every expert. We install our own ep_start_idx/ep_end_idx from the REAL
    rank and world size, known only after dist init.

(B) In TP, `shared_experts` is a sharded MLP built with reduce_output=False, so
    each rank holds a PARTIAL value and adding it before the all_reduce is
    correct. In DP every rank holds the FULL copy -- adding before the reduce
    would DOUBLE-COUNT it. It must be added after. This is the subtle one; if
    token-exact verification fails, suspect this first.

(C) The `world_size > 1` guard never fires (module global stays 1), so without
    an unconditional reduce each rank would keep only its own experts'
    contribution. MOE_REDUCE_DTYPE=bf16 halves the payload; the legacy lane
    measured all_reduce at ~16.6% of decode time.

Pruning note: with the offline-pruned BF16 artifacts the non-local Expert
modules are still CONSTRUCTED (all 256 exist) but their weights were never
written, so they hold uninitialized memory. Setting them to None is therefore
REQUIRED, not merely a memory optimization -- it also lets `gc` reclaim the
allocations that construction made.
"""

import gc
import os
import types
from typing import Any, Dict


def _reduce_dtype_from_env(default: str = "fp32"):
    import torch

    val = os.environ.get("MOE_REDUCE_DTYPE", default).strip().lower()
    return torch.bfloat16 if val == "bf16" else torch.float32


def _make_ep_moe_forward(reduce_dtype):
    import torch
    import torch.distributed as dist

    def _ep_moe_forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x_flat = x.view(-1, self.dim)
        weights, indices = self.gate(x_flat)
        y = torch.zeros_like(x_flat, dtype=torch.float32)
        counts = torch.bincount(
            indices.flatten(), minlength=self.n_routed_experts
        ).tolist()

        # (A) our EP range, not upstream's TP range
        import moe_batched
        if moe_batched.ENABLED:
            y = moe_batched.run_experts_sorted(
                self, x_flat, weights, indices, self.ep_start_idx, self.ep_end_idx)
        else:
            for i in range(self.ep_start_idx, self.ep_end_idx):
                if counts[i] == 0:
                    continue
                idx, top = torch.where(indices == i)
                y[idx] += self.experts[i](x_flat[idx]) * weights[idx, top, None]

        # (C) unconditional -- upstream's `if world_size > 1` never fires in DP
        if reduce_dtype is torch.bfloat16:
            y_comm = y.to(torch.bfloat16)
            dist.all_reduce(y_comm)
            y = y_comm.to(torch.float32)
        else:
            dist.all_reduce(y)

        # (B) AFTER the reduce: shared_experts is replicated in DP
        y += self.shared_experts(x_flat)
        return y.type_as(x_flat).view(shape)

    return _ep_moe_forward


def _make_true_dp_moe_forward(reduce_dtype):
    """MoE forward for TRUE DP, where `x` differs per rank (sequence-sharded).

    `_ep_moe_forward` above assumes every rank holds the SAME `x` -- true only
    because dp2_epon replicates attention. Under true DP rank r holds only
    batch[r::world_size], so a token on rank 0 routinely needs an expert that
    lives on rank 1. Token and expert have to meet somewhere.

    Design (a) from PARALLELISM_PLAN_FINAL.md item 7: gather the TOKENS to every
    rank, run local experts over all of them, scatter the result back.

        x_all   = all_gather(x_local)     every rank sees the whole batch
        w, idx  = gate(x_all)             gate is replicated -> identical routing,
                                          never communicated
        y_all   = local_experts(x_all)    experts [r*n_local, (r+1)*n_local)
        reduce_scatter(y_local, y_all)    sum partials, keep only our slice
        y_local += shared_experts(x_local)

    Design (b) -- all_to_all dispatch, moving only tokens that need a remote
    expert -- is NOT IMPLEMENTABLE on this backend. scripts/probe_collectives.py
    on 2 ranks: `RuntimeError: Backend gloo does not support alltoall`. It would
    need MPI or oneCCL. At world_size=2 it would save little anyway, since with
    top-8-of-256 roughly half of each token's experts are remote regardless.

    reduce_scatter rather than all_reduce+slice: the same probe confirmed gloo
    implements reduce_scatter correctly at world_size=2. The plan hedged with
    all_reduce because it was unverified; reducing straight into the local slice
    moves B/world_size instead of B and drops the discard.

    Requires every rank to hold the same NUMBER of rows, since reduce_scatter
    splits equally -- enforced by install_true_dp_moe refusing B % world_size.
    """
    import torch
    import torch.distributed as dist

    def _true_dp_moe_forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x_local = x.view(-1, self.dim)
        world = dist.get_world_size()

        # (1) gather tokens. Fixed-size: all ranks hold equal rows by construction.
        gathered = [torch.empty_like(x_local) for _ in range(world)]
        dist.all_gather(gathered, x_local.contiguous())
        x_all = torch.cat(gathered, dim=0)

        # (2) routing is derived, never communicated -- the gate is replicated and
        #     sees identical input, so every rank computes identical indices.
        weights, indices = self.gate(x_all)
        y_all = torch.zeros_like(x_all, dtype=torch.float32)
        counts = torch.bincount(
            indices.flatten(), minlength=self.n_routed_experts
        ).tolist()

        # (3) our experts over ALL tokens
        import moe_batched
        if moe_batched.ENABLED:
            y_all = moe_batched.run_experts_sorted(
                self, x_all, weights, indices, self.ep_start_idx, self.ep_end_idx)
        else:
            for i in range(self.ep_start_idx, self.ep_end_idx):
                if counts[i] == 0:
                    continue
                idx, top = torch.where(indices == i)
                y_all[idx] += self.experts[i](x_all[idx]) * weights[idx, top, None]

        # (4) sum partials and keep only our slice, in one collective
        comm = y_all.to(torch.bfloat16) if reduce_dtype is torch.bfloat16 else y_all
        y_local = torch.empty_like(x_local, dtype=comm.dtype)
        dist.reduce_scatter(y_local, list(comm.chunk(world, dim=0)))
        y_local = y_local.to(torch.float32)

        # (5) shared experts are replicated and depend only on local tokens
        y_local += self.shared_experts(x_local)
        return y_local.type_as(x_local).view(shape)

    return _true_dp_moe_forward


def install_true_dp_moe(
    transformer: Any,
    model_module: Any,
    dist_env: Dict[str, Any],
    n_routed_experts: int,
    ep_size: int,
    batch_size: int,
    log_fn=print,
) -> Dict[str, Any]:
    """Prune non-local experts and bind the forward appropriate to `batch_size`.

    TWO MODES, chosen once here from B -- never per call:

      B == 1          -> replicated attention + `_ep_moe_forward` (today's
                         dp2_epon, token-exact validated). With world_size=2 a
                         sharded batch of 1 would leave a rank with ZERO rows,
                         and handling that cleanly would mean overriding
                         upstream `Transformer.forward` -- i.e. copying upstream
                         numerics into this repo, which this lane avoids.
      B >= world_size -> sequence-sharded attention + `_true_dp_moe_forward`,
                         which is where the KV saving and the attention split
                         actually come from.

    The B=1 fallback is logged loudly because it is a silent-wrong-number trap:
    a run tagged dp2_true at B=1 is NOT measuring true DP, and the mode is
    recorded in the returned report so results carry it.

    B must divide world_size evenly: reduce_scatter splits equally, and ragged
    per-rank row counts would misalign the slices.
    """
    world_size = int(dist_env["world_size"])
    if batch_size != 1 and batch_size % world_size != 0:
        raise ValueError(
            f"true DP needs batch_size divisible by world_size "
            f"(reduce_scatter splits equally); got B={batch_size} world={world_size}"
        )

    report = install_ep_moe(
        transformer, model_module, dist_env,
        n_routed_experts=n_routed_experts, ep_size=ep_size, log_fn=log_fn,
    )

    if batch_size == 1:
        report["attention_mode"] = "replicated"
        report["moe_forward"] = "_ep_moe_forward"
        log_fn(
            "[true-dp] B=1: FALLING BACK to replicated attention. This run is "
            "NOT measuring true DP -- it is dp2_epon. Use B>=2 (a multiple of "
            f"world_size={world_size}) for sequence-sharded attention."
        )
        return report

    ep_forward = _make_true_dp_moe_forward(_reduce_dtype_from_env())
    bound = 0
    for layer in transformer.layers:
        ffn = getattr(layer, "ffn", None)
        if not isinstance(ffn, model_module.MoE):
            continue
        ffn.forward = types.MethodType(ep_forward, ffn)
        bound += 1
    report["attention_mode"] = "sequence_sharded"
    report["moe_forward"] = "_true_dp_moe_forward"
    report["true_dp_layers"] = bound
    log_fn(
        f"[true-dp] sequence-sharded: rebound {bound} MoE layers to the "
        f"all_gather/reduce_scatter forward; B={batch_size} over {world_size} ranks "
        f"= {batch_size // world_size} sequences/rank"
    )
    return report


def install_ep_moe(
    transformer: Any,
    model_module: Any,
    dist_env: Dict[str, Any],
    n_routed_experts: int,
    ep_size: int,
    log_fn=print,
) -> Dict[str, Any]:
    """Prune non-local experts and bind the EP-aware MoE forward.

    Must run AFTER dist init (needs the real rank/world_size) and AFTER weight
    loading (pruning drops modules the loader would otherwise populate).
    """
    import torch

    rank = int(dist_env["rank"])
    world_size = int(dist_env["world_size"])
    # The split below is n_routed_experts // world_size, so EP_SIZE is only
    # descriptive unless it matches. Without this, EP_SIZE=1 would still shard
    # experts across both ranks and the config would silently misreport.
    if ep_size != world_size:
        raise ValueError(
            f"EP_SIZE={ep_size} != world_size={world_size}; the EP split is driven "
            f"by world_size, so these must agree (set EP_SIZE={world_size})"
        )
    if n_routed_experts % world_size != 0:
        raise ValueError(
            f"n_routed_experts={n_routed_experts} not divisible by world_size={world_size}"
        )

    n_local = n_routed_experts // world_size
    ep_start = rank * n_local
    ep_end = ep_start + n_local
    reduce_dtype = _reduce_dtype_from_env()
    ep_forward = _make_ep_moe_forward(reduce_dtype)

    log_fn(
        f"[ep-moe] rank {rank}/{world_size}: local experts [{ep_start}, {ep_end}) "
        f"of {n_routed_experts}; reduce dtype={reduce_dtype}"
    )

    moe_layers = 0
    pruned = 0
    for layer in transformer.layers:
        ffn = getattr(layer, "ffn", None)
        if not isinstance(ffn, model_module.MoE):
            continue
        moe_layers += 1
        ffn.ep_start_idx = ep_start
        ffn.ep_end_idx = ep_end
        for i in range(n_routed_experts):
            if ep_start <= i < ep_end:
                continue
            if ffn.experts[i] is not None:
                ffn.experts[i] = None
                pruned += 1
        ffn.forward = types.MethodType(ep_forward, ffn)

    gc.collect()
    log_fn(
        f"[ep-moe] installed on {moe_layers} MoE layers; pruned {pruned} "
        f"non-local Expert modules"
    )
    return {
        "rank": rank,
        "world_size": world_size,
        "ep_start_idx": ep_start,
        "ep_end_idx": ep_end,
        "n_local_experts": n_local,
        "moe_layers": moe_layers,
        "pruned_experts": pruned,
        "reduce_dtype": str(reduce_dtype),
    }
