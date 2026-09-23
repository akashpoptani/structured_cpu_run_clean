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


def install_ep_moe(
    transformer: Any,
    model_module: Any,
    dist_env: Dict[str, Any],
    n_routed_experts: int,
    log_fn=print,
) -> Dict[str, Any]:
    """Prune non-local experts and bind the EP-aware MoE forward.

    Must run AFTER dist init (needs the real rank/world_size) and AFTER weight
    loading (pruning drops modules the loader would otherwise populate).
    """
    import torch

    rank = int(dist_env["rank"])
    world_size = int(dist_env["world_size"])
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
