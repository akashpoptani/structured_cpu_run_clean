"""All-reduce RowParallelLinear outputs in bf16 instead of fp32.

model.py:263-266 does:
    y = linear(x, self.weight, ...)        # bf16
    if self.reduce_output and world_size > 1:
        y = y.float()                      # <-- doubles the bytes on the wire
        dist.all_reduce(y)
    return y.type_as(x)                    # cast straight back

The fp32 round-trip buys nothing: the value is produced in bf16 and consumed in
bf16, and a two-rank sum of two bf16 partials does not need fp32 range. It costs
exactly 2x on every collective.

Scale, measured: at chunk=32768 the out_proj output is [32768, 7168] -- 470 MB
bf16, 940 MB fp32. A 120k prefill fires 476 such collectives (61 out_proj +
58 MoE layers x 4 chunks), so this halves ~450 GB of traffic.

Stacks with routing gloo over ib0 (run_native_distributed.sh): that fixes the
link rate, this halves what crosses it.
"""
from __future__ import annotations

from typing import Any, Callable, Dict


def install_bf16_allreduce(model_module: Any,
                           log_fn: Callable[[str], None] = print) -> Dict[str, Any]:
    import torch
    import torch.distributed as dist

    RPL = model_module.RowParallelLinear
    linear = model_module.linear
    ws = model_module.world_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = linear(x, self.weight, None, self.scale_fmt)
        if self.reduce_output and ws > 1:
            # reduce in the dtype we already have; no .float() round-trip
            y = y.contiguous()
            dist.all_reduce(y)
        if self.bias is not None:
            y = y + self.bias
        # upstream ends with .type_as(x); keep it so the contract is identical
        # even though y is already x's dtype on this path
        return y.type_as(x)

    RPL.forward = forward
    log_fn("[bf16-allreduce] RowParallelLinear.forward rebound: collectives in "
           "bf16 instead of fp32 (half the bytes on every all_reduce)")
    return {"installed": True}
