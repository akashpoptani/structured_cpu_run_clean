"""Pre-dequantize FP8 weights to BF16 once at model load.

DeepSeek-V3.2 stores most Linear weights as `torch.float8_e4m3fn` paired with a
companion `(ceildiv(out, 128), ceildiv(in, 128))` FP32 block scale. The
upstream `linear()` in DeepSeek `model.py` checks `weight.dtype` and either:

  - calls `F.linear` directly when the weight is already BF16 (the OneDNN AMX
    bf16 brgemm path), or
  - calls `fp8_gemm(x, x_scale, weight, weight.scale)` when the weight is
    FP8 — which in this clean lane's `src/overrides/kernel.py` is a
    per-call FP32 dequant + FP32 F.linear + cast-back. That is the legacy
    fallback path, kept for cases where pre-dequant is not desired.

For the TP2 token-exact verification path we pre-dequantize the model in
place once after weight loading, so the per-call FP8 path is never taken.
After this pass, every FP8 Linear weight is replaced with its BF16
equivalent, the `.scale` Parameter is dropped, and `linear()` short-circuits
to `F.linear` in BF16 for the rest of the run.

Algorithm (block-broadcast multiply, no expanded scale grid):
  - View the (out, in) FP8 weight as (s_out, block, s_in, block).
  - Cast to FP32 (this materialization is the dominant cost).
  - Multiply by the (s_out, 1, s_in, 1) FP32 scale.
  - Cast the product to BF16 and reshape to (s_out*block, s_in*block).
  - For ragged shapes (out % block != 0 or in % block != 0) pad with zeros
    in FP8 first, then trim the BF16 result back to (out, in).

An earlier implementation that materialized a full (out, in) FP32 scale grid
via `repeat_interleave` was the dominant cost (60 minutes per rank on the full
checkpoint); the block-broadcast form runs in seconds.

Scopes:
  - "all"   — every FP8 Linear in the model. ~2x memory per converted weight.
              Required for TP2 token-exact baseline.
  - "none"  — skip; FP8 stays FP8. The per-call FP32 fallback in
              `kernel.fp8_gemm` runs instead.

(The legacy lane also supported a "dense" scope for DP2 EP-off memory limits.
That mode is out of scope for the clean lane right now and has been removed
to keep the path narrow. Re-introduce it if and when DP2 EP-off lands.)
"""

import gc
import os
import time
from typing import Callable, Dict

import torch
import torch.nn as nn


DEFAULT_BLOCK_SIZE = 128


def _dequantize_fp8_block(
    weight_fp8: torch.Tensor,
    scale_fp32: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> torch.Tensor:
    """Block-broadcast FP8 -> BF16 dequant. Handles ragged out/in dims.

    Shapes:
      weight_fp8:   (out_features, in_features), dtype=float8_e4m3fn
      scale_fp32:   (s_out, s_in) where s_out = ceildiv(out_features, block_size)
                                  and  s_in  = ceildiv(in_features,  block_size)

    Operation (block dequant pattern):
      - The scale_fp32 tensor is the per-block scale grid. After viewing
        the (padded) weight as (s_out, block, s_in, block), the per-block
        scale is broadcast across the two intra-block dimensions:
          weight_view[s_out, :, s_in, :] *= scale_fp32[s_out, s_in]
      - Reshape back to (s_out*block, s_in*block) and trim back to the
        original (out_features, in_features) if padding was added.

    This is the proven block-aware dequant kernel. It deliberately avoids
    `repeat_interleave` (which would materialize a full (out, in) FP32 scale
    grid — historically the dominant cost in the legacy lane).
    """
    out_features, in_features = weight_fp8.shape
    s_out, s_in = scale_fp32.shape
    pad_out = s_out * block_size - out_features
    pad_in = s_in * block_size - in_features

    if pad_out or pad_in:
        # Pad in FP8 so the view below is contiguous and aligned.
        padded = torch.nn.functional.pad(weight_fp8, (0, pad_in, 0, pad_out))
    else:
        padded = weight_fp8

    bf16 = ((padded.float()
             .view(s_out, block_size, s_in, block_size)
             * scale_fp32.view(s_out, 1, s_in, 1))
            .to(torch.bfloat16)
            .reshape(s_out * block_size, s_in * block_size))

    if pad_out or pad_in:
        return bf16[:out_features, :in_features].contiguous()
    return bf16


def dequantize_fp8_weights(
    model: nn.Module,
    scope: str = "all",
    block_size: int = DEFAULT_BLOCK_SIZE,
    log_fn: Callable[..., None] = print,
) -> Dict[str, int]:
    """Walk the model and convert FP8 Linear weights to BF16 in place.

    Returns a stats dict with counts and byte deltas. The caller may log it.
    """
    if scope == "none":
        log_fn("[dequant] scope=none -> skip; FP8 weights stay FP8")
        return {"dequant_count": 0, "bytes_freed_fp8": 0, "bytes_added_bf16": 0}
    if scope != "all":
        raise ValueError(f"unsupported scope={scope!r}; expected one of 'all', 'none'")

    t0 = time.perf_counter()
    stats = {
        "dequant_count": 0,
        "bytes_freed_fp8": 0,
        "bytes_added_bf16": 0,
    }

    for module_name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if not isinstance(weight, nn.Parameter):
            continue
        if weight.dtype != torch.float8_e4m3fn:
            continue
        scale = getattr(module, "scale", None)
        if not isinstance(scale, nn.Parameter):
            log_fn(f"[dequant] WARN: FP8 weight {module_name}.weight has no .scale Parameter; skipping")
            continue

        numel = weight.data.numel()
        bf16 = _dequantize_fp8_block(weight.data, scale.data, block_size)
        new_weight = nn.Parameter(bf16, requires_grad=False)

        with torch.no_grad():
            del module._parameters["weight"]
            module._parameters["weight"] = new_weight
            if "scale" in module._parameters:
                del module._parameters["scale"]
            module.scale = None

        stats["dequant_count"] += 1
        stats["bytes_freed_fp8"] += numel
        stats["bytes_added_bf16"] += numel * 2

        if stats["dequant_count"] % 200 == 0:
            gc.collect()
            log_fn(
                f"[dequant]   {stats['dequant_count']} weights so far; "
                f"+{stats['bytes_added_bf16']/1e9:.1f} GB BF16, "
                f"-{stats['bytes_freed_fp8']/1e9:.1f} GB FP8"
            )

    gc.collect()
    elapsed = time.perf_counter() - t0
    net = (stats["bytes_added_bf16"] - stats["bytes_freed_fp8"]) / 1e9
    log_fn(
        f"[dequant] scope={scope}: dequantized {stats['dequant_count']} weights "
        f"in {elapsed:.2f}s (mem net +{net:.1f} GB)"
    )
    return stats


def get_scope_from_env(default: str = "all") -> str:
    """Read DEQUANT_FP8_WEIGHTS env var. Only 'all' or 'none' are supported."""
    return os.environ.get("DEQUANT_FP8_WEIGHTS", default).lower()
