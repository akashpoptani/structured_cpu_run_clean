from typing import Optional, Tuple

import torch
import torch.nn.functional as F


FP8_MAX = 448.0


def _round_scale_pow2(scale: torch.Tensor) -> torch.Tensor:
    scale = scale.clamp_min(1e-8)
    return torch.pow(torch.tensor(2.0, device=scale.device), torch.ceil(torch.log2(scale)))


def _maybe_fp8_storage(x: torch.Tensor) -> torch.Tensor:
    dtype = getattr(torch, "float8_e4m3fn", None)
    if dtype is None:
        return x
    try:
        return x.to(dtype)
    except Exception:
        return x


def act_quant(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.size(-1) % block_size == 0, (
        f"Last dimension size must be divisible by block_size (block_size={block_size})"
    )

    x_f32 = x.float()
    reshaped = x_f32.view(*x.shape[:-1], -1, block_size)
    scales = reshaped.abs().amax(dim=-1).clamp_min(1e-4) / FP8_MAX
    if scale_fmt is not None:
        scales = _round_scale_pow2(scales)

    q = torch.clamp(reshaped / scales.unsqueeze(-1), -FP8_MAX, FP8_MAX).reshape_as(x_f32)
    q = _maybe_fp8_storage(q).contiguous()
    return q, scales.to(torch.float32).contiguous()


def _expand_activation_scales(scales: torch.Tensor, block_size: int, width: int) -> torch.Tensor:
    expanded = scales.repeat_interleave(block_size, dim=-1)
    return expanded[..., :width]


def _dequantize_activation(x: torch.Tensor, scales: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    return x.float() * _expand_activation_scales(scales.float(), block_size, x.shape[-1])


def _dequantize_weight(weight: torch.Tensor, scales: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    expanded = scales.float().repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
    expanded = expanded[: weight.shape[0], : weight.shape[1]]
    return weight.float() * expanded


def fp8_gemm(
    a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor
) -> torch.Tensor:
    """FP8 fallback used by upstream linear() when weight.dtype == float8_e4m3fn.

    This is the LEGACY FP32 path. After Step 7b iter 1, weights are
    pre-dequantized to BF16 once at load (overrides/dequant_weights.py),
    so upstream linear() short-circuits to F.linear and this function
    is NOT taken for tp2 / dp2_epon (where DEQUANT_FP8_WEIGHTS=all).

    For dp2 EP-off (DEQUANT_FP8_WEIGHTS=dense, the only mode that can't
    fit BF16 experts in 1 TiB), this function is hit once per expert
    linear per token, 1392+ times total. Step 7b iter 1.1 and 1.2 tried
    to add BF16 casts (.to(bfloat16) on the big 28 MB dequanted weight)
    so F.linear could engage AMX — but the per-call BF16 cast added
    ~600 s of memory traffic per 15-token gen, more than the AMX speedup
    saved. Kept legacy FP32-dequant + FP32-matmul + output BF16 cast
    here (matches Step 7a baseline). dp2 EP-off optimization needs a
    fused kernel that streams FP8 -> BF16 inside the AMX brgemm — see
    Step 7b iter 3 (deferred).
    """
    assert a.is_contiguous() and b.is_contiguous(), "Input tensors must be contiguous"
    assert a_s.is_contiguous() and b_s.is_contiguous(), "Scale tensors must be contiguous"

    a_deq = _dequantize_activation(a, a_s)
    b_deq = _dequantize_weight(b, b_s)
    out = F.linear(a_deq, b_deq)
    return out.to(torch.get_default_dtype())


def fp8_index(
    q: torch.Tensor,
    q_s: torch.Tensor,
    k: torch.Tensor,
    k_s: torch.Tensor,
) -> torch.Tensor:
    # The upstream kernel consumes per-head and per-token scales without the
    # trailing singleton block dimension that the Python fallback currently sees.
    if q_s.dim() == q.dim():
        q_s = q_s.squeeze(-1)
    if k_s.dim() == k.dim():
        k_s = k_s.squeeze(-1)

    logits = torch.einsum("bmhd,bnd->bmhn", q.float(), k.float())
    logits = logits.clamp_min_(0)
    logits = logits * q_s.float().unsqueeze(-1)
    logits = logits.sum(dim=2)
    logits = logits * k_s.float().unsqueeze(1)
    return logits
