import time
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


# ---------------------------------------------------------------------------
# AMX indexer. The pure-torch fp8_index below measured 0.22 TFLOP/s in-model,
# which alone is ~9.1 h of a 9.2 h prefill at Lin=120k: it materialises a
# PER-HEAD [M, 64, N] fp32 tensor (64x the result it returns -- 17 GB at
# M=N=8192, 1.0 TB at M=32768/N=120k) and the GEMM is only 1.4% of the call,
# the rest being four elementwise passes over a 268 MB score matrix, 64 times.
#
# The kernel computes the same thing tile-by-tile with the 16x16 accumulator in
# registers, writing the score matrix exactly once: 23.9-27.9 TFLOP/s, 85-95x.
# bf16 is not a precision loss -- inputs are fp8_e4m3 (3 mantissa bits), bf16
# has 8. Validated: top-2048 SET overlap 100.00% on fp8-quantised inputs
# (job 27930609), which is all the model consumes from these scores.
_IDX_LIB = None
_IDX_TRIED = False


def _idx_lib():
    global _IDX_LIB, _IDX_TRIED
    if _IDX_TRIED:
        return _IDX_LIB
    _IDX_TRIED = True
    import ctypes, os
    so = os.environ.get("FLASHMLA_INDEXER_SO") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "FlashMLA_CPU", "libflashmla_indexer.so")
    if os.environ.get("USE_AMX_INDEXER", "1") == "0" or not os.path.exists(so):
        return None
    lib = ctypes.CDLL(so)
    lib.flashmla_indexer.restype = ctypes.c_int
    lib.flashmla_indexer.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 5
    _IDX_LIB = lib
    return lib


def _fp8_index_amx(q, q_s, k, k_s):
    """None if the kernel cannot take these shapes; caller falls back."""
    import ctypes
    lib = _idx_lib()
    if lib is None or q.dim() != 4:
        return None
    B, M, H, D = q.shape
    N = k.shape[1]
    if D % 32:
        return None
    # kernel needs M%16 and N%64; pad with zeros (relu(0)*scale = 0, so padded
    # columns score 0 and can never enter a top-k of non-negative scores)
    Mp = (M + 15) // 16 * 16
    Np = (N + 63) // 64 * 64
    qb = q.to(torch.bfloat16)
    kb = k.to(torch.bfloat16)
    if Mp != M:
        qb = torch.cat([qb, qb.new_zeros(B, Mp - M, H, D)], 1)
        q_s = torch.cat([q_s, q_s.new_zeros(B, Mp - M, H)], 1)
    if Np != N:
        kb = torch.cat([kb, kb.new_zeros(B, Np - N, D)], 1)
        k_s = torch.cat([k_s, k_s.new_zeros(B, Np - N)], 1)
    out = torch.empty(B, Mp, Np, dtype=torch.float32)
    rc = lib.flashmla_indexer(
        qb.contiguous().data_ptr(), kb.contiguous().data_ptr(),
        q_s.float().contiguous().data_ptr(), k_s.float().contiguous().data_ptr(),
        out.data_ptr(), B, Mp, Np, H, D)
    if rc != 0:
        return None
    return out[:, :M, :N]


# In-model indexer timing. The gather already has an accumulator inside the
# decode kernel (g_acc[0]); this is its counterpart for fp8_index, so the two
# quantities that actually scale with context can be attributed end-to-end
# instead of inferred from standalone harnesses that have disagreed by 188x.
IDX_SECONDS = 0.0
IDX_CALLS = 0


def reset_index_timer():
    global IDX_SECONDS, IDX_CALLS
    IDX_SECONDS = 0.0
    IDX_CALLS = 0


def get_index_timer():
    return {"seconds": IDX_SECONDS, "calls": IDX_CALLS}


# Prefill indexer-score kernel, set by
# amx_prefill_indexer.install_amx_prefill_indexer(). Read at call time and used
# only when M > 1; decode, and prefill without it, keep _fp8_index_default.
INDEX_IMPL = None

# Prefill (M > 1) wall time of fp8_index in ms, summed over layers and drained by
# generation._stage_summary. A sub-part of mla_absorbed's "indexer" stage, which
# also covers the indexer projections, mask add and top-k.
INDEX_MS = 0.0


def fp8_index(
    q: torch.Tensor,
    q_s: torch.Tensor,
    k: torch.Tensor,
    k_s: torch.Tensor,
) -> torch.Tensor:
    global IDX_SECONDS, IDX_CALLS
    import time as _t
    _t0 = _t.perf_counter()
    try:
        return _fp8_index_inner(q, q_s, k, k_s)
    finally:
        IDX_SECONDS += _t.perf_counter() - _t0
        IDX_CALLS += 1


def _fp8_index_inner(
    q: torch.Tensor,
    q_s: torch.Tensor,
    k: torch.Tensor,
    k_s: torch.Tensor,
) -> torch.Tensor:
    global INDEX_MS
    # The upstream kernel consumes per-head and per-token scales without the
    # trailing singleton block dimension that the Python fallback currently sees.
    if q_s.dim() == q.dim():
        q_s = q_s.squeeze(-1)
    if k_s.dim() == k.dim():
        k_s = k_s.squeeze(-1)
    if q.size(1) == 1:
        return _fp8_index_default(q, q_s, k, k_s)
    t0 = time.perf_counter()
    impl = INDEX_IMPL if INDEX_IMPL is not None else _fp8_index_default
    out = impl(q, q_s, k, k_s)
    INDEX_MS += (time.perf_counter() - t0) * 1e3
    return out


def _fp8_index_default(q, q_s, k, k_s):
    """Decode, and prefill without INDEX_IMPL: libflashmla_indexer.so when it
    exists and USE_AMX_INDEXER != 0, else torch."""
    fast = _fp8_index_amx(q, q_s, k, k_s)
    if fast is not None:
        return fast
    return _fp8_index_torch(q, q_s, k, k_s)


def _fp8_index_torch(q, q_s, k, k_s):
    logits = torch.einsum("bmhd,bnd->bmhn", q.float(), k.float())
    logits = logits.clamp_min_(0)
    logits = logits * q_s.float().unsqueeze(-1)
    logits = logits.sum(dim=2)
    logits = logits * k_s.float().unsqueeze(1)
    return logits
