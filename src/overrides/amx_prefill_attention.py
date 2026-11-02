"""ctypes binding for the AMX sparse PREFILL attention kernel.

Loads FlashMLA_CPU/amx_prefill_attention.so (build: FlashMLA_CPU/build_libs.sh) and
plugs it into mla_absorbed as PREFILL_IMPL, which replaces the torch
score-everything-then-mask core for seqlen > 1. Decode is not touched.

Binding is ctypes on tensor.data_ptr(): the .so is loaded into this process and
called like a C function, reading and writing torch's memory directly, no copies.
The operand contract is asserted in prefill_attention, because a silent layout
mismatch would produce plausible wrong logits rather than an error.
"""
from __future__ import annotations

import ctypes
import os
from typing import Any, Callable, Dict, Optional

_LIB: Optional[ctypes.CDLL] = None
_DEFAULT_SO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "FlashMLA_CPU", "amx_prefill_attention.so")


def _load(so_path: str) -> ctypes.CDLL:
    lib = ctypes.CDLL(so_path)
    lib.flashmla_prefill_sparse.restype = ctypes.c_int
    lib.flashmla_prefill_sparse.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,          # q_nope, q_pe
        ctypes.c_void_p, ctypes.c_void_p,          # kv_cache, pe_cache slabs
        ctypes.c_void_p, ctypes.c_void_p,          # topk (int64), out (bf16)
        ctypes.c_int, ctypes.c_int, ctypes.c_int,  # B, H, Sq
        ctypes.c_int, ctypes.c_int, ctypes.c_int,  # Skv (= end_pos), kv_stride, K
        ctypes.c_float, ctypes.c_int,              # scale, start_pos
    ]
    return lib


def prefill_attention(q_nope, q_pe, kv_cache, pe_cache, topk, scale: float,
                      start_pos: int, end_pos: int):
    """Sparse prefill attention, zero-copy: every operand is passed as-is.

        q_nope   [B,Sq,H,512] bf16   absorbed (q_nope @ wkv_b_k)
        q_pe     [B,Sq,H,64]  bf16
        kv_cache [B,S,512]    bf16   the model's cache slab, S = max_seq_len;
        pe_cache [B,S,64]     bf16   only rows < end_pos are read
        topk     [B,Sq,K]     int64  straight from the indexer
        -> out   [B,Sq,H,512] bf16

    Positions < start_pos come from earlier chunks via the cache; start_pos is
    also what lets the kernel drop the FUTURE positions topk pads in for
    queries with < 2048 past tokens (CAUSALITY note in the .cpp).
    """
    import torch

    assert _LIB is not None, "call install_amx_prefill_attention() first"
    B, Sq, H, Dn = q_nope.shape
    S, K = kv_cache.shape[1], topk.shape[-1]
    assert Dn == 512 and q_pe.shape == (B, Sq, H, 64), (q_nope.shape, q_pe.shape)
    assert kv_cache.shape == (B, S, 512) and pe_cache.shape == (B, S, 64), \
        (kv_cache.shape, pe_cache.shape)
    assert topk.shape == (B, Sq, K) and topk.dtype == torch.int64, (topk.shape, topk.dtype)
    for t in (q_nope, q_pe, kv_cache, pe_cache):
        assert t.dtype == torch.bfloat16, t.dtype
    # is_contiguous() is a flag check, not a copy. A [:bsz] slice of the cache
    # IS contiguous (leading-dim slice); a [:bsz, :end_pos] slice is NOT, which
    # is why the slab and end_pos are passed separately.
    for t in (q_nope, q_pe, kv_cache, pe_cache, topk):
        assert t.is_contiguous(), t.stride()
    assert 0 <= start_pos and start_pos + Sq == end_pos <= S, (start_pos, Sq, end_pos, S)

    out = torch.empty((B, Sq, H, 512), dtype=torch.bfloat16)
    rc = _LIB.flashmla_prefill_sparse(
        q_nope.data_ptr(), q_pe.data_ptr(), kv_cache.data_ptr(), pe_cache.data_ptr(),
        topk.data_ptr(), out.data_ptr(),
        B, H, Sq, end_pos, S, K, ctypes.c_float(scale), start_pos)
    if rc != 0:
        raise RuntimeError(f"flashmla_prefill_sparse failed (rc={rc}); rc=-1: AMX tile "
                           f"permission denied, rc=-2: bad start_pos/end_pos/stride, "
                           f"rc=-3: H not a multiple of 16")
    return out


def install_amx_prefill_attention(so_path: Optional[str] = None,
                          log_fn: Callable[[str], None] = print) -> Dict[str, Any]:
    """Point mla_absorbed.PREFILL_IMPL at the kernel. Fails loudly if the .so is
    missing: a run that asked for the kernel must not silently measure torch."""
    global _LIB
    import mla_absorbed

    path = so_path or _DEFAULT_SO
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; run bash FlashMLA_CPU/build_libs.sh on an AMX node")
    _LIB = _load(path)
    mla_absorbed.PREFILL_IMPL = prefill_attention
    log_fn(f"[amx-prefill-attention] prefill kernel: {path}")
    return {"installed": True, "prefill": path}
