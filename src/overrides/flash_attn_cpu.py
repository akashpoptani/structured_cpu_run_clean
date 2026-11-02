"""ctypes binding for the AMX sparse attention kernel.

Loads FlashMLA_CPU/libflashmla_cpu.so and installs it as the attention core
used by chunked_prefill.ATTN_IMPL. It serves BOTH prefill chunks and decode,
because the absorbed (MQA) path gives them the same shape -- the only
difference is seqlen, which is the kernel's M dimension. seqlen=1 is decode;
seqlen=chunk is prefill, and the larger M is strictly better for AMX.

Binding is ctypes on tensor.data_ptr() rather than pybind so there is no build
step beyond the .so, and no second copy of the tensors.

Operand contract (asserted below, because a silent layout mismatch here would
produce plausible-looking wrong logits):
    Q    [B][Sq][H][576]  bf16 contiguous
    KV   [B][Skv][576]    bf16 contiguous
    topk [B][Sq][K]       int32 contiguous
    out  [B][Sq][H][512]  float32
"""
from __future__ import annotations

import ctypes
import os
from typing import Any, Callable, Dict, Optional

_LIB: Optional[ctypes.CDLL] = None          # prefill kernel
_LIB_DEC: Optional[ctypes.CDLL] = None      # decode kernel
_FLASHMLA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "FlashMLA_CPU")
_DEFAULT_SO = os.path.join(_FLASHMLA_DIR, "libflashmla_prefill.so")
_DEFAULT_SO_DEC = os.path.join(_FLASHMLA_DIR, "libflashmla_decode.so")

# Tunables. AMX tiles are 16x16, so a 16-wide M is fixed by the hardware -- what
# IS tunable is the KV block size, the thread count, and (decode) the KV-split
# chunk count the built-in dispatcher picks. 0 = let the kernel decide.
# FLASHMLA_DECODE=0 keeps the prefill kernel but leaves DECODE on torch, so an
# A/B differs in exactly one thing. Without this the only decode comparisons we
# had were across different jobs on different nodes.
USE_DECODE_KERNEL = os.environ.get("FLASHMLA_DECODE", "1") != "0"
BLOCK_SIZE = int(os.environ.get("FLASHMLA_BLOCK_SIZE", "0"))
N_THREADS = int(os.environ.get("FLASHMLA_THREADS", "0"))


def _load_decode(so_path: str) -> ctypes.CDLL:
    lib = ctypes.CDLL(so_path)
    lib.flashmla_decode_sparse.restype = ctypes.c_int
    lib.flashmla_reset_timings.restype = None
    lib.flashmla_get_timings.restype = None
    lib.flashmla_get_timings.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
    lib.flashmla_decode_sparse.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
        ctypes.c_int, ctypes.c_int,
    ]
    return lib


def decode_attention(q576, kv, pe, topk, scale: float):
    """[B,1,H,576] x [B,Skv,512] x [B,Skv,64] x [B,1,K] -> [B,1,H,512] bf16.

    Replaces upstream's score-everything-then-mask, which at end_pos=131072
    computes 131072 scores per head to keep 2048.
    """
    import torch

    assert _LIB_DEC is not None
    B, S, H, Dlat = q576.shape
    assert S == 1, f"decode kernel expects seqlen 1, got {S}"
    Skv, K = kv.shape[1], topk.shape[-1]
    out = torch.empty((B, 1, H, 512), dtype=torch.bfloat16)
    rc = _LIB_DEC.flashmla_decode_sparse(
        ctypes.c_void_p(q576.data_ptr()), ctypes.c_void_p(kv.data_ptr()),
        ctypes.c_void_p(pe.data_ptr()), ctypes.c_void_p(topk.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        B, H, Skv, K, ctypes.c_float(scale), BLOCK_SIZE, N_THREADS)
    if rc != 0:
        raise RuntimeError(f"flashmla_decode_sparse failed (rc={rc})")
    return out


_STAGES = ("gather", "kpack", "qkt", "softmax", "vpack", "amxv",
           "barrier", "reduce", "map_total", "calls")


def reset_timings() -> None:
    """Zero the decode kernel's stage accumulators."""
    if _LIB_DEC is not None:
        _LIB_DEC.flashmla_reset_timings()


def get_timings() -> Dict[str, float]:
    """Decode kernel stage breakdown, summed since the last reset.

    This is what separates gather cost from the rest of TPOT: the kernel runs
    61x per token, so per-call printing is useless, but the accumulators give a
    clean per-step split at any batch size.
    """
    if _LIB_DEC is None:
        return {}
    buf = (ctypes.c_double * 10)()
    _LIB_DEC.flashmla_get_timings(buf, 10)
    d = {k: float(v) for k, v in zip(_STAGES, buf)}
    n = d.pop("calls", 0.0)
    d["calls"] = n
    if n > 0:
        d.update({f"{k}_per_call_ms": d[k] / n * 1e3
                  for k in ("gather", "kpack", "qkt", "softmax", "vpack", "amxv")})
    return d


def _load(so_path: str) -> ctypes.CDLL:
    lib = ctypes.CDLL(so_path)
    lib.flashmla_prefill_sparse.restype = ctypes.c_int
    lib.flashmla_prefill_sparse.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_float,
    ]
    return lib


def sparse_attention(q576, kv, pe, topk, scale: float):
    """[B,Sq,H,576] x [B,Skv,512] x [B,Skv,64] x [B,Sq,K] -> [B,Sq,H,512] f32.

    kv_cache and pe_cache are passed SEPARATELY and concatenated inside the
    gather. Concatenating them in torch first would copy 138 MB per layer per
    chunk -- about 497 GB over a 120k prefill -- for data the kernel touches
    only ~1.2x topk of anyway.
    """
    import torch

    assert _LIB is not None, "call install_flash_attention() first"
    B, Sq, H, Dlat = q576.shape
    Skv = kv.shape[1]
    K = topk.shape[-1]
    assert Dlat == 576 and kv.shape[-1] == 512 and pe.shape[-1] == 64, \
        (q576.shape, kv.shape, pe.shape)
    assert q576.dtype == torch.bfloat16 and kv.dtype == torch.bfloat16 and pe.dtype == torch.bfloat16
    assert topk.dtype == torch.int32
    assert q576.is_contiguous() and kv.is_contiguous() and pe.is_contiguous() and topk.is_contiguous()

    out = torch.empty((B, Sq, H, 512), dtype=torch.float32)
    rc = _LIB.flashmla_prefill_sparse(
        ctypes.c_void_p(q576.data_ptr()), ctypes.c_void_p(kv.data_ptr()),
        ctypes.c_void_p(pe.data_ptr()), ctypes.c_void_p(topk.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        B, H, Sq, Skv, Dlat, K, ctypes.c_float(scale))
    if rc != 0:
        raise RuntimeError(f"flashmla_prefill_sparse failed (rc={rc}); "
                           f"rc=-1 means AMX tile permission was denied")
    return out


def install_flash_attention(so_path: Optional[str] = None,
                            log_fn: Callable[[str], None] = print) -> Dict[str, Any]:
    """Point chunked_prefill.ATTN_IMPL at the kernel. Call BEFORE
    install_chunked_prefill(), which reads ATTN_IMPL when it binds the forward."""
    global _LIB, _LIB_DEC
    import chunked_prefill

    rep: Dict[str, Any] = {"block_size": BLOCK_SIZE or "auto",
                           "threads": N_THREADS or "auto"}
    path = so_path or _DEFAULT_SO
    if os.path.exists(path):
        _LIB = _load(path)
        chunked_prefill.ATTN_IMPL = sparse_attention
        log_fn(f"[flash-attn-cpu] prefill kernel: {path}")
        rep["prefill"] = path
    else:
        log_fn(f"[flash-attn-cpu] prefill kernel missing ({path}); torch einsum")
        rep["prefill"] = None

    dec = os.path.join(_FLASHMLA_DIR, "libflashmla_decode.so")
    if not USE_DECODE_KERNEL:
        log_fn("[flash-attn-cpu] decode kernel DISABLED (FLASHMLA_DECODE=0); torch einsum")
        rep["decode"] = None
    elif os.path.exists(dec):
        _LIB_DEC = _load_decode(dec)
        chunked_prefill.DECODE_IMPL = decode_attention
        log_fn(f"[flash-attn-cpu] decode kernel:  {dec}")
        rep["decode"] = dec
    else:
        log_fn(f"[flash-attn-cpu] decode kernel missing ({dec}); torch einsum")
        rep["decode"] = None

    rep["installed"] = bool(rep["prefill"] or rep["decode"])
    return rep
