"""ctypes binding for the AMX indexer-score kernel (prefill only).

Loads FlashMLA_CPU/amx_prefill_indexer.so (build: FlashMLA_CPU/build_libs.sh) and
plugs it into kernel.py as INDEX_IMPL, which fp8_index uses whenever M > 1. It
replaces only the score computation; the causal mask and top-k stay in model.py.

The kernel's tile rows are 16 queries and its key tiles 64 wide, so the inputs
are padded to M % 16 == 0 and N % 64 == 0 and the padding is sliced off the
result. fp8 -> bf16 is exact (3 mantissa bits into 8).
"""
from __future__ import annotations

import ctypes
import os
from typing import Any, Callable, Dict, Optional

_LIB: Optional[ctypes.CDLL] = None
_SCRATCH: Dict[str, Any] = {}
_DEFAULT_SO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "FlashMLA_CPU", "amx_prefill_indexer.so")


def _load(so_path: str) -> ctypes.CDLL:
    lib = ctypes.CDLL(so_path)
    lib.flashmla_indexer.restype = ctypes.c_int
    lib.flashmla_indexer.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,          # q, k (bf16)
        ctypes.c_void_p, ctypes.c_void_p,          # q_s [B,M,H], k_s [B,N] (fp32)
        ctypes.c_void_p,                           # out [B,M,N] fp32
        ctypes.c_int, ctypes.c_int, ctypes.c_int,  # B, M, N
        ctypes.c_int, ctypes.c_int,                # H, D
    ]
    return lib


def _scratch(name: str, shape, dtype):
    """Persistent buffer, grown on demand. A fresh large tensor is new mmap'd
    pages, and 88 threads first-touching a 33 MB output cost ~85 ms per call
    (measured at M=2048, N=4096), 20x the kernel's own 4 ms."""
    import math
    import torch
    n = math.prod(shape)
    buf = _SCRATCH.get(name)
    if buf is None or buf.numel() < n:
        buf = _SCRATCH[name] = torch.zeros(n, dtype=dtype)
    return buf[:n].view(shape)


def index_scores(q, q_s, k, k_s):
    """Same contract as kernel.fp8_index, scales already squeezed:

        q   [B,M,H,D] fp8   q_s [B,M,H] fp32   (weights * q_scale * softmax_scale)
        k   [B,N,D]   fp8   k_s [B,N]   fp32
        -> [B,M,N] fp32 = sum_h relu(q.k) * q_s, times k_s

    The result is a view of a persistent buffer, valid until the next call.
    Indexer.forward consumes it (mask add, top-k) before the next layer runs.
    """
    import torch

    assert _LIB is not None, "call install_amx_prefill_indexer() first"
    B, M, H, D = q.shape
    N = k.shape[1]
    assert k.shape == (B, N, D) and q_s.shape == (B, M, H) and k_s.shape == (B, N), \
        (q.shape, q_s.shape, k.shape, k_s.shape)
    assert D % 32 == 0, D
    # One converting copy each into persistent buffers. The padding is left as
    # is: out[m, n] reads only query row m and key row n, so a padded row or key
    # only reaches an output entry that is sliced off below.
    Mp, Np = -(-M // 16) * 16, -(-N // 64) * 64
    qb = _scratch("q", (B, Mp, H, D), torch.bfloat16)
    qb[:, :M] = q
    kb = _scratch("k", (B, Np, D), torch.bfloat16)
    kb[:, :N] = k
    qs = _scratch("q_s", (B, Mp, H), torch.float32)
    qs[:, :M] = q_s
    ks = _scratch("k_s", (B, Np), torch.float32)
    ks[:, :N] = k_s
    out = _scratch("out", (B, Mp, Np), torch.float32)
    rc = _LIB.flashmla_indexer(qb.data_ptr(), kb.data_ptr(), qs.data_ptr(), ks.data_ptr(),
                               out.data_ptr(), B, Mp, Np, H, D)
    if rc != 0:
        raise RuntimeError(f"flashmla_indexer failed (rc={rc}); rc=-1: AMX tile permission "
                           f"denied, rc=-2: shape not padded to M%16 / N%64 / D%32")
    return out[:, :M, :N]


def install_amx_prefill_indexer(so_path: Optional[str] = None,
                        log_fn: Callable[[str], None] = print) -> Dict[str, Any]:
    """Point kernel.INDEX_IMPL at the kernel. Fails loudly if the .so is missing:
    a run that asked for the kernel must not silently measure torch."""
    global _LIB
    import kernel  # src/overrides/kernel.py, the module model.py imports fp8_index from

    path = so_path or _DEFAULT_SO
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; run bash FlashMLA_CPU/build_libs.sh on an AMX node")
    _LIB = _load(path)
    kernel.INDEX_IMPL = index_scores
    log_fn(f"[amx-prefill-indexer] prefill indexer kernel: {path} (kernel module: {kernel.__file__})")
    return {"installed": True, "indexer": path}
