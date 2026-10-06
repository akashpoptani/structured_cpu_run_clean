"""Suppress the Indexer's debug broadcast in upstream `model.py`.

`Indexer.forward` ends with:

    topk_indices_ = topk_indices.clone()
    dist.broadcast(topk_indices_, src=0)
    assert torch.all(topk_indices == topk_indices_)
    return topk_indices

The broadcast writes into a *clone* and the function returns the *original*, so
the received data is discarded: this is a cross-rank consistency assertion, not
a computation. Every rank already derives `topk_indices` identically --

  * `qr` comes from `MLA.wq_a`, a plain `Linear` (model.py:526), not a
    `ColumnParallelLinear`, so it is rank-identical;
  * the Indexer's own `wq_b` / `wk` / `k_norm` / `weights_proj` all map with
    dim `None` in `convert.py` (:30-33), i.e. fully replicated;
  * `Indexer.__init__` sizes `wq_b` with the full `n_heads` (:445);
    `n_local_heads` (:437) is never used in `forward`.

so `index_score` and therefore `topk_indices` are bit-identical by construction.

Why a dist shim rather than rebinding `Indexer.forward`: the broadcast sits in
the middle of the method with no seam, so rebinding would mean copying upstream
numerics (RoPE, act_quant, fp8_index, cache writes) into this repo, where they
would silently diverge from upstream. `dist.broadcast` is called exactly once in
`model.py` (:485) -- every other site is all_reduce / all_gather -- so shimming
that one attribute is surgical.

Cost removed: `topk_indices` is int64 `[bsz, seqlen, min(index_topk, end_pos)]`.
At prefill with seqlen=2048 that is ~33.5 MB per call x 61 layers, i.e. ~2 GB
moved and thrown away per forward.

This is a temporary scaffold. Item 7 (true DP) makes suppression mandatory --
under sharded batches ranks legitimately hold different tokens and the assert
would fail -- at which point the config flag should be deleted and the shim made
unconditional.
"""

from typing import Any, Dict


class _DistShim:
    """Delegates everything to torch.distributed except `broadcast`."""

    def __init__(self, real: Any):
        self._real = real

    def broadcast(self, tensor, src=0, group=None, async_op=False):  # noqa: ARG002
        # Deliberate no-op: the only caller discards the result (model.py:485).
        return None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def install_indexer_broadcast_skip(model_module: Any, log_fn=print) -> Dict[str, Any]:
    """Bind the shim onto `model.py`'s module-global `dist`.

    Scoped to that module's namespace: `torch.distributed` itself is untouched,
    so collectives issued from anywhere else are unaffected.
    """
    current = getattr(model_module, "dist", None)
    if isinstance(current, _DistShim):
        log_fn("[indexer-bcast] shim already installed; skipping")
        return {"installed": False, "reason": "already_installed"}

    model_module.dist = _DistShim(current)
    log_fn(
        "[indexer-bcast] dist.broadcast suppressed in model.py "
        "(61 discarded collectives/forward); all other collectives unchanged"
    )
    return {"installed": True, "suppressed_call_sites": ["model.py:485 Indexer.forward"]}
