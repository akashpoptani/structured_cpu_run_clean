"""Dump real Indexer top-k indices from a prefill pass.

Why: sparse PREFILL cannot be designed without knowing how much adjacent query
positions' top-k sets overlap. In DECODE there is one index list for the single
query, so the kernel gathers 2048 tokens once. In PREFILL every query position
has its OWN top-2048 (model.py:583-586), so a 16-row AMX query tile wants 16
different sets. Which kernel structure wins depends entirely on that overlap:

  * high overlap  -> union-gather per tile is cheap, tile stays dense
  * low overlap   -> union explodes toward 16*2048 and per-query gather (M=1,
                     the thin-M regime that caps decode) is the only option
  * clustered     -> mask-over-dense-blocks wins, driven by block skip rate

model.py's own formulation does not settle this and does not scale either: line
584 materializes index_mask as a dense [bsz, seqlen, seqlen] tensor, which is
1.7e10 entries at 128k. There is no upstream implementation to copy.

Sparsity only EXISTS when end_pos > index_topk (2048): below that
topk(min(index_topk, end_pos)) selects every position and the mask is all-zero.
So the run that feeds this must use Lin > 2048.

Installed per-instance (not by patching the class) so each wrapped indexer knows
its own layer index without the forward having to identify itself.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, Iterable, Optional


def install_indexer_dump(
    transformer: Any,
    outdir: str,
    layers: Iterable[int] = (0,),
    rank: int = 0,
    log_fn: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Wrap Indexer.forward on selected layers to save prefill top-k indices.

    Only the prefill call is captured (seqlen > 1); decode calls are passed
    through untouched. Only `rank` writes -- model.py:485-486 asserts every rank
    computes identical topk_indices, so the other ranks would duplicate it.
    """
    import torch

    wanted = sorted(set(int(i) for i in layers))
    if rank == 0:
        os.makedirs(outdir, exist_ok=True)
    manifest: Dict[str, Any] = {"layers": [], "outdir": outdir}
    installed = 0

    for i, layer in enumerate(transformer.layers):
        if i not in wanted:
            continue
        indexer = getattr(getattr(layer, "attn", None), "indexer", None)
        if indexer is None:
            log_fn(f"[indexer-dump] layer {i}: no .attn.indexer, skipped")
            continue

        def make(layer_idx: int, orig: Callable[..., Any]) -> Callable[..., Any]:
            def fwd(x, qr, start_pos, freqs_cis, mask):
                out = orig(x, qr, start_pos, freqs_cis, mask)
                seqlen = x.size(1)
                if seqlen > 1 and rank == 0:          # prefill pass only
                    path = os.path.join(outdir, f"topk_layer{layer_idx:03d}.npy")
                    if not os.path.exists(path):      # first prefill wins
                        arr = out.detach().to(torch.int32).cpu().numpy()
                        import numpy as np
                        np.save(path, arr)
                        entry = {
                            "layer": layer_idx, "path": path,
                            "shape": list(arr.shape), "start_pos": int(start_pos),
                            "seqlen": int(seqlen), "topk": int(arr.shape[-1]),
                            "sparse": bool(arr.shape[-1] < start_pos + seqlen),
                        }
                        manifest["layers"].append(entry)
                        with open(os.path.join(outdir, "manifest.json"), "w") as f:
                            json.dump(manifest, f, indent=2)
                        log_fn(f"[indexer-dump] layer {layer_idx}: saved {arr.shape} -> {path}"
                               + ("" if entry["sparse"] else
                                  "  WARNING: topk == end_pos, NOT sparse (need Lin > 2048)"))
                return out
            return fwd

        indexer.forward = make(i, indexer.forward)
        installed += 1

    log_fn(f"[indexer-dump] wrapped {installed} indexer(s) on layers {wanted}; outdir={outdir}")
    manifest["installed"] = installed
    return manifest
