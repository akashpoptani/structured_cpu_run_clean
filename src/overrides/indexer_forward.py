"""Indexer.forward override: KV_CACHE_DTYPE picks the indexer key cache precision.

  fp8   upstream numerics, line for line (model.py:459-487): q and k go through
        act_quant, k_cache holds fp8 plus a per-key fp32 k_scale_cache, and the
        q scale is folded into the head weights.
  bf16  no quantisation: k_cache is re-allocated as bf16 and holds k as is, q
        stays bf16, and k_scale_cache is set to 1 once so fp8_index's final
        "x k_s" is exact and the scorers need no second signature.

Prefill and decode both run through here, since they share k_cache and a buffer
has one dtype. The upstream debug broadcast (a cross-rank assert whose result is
discarded, see indexer_broadcast.py) is not copied.

STAGE_MS splits the prefill indexer time (seqlen > 1 only) and is drained by
generation._stage_summary next to mla_absorbed's "indexer" total.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict

STAGE_MS: Dict[str, float] = {}


def install_indexer_forward(transformer: Any, model_module: Any, kv_cache_dtype: str = "fp8",
                            log_fn: Callable[[str], None] = print) -> Dict[str, Any]:
    import torch

    assert kv_cache_dtype in ("fp8", "bf16"), kv_cache_dtype
    bf16 = kv_cache_dtype == "bf16"
    act_quant, fp8_index = model_module.act_quant, model_module.fp8_index
    apply_rotary_emb, rotate_activation = model_module.apply_rotary_emb, model_module.rotate_activation
    block_size = model_module.block_size

    def forward(self, x, qr, start_pos: int, freqs_cis, mask):
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        timed = seqlen > 1
        t = time.perf_counter()

        def tick(name):
            nonlocal t
            if timed:
                now = time.perf_counter()
                STAGE_MS[name] = STAGE_MS.get(name, 0.0) + (now - t) * 1e3
                t = now

        q = self.wq_b(qr)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        q_pe, q_nope = torch.split(q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis, False)   # indexer rope is not interleaved
        q = torch.cat([q_pe, q_nope], dim=-1)
        k = self.k_norm(self.wk(x))
        k_pe, k_nope = torch.split(k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, False).squeeze(2)
        k = torch.cat([k_pe, k_nope], dim=-1)
        weights = self.weights_proj(x.float()) * self.n_heads ** -0.5
        tick("idx_proj")
        q = rotate_activation(q)
        k = rotate_activation(k)
        tick("idx_rotate")
        if bf16:
            self.k_cache[:bsz, start_pos:end_pos] = k
            weights = weights.unsqueeze(-1) * self.softmax_scale
        else:
            q, q_scale = act_quant(q, block_size, self.scale_fmt)
            k_fp8, k_scale = act_quant(k, block_size, self.scale_fmt)
            self.k_cache[:bsz, start_pos:end_pos] = k_fp8
            self.k_scale_cache[:bsz, start_pos:end_pos] = k_scale
            weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale
        tick("idx_quant")
        index_score = fp8_index(q.contiguous(), weights, self.k_cache[:bsz, :end_pos].contiguous(),
                                self.k_scale_cache[:bsz, :end_pos].contiguous())
        tick("idx_score")
        if mask is not None:
            index_score += mask
        topk_indices = index_score.topk(min(self.index_topk, end_pos), dim=-1)[1]
        tick("idx_topk")
        return topk_indices

    if bf16:
        for layer in transformer.layers:
            idx = layer.attn.indexer
            idx.k_cache = torch.zeros(idx.k_cache.shape, dtype=torch.bfloat16, device=idx.k_cache.device)
            idx.k_scale_cache.fill_(1.0)
    model_module.Indexer.forward = forward
    log_fn(f"[indexer-forward] Indexer.forward rebound: KV_CACHE_DTYPE={kv_cache_dtype} "
           f"(k_cache {'bf16, no act_quant' if bf16 else 'fp8 + k_scale_cache, upstream act_quant'})")
    return {"installed": True, "kv_cache_dtype": kv_cache_dtype}
