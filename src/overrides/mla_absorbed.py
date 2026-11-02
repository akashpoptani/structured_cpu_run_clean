"""Chunked prefill for long context, via the ABSORBED (MQA) attention path.

Upstream only supports single-shot prefill, in two ways that both break the
moment start_pos > 0:

  1. model.py:903 builds mask as [seqlen, seqlen].
  2. The MHA prefill branch (model.py:574-589) computes
         scores = einsum("bshd,bthd->bsht", q, k)
     where k comes from wkv_b(kv) on the LOCAL chunk -- it never reads
     kv_cache. Chunk 2 would simply not see chunk 1.
  3. Indexer does index_score[bsz, seqlen, end_pos] += mask[seqlen, seqlen],
     which only lines up when end_pos == seqlen.

That design forces the whole prompt through one forward, so index_score and
index_mask are [bsz, Lin, Lin] fp32 -- 57.6 GB EACH at Lin=120k, B=1. Long
context is unreachable before any kernel of ours runs.

Fix: use the absorbed path for prefill as well as decode. MLA absorption makes
it exact, not an approximation --
    q_nope @ wkv_b_k^T @ kv_cache^T  ==  (q_nope @ wkv_b_k^T) @ kv_cache^T
is plain associativity, and the decode branch already reads the full cache
correctly for any seqlen. Consequences:

  * no per-head K/V materialization. wkv_b(kv) expands 576 shared latent dims
    to 128 heads x 320 dims = 10.7 GB per layer at 128k; the absorbed form
    never builds it.
  * ~3x more Q*K^T compute (576 latent dims vs 192 per-head). That is why
    upstream prefers MHA for short prefill. native_run.py installs this only
    when PREFILL_CHUNK > 0, so short-prompt runs keep the upstream path.
  * the score tensor is bounded by chunk_size, not Lin.

Masks are rebuilt here as [seqlen, end_pos] and passed down to the indexer, so
both stay correct at any start_pos.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

# Per-stage wall time inside MLA.forward, summed across layers and drained by
# generation._stage_summary after each prefill forward. Says which of
# qkv / mask / indexer / attention grows with end_pos, instead of guessing.
STAGE_MS: Dict[str, float] = {}


def _tick(name, t0):
    STAGE_MS[name] = STAGE_MS.get(name, 0.0) + (time.perf_counter() - t0) * 1e3


# cmask is identical for all 61 layers of a chunk -- it depends only on
# (start_pos, seqlen, end_pos). Rebuilding it per layer allocated a
# [2048 x 88064] fp32 tensor (721 MB) 61 times per chunk. One-entry cache.
_CMASK_KEY = None
_CMASK = None

# Prefill attention-core kernel, set by amx_prefill_attention.install_amx_prefill_attention().
# Read at call time: when set, it replaces the torch score/mask/softmax/value
# core below; projections, RoPE, cache writes, indexer and wo stay shared.
PREFILL_IMPL: Optional[Callable[..., Any]] = None

# KV_CACHE_DTYPE, set by install_mla_absorbed. "fp8" keeps upstream's fp8
# round-trip on kv before the bf16 cache write; "bf16" stores kv as is, and then
# decode runs here too, since upstream's decode branch would round-trip again.
KV_CACHE_DTYPE = "fp8"


def _causal_mask(x, seqlen: int, start_pos: int, end_pos: int):
    """[seqlen, end_pos] causal mask, cached across the layers of one chunk."""
    import torch
    global _CMASK_KEY, _CMASK
    if seqlen <= 1:
        return None
    key = (seqlen, start_pos, end_pos)
    if key == _CMASK_KEY:
        return _CMASK
    t0 = time.perf_counter()
    j = torch.arange(end_pos, device=x.device)[None, :]
    i = torch.arange(seqlen, device=x.device)[:, None] + start_pos
    _CMASK = torch.where(j <= i, 0.0, float("-inf")).to(torch.float32)
    _CMASK_KEY = key
    _tick("mask_build", t0)
    return _CMASK


def install_mla_absorbed(
    transformer: Any,
    model_module: Any,
    log_fn: Callable[[str], None] = print,
    kv_cache_dtype: str = "fp8",
) -> Dict[str, Any]:
    """Rebind MLA.forward so prefill uses the absorbed path with correct masks."""
    import torch
    global KV_CACHE_DTYPE
    assert kv_cache_dtype in ("fp8", "bf16"), kv_cache_dtype
    KV_CACHE_DTYPE = kv_cache_dtype

    MLA = model_module.MLA
    act_quant = model_module.act_quant
    block_size = model_module.block_size
    apply_rotary_emb = model_module.apply_rotary_emb
    weight_dequant = model_module.weight_dequant

    def attention_torch(self, q_nope, q_pe, topk_indices, cmask, bsz, seqlen, end_pos):
        """Dense core: score every cached position, -inf all but top-k, softmax, x V."""
        scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos])
                  + torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])
                  ) * self.softmax_scale
        # In-place throughout. Each [bsz, seqlen, H, end_pos] fp32 temporary is
        # 2.1 GB at chunk=1024/end_pos=4096, and the out-of-place form allocated
        # three of them per layer (mask-add, softmax, cast) on top of ~600 GB of
        # resident weights. That is what wedged the first attempt, not the math:
        # a per-layer microbenchmark put the absorbed core at 0.68 s, on par with
        # MHA's 2.66 s for 4x the tokens.
        index_mask = torch.full((bsz, seqlen, end_pos), float("-inf"),
                                device=q_nope.device).scatter_(-1, topk_indices, 0)
        if cmask is not None:
            index_mask += cmask
        scores += index_mask.unsqueeze(2)
        del index_mask
        scores = scores.softmax(dim=-1, dtype=torch.float32).to(q_nope.dtype)
        return torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])

    def forward_prefill(self, x, start_pos: int, freqs_cis, mask):
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen

        _t = time.perf_counter()
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr)
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)

        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv = self.kv_norm(kv)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        if KV_CACHE_DTYPE == "fp8":
            # upstream simulates an fp8 kv cache by round-tripping through fp8
            kv_fp8, kv_scale = act_quant(kv, block_size, self.scale_fmt)
            kv = (kv_fp8.view(-1, block_size).float()
                  * kv_scale.view(-1, 1)).to(kv.dtype).view_as(kv)
        self.kv_cache[:bsz, start_pos:end_pos] = kv
        self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
        _tick("qkv_proj", _t)

        # [seqlen, end_pos] causal mask, valid at any start_pos. None in decode.
        cmask = _causal_mask(x, seqlen, start_pos, end_pos)

        if self.dequant_wkv_b is None and self.wkv_b.scale is not None:
            self.dequant_wkv_b = weight_dequant(self.wkv_b.weight, self.wkv_b.scale)
        wkv_b = self.wkv_b.weight if self.dequant_wkv_b is None else self.dequant_wkv_b
        wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)

        _t = time.perf_counter()
        q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
        _tick("absorb_q", _t)

        # the indexer needs the SAME [seqlen, end_pos] mask, not [seqlen, seqlen]
        _t = time.perf_counter()
        topk_indices = self.indexer(x, qr, start_pos, freqs_cis, cmask)
        _tick("indexer", _t)

        _t = time.perf_counter()
        if PREFILL_IMPL is not None and seqlen > 1:
            # AMX kernel: reads only each query's top-k cache rows and applies
            # the causal filter itself. Its tiles are 16 heads of one query, so
            # q needs [B,S,H,D] memory order; the absorb einsum above returns a
            # head-major view ([B,H,S,D] in memory), hence one transpose-copy
            # (134 MB at chunk 2048). Cache slabs and topk are passed as-is.
            y = PREFILL_IMPL(q_nope.contiguous(), q_pe.contiguous(),
                             self.kv_cache[:bsz], self.pe_cache[:bsz],
                             topk_indices, float(self.softmax_scale),
                             start_pos=start_pos, end_pos=end_pos)
        else:
            y = attention_torch(self, q_nope, q_pe, topk_indices, cmask, bsz, seqlen, end_pos)
        _tick("attention", _t)
        _t = time.perf_counter()
        y = torch.einsum("bshc,hdc->bshd", y, wkv_b[:, -self.v_head_dim:])
        out = self.wo(y.flatten(2))
        _tick("out_proj", _t)
        return out

    upstream_forward = MLA.forward

    def forward_dispatch(self, x, start_pos, freqs_cis, mask):
        # fp8 decode (seqlen == 1) stays on upstream's own MQA branch, untouched.
        # forward_prefill is that same branch generalised to seqlen > 1; bf16
        # decode uses it (torch core) so the cache write skips the round-trip.
        if x.size(1) == 1 and KV_CACHE_DTYPE == "fp8":
            return upstream_forward(self, x, start_pos, freqs_cis, mask)
        return forward_prefill(self, x, start_pos, freqs_cis, mask)

    MLA.forward = forward_dispatch
    log_fn("[mla-absorbed] MLA.forward rebound: absorbed path for prefill "
           f"([seqlen, end_pos] masks, no per-head K/V); KV_CACHE_DTYPE={KV_CACHE_DTYPE}; "
           f"decode {'stays upstream' if KV_CACHE_DTYPE == 'fp8' else 'absorbed, torch core'}; "
           f"prefill attention core: {'AMX kernel' if PREFILL_IMPL is not None else 'torch'}")
    return {"installed": True, "path": "absorbed", "layers": len(transformer.layers)}
