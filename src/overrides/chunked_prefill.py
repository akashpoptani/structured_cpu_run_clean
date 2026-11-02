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
    upstream prefers MHA for short prefill, and why this override keeps the
    upstream path when the prompt fits in one chunk.
  * the score tensor is bounded by chunk_size, not Lin.
  * it is the same math our sparse DECODE kernel implements, so one kernel
    serves both -- with M = chunk_size rather than 1, which is the large-M
    regime AMX actually wants.

Masks are rebuilt here as [seqlen, end_pos] and passed down to the indexer, so
both stay correct at any start_pos.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional

# Set by src/overrides/flash_attn_cpu.install_flash_attention(). When present it
# replaces the einsum/softmax/einsum core below with the AMX kernel; everything
# else in the forward (projections, rope, cache writes, wo) is untouched.
# Per-stage wall time inside MLA.forward, summed across layers. The chunk cost
# decomposes as ~150 s constant (MoE + projections) plus a term growing with
# end_pos; this says which of qkv / indexer / mask / attention the growing part
# actually is, instead of guessing.
STAGE_MS: Dict[str, float] = {}

def _tick(name, t0):
    import time
    STAGE_MS[name] = STAGE_MS.get(name, 0.0) + (time.perf_counter() - t0) * 1e3

# cmask is identical for all 61 layers of a chunk -- it depends only on
# (start_pos, seqlen, end_pos). Rebuilding it per layer allocated a
# [2048 x 88064] fp32 tensor (721 MB) 61 times per chunk. One-entry cache.
_CMASK_KEY = None
_CMASK = None

ATTN_IMPL: Optional[Callable[..., Any]] = None      # prefill, seqlen > 1
DECODE_IMPL: Optional[Callable[..., Any]] = None    # decode,  seqlen == 1


def _causal_mask(x, seqlen: int, start_pos: int, end_pos: int):
    """[seqlen, end_pos] causal mask, cached across the layers of one chunk."""
    import time, torch
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


def install_chunked_prefill(
    transformer: Any,
    model_module: Any,
    log_fn: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Rebind MLA.forward so prefill uses the absorbed path with correct masks."""
    import torch

    MLA = model_module.MLA
    act_quant = model_module.act_quant
    block_size = model_module.block_size
    apply_rotary_emb = model_module.apply_rotary_emb
    weight_dequant = model_module.weight_dequant

    def forward(self, x, start_pos: int, freqs_cis, mask):
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen

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
        # upstream simulates an fp8 kv cache by round-tripping through fp8
        kv_fp8, kv_scale = act_quant(kv, block_size, self.scale_fmt)
        kv = (kv_fp8.view(-1, block_size).float()
              * kv_scale.view(-1, 1)).to(kv.dtype).view_as(kv)
        self.kv_cache[:bsz, start_pos:end_pos] = kv
        self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)

        # [seqlen, end_pos] causal mask, valid at any start_pos. None in decode.
        cmask = _causal_mask(x, seqlen, start_pos, end_pos)

        if self.dequant_wkv_b is None and self.wkv_b.scale is not None:
            self.dequant_wkv_b = weight_dequant(self.wkv_b.weight, self.wkv_b.scale)
        wkv_b = self.wkv_b.weight if self.dequant_wkv_b is None else self.dequant_wkv_b
        wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)

        q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
        scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos])
                  + torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])
                  ) * self.softmax_scale

        # the indexer needs the SAME [seqlen, end_pos] mask, not [seqlen, seqlen]
        # In-place throughout. Each [bsz, seqlen, H, end_pos] fp32 temporary is
        # 2.1 GB at chunk=1024/end_pos=4096, and the out-of-place form allocated
        # three of them per layer (mask-add, softmax, cast) on top of ~600 GB of
        # resident weights. That is what wedged the first attempt, not the math:
        # a per-layer microbenchmark put the absorbed core at 0.68 s, on par with
        # MHA's 2.66 s for 4x the tokens.
        topk_indices = self.indexer(x, qr, start_pos, freqs_cis, cmask)
        index_mask = torch.full((bsz, seqlen, end_pos), float("-inf"),
                                device=x.device).scatter_(-1, topk_indices, 0)
        if cmask is not None:
            index_mask += cmask
        scores += index_mask.unsqueeze(2)
        del index_mask

        scores = scores.softmax(dim=-1, dtype=torch.float32).to(x.dtype)
        y = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
        y = torch.einsum("bshc,hdc->bshd", y, wkv_b[:, -self.v_head_dim:])
        return self.wo(y.flatten(2))

    def forward_kernel(self, x, start_pos: int, freqs_cis, mask):
        """Same as forward(), but the attention core runs in the AMX kernel.

        Only the scores/softmax/output part is replaced. The kernel consumes
        exactly what the absorbed path already builds:
            Q  = cat(q_nope_absorbed, q_pe)  [bsz, seqlen, H, 576]
            KV = cat(kv_cache, pe_cache)     [bsz, end_pos, 576]
        """
        import time
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen

        _t = time.perf_counter()
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)

        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv = self.kv_norm(kv)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        kv_fp8, kv_scale = act_quant(kv, block_size, self.scale_fmt)
        kv = (kv_fp8.view(-1, block_size).float()
              * kv_scale.view(-1, 1)).to(kv.dtype).view_as(kv)
        self.kv_cache[:bsz, start_pos:end_pos] = kv
        self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
        _tick("qkv_proj", _t)

        cmask = _causal_mask(x, seqlen, start_pos, end_pos)

        _t = time.perf_counter()
        if self.dequant_wkv_b is None and self.wkv_b.scale is not None:
            self.dequant_wkv_b = weight_dequant(self.wkv_b.weight, self.wkv_b.scale)
        wkv_b = self.wkv_b.weight if self.dequant_wkv_b is None else self.dequant_wkv_b
        wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)

        q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
        _tick("absorb_q", _t)

        _t = time.perf_counter()
        topk_indices = self.indexer(x, qr, start_pos, freqs_cis, cmask)
        _tick("indexer", _t)
        _t = time.perf_counter()

        q576 = torch.cat([q_nope, q_pe], dim=-1).to(torch.bfloat16).contiguous()
        # Two kernels, chosen by seqlen, because M means different things:
        #   decode  (seqlen==1): 16 HEADS fill the AMX tile exactly
        #   prefill (seqlen>1):  16 QUERIES fill it, heads loop outside
        # Using the prefill kernel at seqlen==1 leaves 15/16 of every tile empty
        # (measured 2213 ms/token vs 1209 on torch).
        impl = DECODE_IMPL if (seqlen == 1 and DECODE_IMPL is not None) else ATTN_IMPL
        # NOTE: at bsz==1 the slice kv_cache[:1, :end_pos] already reports
        # is_contiguous()==True (a size-1 leading dim makes its stride
        # irrelevant), so .contiguous() was NOT copying and this is not where
        # the 4770 ms TPOT at Lin=120k goes -- that is still undiagnosed. The
        # explicit bsz==1 path is kept only to guarantee no copy regardless of
        # how max_batch is configured; for bsz>1 the copy is genuine.
        if impl is DECODE_IMPL and os.environ.get("DECODE_KV_NOCOPY", "1") == "1":
            # Decode: hand over the whole [:bsz] slab, which is contiguous as a
            # leading-dim slice. The kernel uses Skv only as row stride + bounds
            # (bench_sparse_amx.cpp:688-691) and topk < end_pos, so this is the
            # same math with no copy. The [:bsz, :end_pos].contiguous() below
            # copied bsz x end_pos x 576 bf16 per layer per step (~8.4 GB/seq
            # at 120k) and made TPOT linear in BS (HCTP2, job 27933027).
            kv_in, pe_in = self.kv_cache[:bsz], self.pe_cache[:bsz]
        elif bsz == 1:
            kv_in = self.kv_cache[0, :end_pos].unsqueeze(0)
            pe_in = self.pe_cache[0, :end_pos].unsqueeze(0)
            if not kv_in.is_contiguous():
                kv_in = kv_in.contiguous()
            if not pe_in.is_contiguous():
                pe_in = pe_in.contiguous()
        else:
            kv_in = self.kv_cache[:bsz, :end_pos].contiguous()
            pe_in = self.pe_cache[:bsz, :end_pos].contiguous()
        y = impl(q576, kv_in.to(torch.bfloat16), pe_in.to(torch.bfloat16),
                 topk_indices.to(torch.int32).contiguous(),
                 float(self.softmax_scale))
        _tick("attention", _t)
        _t = time.perf_counter()
        y = torch.einsum("bshc,hdc->bshd", y.to(x.dtype), wkv_b[:, -self.v_head_dim:])
        out = self.wo(y.flatten(2))
        _tick("out_proj", _t)
        return out

    if ATTN_IMPL is not None or DECODE_IMPL is not None:
        def forward_dispatch(self, x, start_pos, freqs_cis, mask):
            # seqlen==1 is decode: the prefill kernel would put ONE real query
            # in a 16-row AMX tile and waste 15/16 of it (measured 2213 ms/token
            # vs 1209 on torch). Decode wants 16 HEADS in those rows instead,
            # which is what bench_sparse_amx.cpp does -- not yet wired here.
            if x.size(1) == 1:
                if DECODE_IMPL is not None:
                    return forward_kernel(self, x, start_pos, freqs_cis, mask)
                return forward(self, x, start_pos, freqs_cis, mask)
            if ATTN_IMPL is not None or DECODE_IMPL is not None:
                return forward_kernel(self, x, start_pos, freqs_cis, mask)
            return forward(self, x, start_pos, freqs_cis, mask)
        MLA.forward = forward_dispatch
    else:
        MLA.forward = forward
    log_fn(f"[chunked-prefill] attention core -- prefill: "
           f"{'AMX' if ATTN_IMPL is not None else 'torch'}, decode: "
           f"{'AMX' if DECODE_IMPL is not None else 'torch'}")
    log_fn("[chunked-prefill] MLA.forward rebound: absorbed path for prefill AND "
           "decode, [seqlen, end_pos] masks, no per-head K/V materialization")
    return {"installed": True, "path": "absorbed", "layers": len(transformer.layers)}
