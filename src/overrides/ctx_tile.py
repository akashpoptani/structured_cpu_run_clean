"""Emulate a long KV cache without paying for a long prefill.

Why this exists
---------------
TPOT at Lin=120k is the number we want across a batch sweep. Prefill cost
scales with BS x Lin, and a real 120k prefill is ~65 min at BS=1, so the sweep
BS=1,2,4,8,16,32 would need ~60 h of prefill to produce ~30 min of the decode
measurements we actually care about.

What this does instead: really prefill a SHORT prompt (a few thousand tokens),
then tile that cache content across the rest of the window to the target
length, and decode from there. The decode step then sees a cache of the right
SHAPE, which is what sets its cost:

  * indexer  -- fp8_index scores every position in [0, end_pos); cost is driven
                by end_pos, not by the values.
  * gather   -- always the top-2048 rows, independent of context.
  * attention-- over those 2048 gathered rows.

The values are tiled from a REAL prefill rather than randomised because MoE
routing (top-8 of 256) keys off the hidden state: noise would push routing
off-distribution and change the expert load, which does move TPOT.

What this does NOT reproduce
----------------------------
The decode attends to repeated content, so the indexer's top-2048 selection is
drawn from a degenerate set. The COST is unaffected (still 2048 rows) but the
selected positions are not what a real 120k prompt would select, and generated
text is meaningless. This measures TPOT, not correctness.

Validity gate: at BS=1/Lin=120k this must reproduce the real-prefill TPOT
(4770 ms, job 27930739, n=228 sd 79). If it does not, the tiling is not a
faithful stand-in and the matrix built on it is void.
"""
import time
import torch

CACHES = ("kv_cache", "pe_cache", "k_cache", "k_scale_cache")


def tile_to_context(transformer, bsz: int, src_len: int, target_len: int, log_fn=print) -> int:
    """Replicate cache[:, :src_len] forward until target_len is filled."""
    n = 0
    with torch.no_grad():
        for mod in transformer.modules():
            for name in CACHES:
                c = getattr(mod, name, None)
                if not torch.is_tensor(c) or c.ndim < 2 or c.shape[1] < target_len:
                    continue
                src = c[:bsz, :src_len].clone()
                pos = src_len
                while pos < target_len:
                    w = min(src_len, target_len - pos)
                    c[:bsz, pos:pos + w] = src[:, :w]
                    pos += w
                n += 1
    log_fn(f"[ctx-tile] filled {n} cache tensors to pos {target_len} "
           f"from a real {src_len}-token prefill (B={bsz})")
    return n


def _reset_stage_timers():
    """Zero every stage accumulator so the breakdown is DECODE-ONLY.

    native_run resets these before greedy_decode, i.e. before prefill, then
    divides by decode steps -- so the printed "ms per decode step" silently
    includes the prefill call. At Lin=31744 that one prefill indexer call
    dwarfs all 63 decode calls. Resetting here, after prefill and before the
    timed loop, removes the ambiguity instead of forcing a model of what
    fraction came from where.
    """
    out = {}
    try:
        import chunked_prefill as _cp
        _cp.STAGE_MS.clear(); out["stage"] = _cp
    except Exception:
        pass
    try:
        import kernel as _kn
        _kn.reset_index_timer(); out["idx"] = _kn
    except Exception:
        pass
    try:
        import flash_attn_cpu as _fa
        _fa.reset_timings(); out["attn"] = _fa
    except Exception:
        pass
    return out


def _report_stages(mods, steps: int, log_fn):
    cp = mods.get("stage")
    if cp is not None and cp.STAGE_MS:
        log_fn("[ctx-tile] DECODE-ONLY stage breakdown, ms per step:")
        for k, v in sorted(cp.STAGE_MS.items(), key=lambda kv: -kv[1]):
            log_fn(f"[ctx-tile]   {k:12s} {v / steps:9.2f}")
        log_fn(f"[ctx-tile]   {'SUM':12s} {sum(cp.STAGE_MS.values()) / steps:9.2f}")
    kn = mods.get("idx")
    if kn is not None:
        it = kn.get_index_timer()
        if it.get("calls"):
            log_fn(f"[ctx-tile]   {'indexer_k':12s} {it['seconds'] / steps * 1e3:9.2f}"
                   f"   ({int(it['calls'])} calls, decode only)")


def timed_decode(transformer, tokens, start_pos: int, steps: int, log_fn=print):
    """Time `steps` single-token decode forwards starting at start_pos.

    Under dp2_true the caller must pass only this rank's rows (native_run's
    SYNTH_CTX path slices tokens[rank::world] before prefill). Before that
    slice existed the full batch went through every rank: job 27933614 died at
    BS=4 with "expanded size of the tensor (2) must match the existing size
    (4)", and at BS=2 the shapes lined up by luck and the number did NOT
    measure true DP.
    """
    _mods = _reset_stage_timers()
    ms = []
    cur = tokens[:, :1].clone()
    with torch.no_grad():
        for i in range(steps):
            t0 = time.perf_counter()
            logits = transformer.forward(cur, start_pos + i)
            dt = (time.perf_counter() - t0) * 1000.0
            ms.append(dt)
            cur = logits.argmax(dim=-1, keepdim=True).to(tokens.dtype).reshape(-1, 1)
            if i % 8 == 0 or i == steps - 1:
                log_fn(f"[ctx-tile] decode step {i + 1}/{steps} in {dt:.0f} ms")
    _report_stages(_mods, steps, log_fn)
    return ms
