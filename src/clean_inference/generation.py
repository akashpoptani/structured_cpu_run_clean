"""Greedy decode for a batch of cases (B >= 1).

Matches the upstream DeepSeek generate.py shape and the legacy verify_cpu.py
loop exactly so token IDs reproduce against the GPU reference.

Loop shape:
    total_len = prompt_len + lout
    tokens = torch.full((B, total_len), -1, dtype=torch.long)
    tokens[:, :prompt_len] = prompt_tokens          # every row the same length
    prev_pos = 0
    for cur_pos in range(prompt_len, total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        next_token = logits.argmax(dim=-1)          # greedy, T=0 -> [B]
        tokens[:, cur_pos] = next_token
        prev_pos = cur_pos
    return tokens[:, prompt_len:].tolist()          # B lists

EQUAL PROMPT LENGTHS ARE REQUIRED, and are free here. `prev_pos` is a single
scalar shared by the whole batch and the slice `tokens[:, prev_pos:cur_pos]` is
uniform, so rows of differing length would desynchronise upstream's KV cache
offsets and RoPE positions. Both prompt sources already emit exactly
LIN_TOKENS per slot -- the synthetic builder and the InfiniteBench corpus are
both exact-length by construction -- so no padding or per-row position
bookkeeping is needed. The function asserts this rather than assuming it.

NO PER-ROW EARLY TERMINATION. Every case carries sampling
`{ignore_eos: True, min_tokens == max_tokens == lout}`, so all rows run for
exactly `lout` steps by design. Tracking per-row EOS would be dead code today;
add it when a case appears that actually stops early.

Why this loop is deliberately bare:
  - This is the native DeepSeek generate.py / legacy verify shape. Re-implementing
    the same shape is the safest path to token-exact reproduction.
  - We intentionally do not manage the KV cache in Python. Upstream
    DeepSeek `model.py` owns the cache and RoPE position via
    `start_pos = prev_pos`; passing `tokens[:, prev_pos:cur_pos]` plus
    `prev_pos` is enough for it to grow the cache by exactly the new
    token(s) and read past keys at the right offsets.
  - For TP2 the forward / logits path handles the distributed parallel
    layers internally — the Python loop is identical to the single-rank
    case.
  - For DP / EP modes later, rank/output coordination may be needed
    (currently DP rank deduplication happens inside MoE.forward), but that
    is out of scope for the first TP2 clean lane.
"""

import sys
import time
from typing import List, Sequence, Tuple, Union

import torch


def _stage_summary(dt_ms: float) -> str:
    """MLA stage times since the last call, as a log suffix; resets them.

    Called after every prefill forward -- including the final slice -- so
    STAGE_MS never carries prefill time into decode. Empty unless
    mla_absorbed is installed (PREFILL_CHUNK > 0).
    """
    import sys as _sys
    _cp = _sys.modules.get("mla_absorbed")
    # kernel.INDEX_MS: the fp8_index part of "indexer", shown as indexer=X(score=Y)
    # and not added to MLA_sum again.
    _kn = _sys.modules.get("kernel")
    idx = getattr(_kn, "INDEX_MS", 0.0)
    if idx:
        _kn.INDEX_MS = 0.0
    # indexer_forward.STAGE_MS: the same "indexer" time split by piece; its
    # idx_score includes the cache-slice copies around the kernel's score time.
    _ix = getattr(_sys.modules.get("indexer_forward"), "STAGE_MS", None) or {}
    sub = " ".join(f"{k[4:]}={v:.0f}" for k, v in _ix.items())
    _ix.clear()
    if _cp is None or not _cp.STAGE_MS:
        return f"  [index_score={idx:.0f}]" if idx else ""
    tot = sum(_cp.STAGE_MS.values())
    parts = " ".join(f"{k}={v:.0f}" + (f"(score={idx:.0f})" if k == "indexer" and idx else "")
                     + (f"[{sub}]" if k == "indexer" and sub else "")
                     for k, v in sorted(_cp.STAGE_MS.items(), key=lambda kv: -kv[1]))
    _cp.STAGE_MS.clear()
    return f"  [{parts} MLA_sum={tot:.0f} other={dt_ms - tot:.0f}]"


def greedy_decode(
    transformer,
    prompt_tokens: Union[List[int], Sequence[Sequence[int]]],
    lout_tokens: int,
    log_fn=print,
    prefill_chunk: int = 0,
) -> Tuple[List[List[int]], float, float]:
    """Run greedy decode for exactly `lout_tokens` steps, for a batch of B rows.

    `prompt_tokens` is a list of per-row token lists; a bare list of ints is
    accepted and treated as B=1. All rows must be the same length.

    Returns (generated_token_ids_per_row, prefill_seconds, decode_seconds_total)
    where the first element is always a list of B lists, even for B=1.
    """
    if lout_tokens <= 0:
        raise ValueError(f"lout_tokens must be > 0; got {lout_tokens}")

    rows: List[List[int]] = (
        [list(prompt_tokens)] if prompt_tokens and isinstance(prompt_tokens[0], int)
        else [list(r) for r in prompt_tokens]
    )
    if not rows:
        raise ValueError("prompt_tokens is empty")
    B = len(rows)
    lengths = {len(r) for r in rows}
    if len(lengths) != 1:
        raise ValueError(
            f"all rows must have the same prompt length; got {sorted(lengths)}. "
            f"prev_pos is shared across the batch, so ragged rows would "
            f"desynchronise the KV cache."
        )
    prompt_len = lengths.pop()
    if prompt_len == 0:
        raise ValueError("prompt length is 0")

    max_bs = getattr(transformer, "max_batch_size", None)
    if max_bs is not None and B > max_bs:
        raise ValueError(
            f"batch={B} exceeds model.max_batch_size={max_bs}; "
            f"BATCH_SIZE must be set before construction"
        )

    total_len = prompt_len + lout_tokens
    max_seq_len = getattr(transformer, "max_seq_len", None)
    if max_seq_len is not None and total_len > max_seq_len:
        raise ValueError(
            f"total_len={total_len} > model.max_seq_len={max_seq_len}; "
            f"increase max_seq_len override before construction"
        )

    tokens = torch.full((B, total_len), -1, dtype=torch.long)
    for b, row in enumerate(rows):
        tokens[b, :prompt_len] = torch.tensor(row, dtype=torch.long)

    prev_pos = 0
    prefill_seconds = 0.0
    decode_seconds_total = 0.0
    start = time.perf_counter()

    # Chunked prefill. Upstream pushes the whole prompt through one forward, so
    # the indexer's index_score and model.py's index_mask are both
    # [bsz, Lin, end_pos] -- 57.6 GB EACH at Lin=120k, B=1. Feeding the prompt in
    # chunks bounds them by chunk size instead. REQUIRES install_mla_absorbed:
    # the stock MHA prefill branch reads only the local chunk and never the kv
    # cache, so chunking without that override silently makes each chunk blind to
    # the ones before it. All but the last chunk are cache-fill only; the final
    # slice falls through to the loop below, which needs the logits.
    if prefill_chunk and prompt_len > prefill_chunk:
        n_chunks = (prompt_len + prefill_chunk - 1) // prefill_chunk
        log_fn(f"[gen] chunked prefill: {prompt_len} tokens in {n_chunks} x {prefill_chunk}")
        t_pf = time.perf_counter()
        while prev_pos + prefill_chunk < prompt_len:
            c0 = time.perf_counter()
            transformer.forward(tokens[:, prev_pos:prev_pos + prefill_chunk], prev_pos)
            prev_pos += prefill_chunk
            _dt = (time.perf_counter() - c0) * 1000
            log_fn(f"[gen]   prefill chunk -> pos {prev_pos}/{prompt_len} "
                   f"in {_dt:.0f} ms{_stage_summary(_dt)}")
        prefill_seconds = time.perf_counter() - t_pf

    for cur_pos in range(prompt_len, total_len):
        is_prefill = cur_pos == prompt_len
        t0 = time.perf_counter()
        sys.stdout.flush()
        logits = transformer.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        next_token = logits.argmax(dim=-1).reshape(B)
        tokens[:, cur_pos] = next_token
        dt = time.perf_counter() - t0
        ids = next_token.tolist()
        shown = ids[0] if B == 1 else ids
        if is_prefill:
            # prefill_seconds, not dt: with chunking, dt is only the final slice
            prefill_seconds += dt
            log_fn(
                f"[gen] PREFILL done in {prefill_seconds*1000:.0f} ms (B={B}, "
                f"Lin={prompt_len}; final slice {dt*1000:.0f} ms"
                f"{_stage_summary(dt * 1000)}); "
                f"first token id={shown}"
            )
        else:
            decode_seconds_total += dt
            tok_idx = cur_pos - prompt_len
            log_fn(
                f"[gen] DECODE token {tok_idx}/{lout_tokens - 1} in "
                f"{dt*1000:.0f} ms; id={shown}"
            )
        prev_pos = cur_pos

    total_seconds = time.perf_counter() - start
    log_fn(
        f"[gen] generated {lout_tokens} tokens x B={B} in {total_seconds:.2f}s "
        f"(prefill={prefill_seconds*1000:.0f} ms, "
        f"decode={decode_seconds_total*1000:.0f} ms, "
        f"{decode_seconds_total*1000/max(lout_tokens-1,1):.0f} ms/step)"
    )
    return tokens[:, prompt_len:].tolist(), prefill_seconds, decode_seconds_total
