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


def greedy_decode(
    transformer,
    prompt_tokens: Union[List[int], Sequence[Sequence[int]]],
    lout_tokens: int,
    log_fn=print,
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
            prefill_seconds = dt
            log_fn(
                f"[gen] PREFILL done in {dt*1000:.0f} ms (B={B}, "
                f"Lin={prompt_len}); first token id={shown}"
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
