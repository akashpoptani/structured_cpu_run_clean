"""Greedy decode for one reference case (batch size 1).

Matches the upstream DeepSeek generate.py shape and the legacy verify_cpu.py
loop exactly so token IDs reproduce against the GPU reference.

Loop shape:
    total_len = len(prompt_tokens) + lout
    tokens = torch.full((1, total_len), -1, dtype=torch.long)
    tokens[0, :len(prompt_tokens)] = prompt_tokens
    prev_pos = 0
    for cur_pos in range(len(prompt_tokens), total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        next_token = logits.argmax(dim=-1)          # greedy, T=0
        tokens[0, cur_pos] = next_token.item()
        prev_pos = cur_pos
    return tokens[0, len(prompt_tokens):].tolist()

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
from typing import List, Tuple

import torch


def greedy_decode(
    transformer,
    prompt_tokens: List[int],
    lout_tokens: int,
    log_fn=print,
) -> Tuple[List[int], float, float]:
    """Run greedy decode for exactly `lout_tokens` steps.

    Returns (generated_token_ids, prefill_seconds, decode_seconds_total).
    """
    if lout_tokens <= 0:
        raise ValueError(f"lout_tokens must be > 0; got {lout_tokens}")

    total_len = len(prompt_tokens) + lout_tokens
    max_seq_len = getattr(transformer, "max_seq_len", None)
    if max_seq_len is not None and total_len > max_seq_len:
        raise ValueError(
            f"total_len={total_len} > model.max_seq_len={max_seq_len}; "
            f"increase max_seq_len override before construction"
        )

    tokens = torch.full((1, total_len), -1, dtype=torch.long)
    for i, t in enumerate(prompt_tokens):
        tokens[0, i] = t

    prev_pos = 0
    prefill_seconds = 0.0
    decode_seconds_total = 0.0
    start = time.perf_counter()

    for cur_pos in range(len(prompt_tokens), total_len):
        is_prefill = cur_pos == len(prompt_tokens)
        t0 = time.perf_counter()
        sys.stdout.flush()
        logits = transformer.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        next_token = logits.argmax(dim=-1)
        tokens[0, cur_pos] = next_token.item()
        dt = time.perf_counter() - t0
        if is_prefill:
            prefill_seconds = dt
            log_fn(
                f"[gen] PREFILL done in {dt*1000:.0f} ms; "
                f"first token id={next_token.item()}"
            )
        else:
            decode_seconds_total += dt
            tok_idx = cur_pos - len(prompt_tokens)
            log_fn(
                f"[gen] DECODE token {tok_idx}/{lout_tokens - 1} in "
                f"{dt*1000:.0f} ms; id={next_token.item()}"
            )
        prev_pos = cur_pos

    total_seconds = time.perf_counter() - start
    log_fn(
        f"[gen] generated {lout_tokens} tokens in {total_seconds:.2f}s "
        f"(prefill={prefill_seconds*1000:.0f} ms, "
        f"decode={decode_seconds_total*1000:.0f} ms)"
    )
    return tokens[0, len(prompt_tokens):].tolist(), prefill_seconds, decode_seconds_total
