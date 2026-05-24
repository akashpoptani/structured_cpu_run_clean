"""Synthetic exact-token prompt construction for bench/generate modes.

`build_exact_prompt(tokenizer, target_tokens)` produces a `prompt_text` whose
`tokenizer.encode(text, add_special_tokens=False)` is exactly `target_tokens`
tokens long. This is the bench-friendly equivalent of using a real prompt:
the workload is shaped purely by `LIN_TOKENS` / `LOUT_TOKENS`, independent of
any reference case.

Algorithm:
  - Start from a seed sentence and tokenize it (add_special_tokens=False).
  - Repeat the seed token sequence until at least `target_tokens + slack` ids
    are available.
  - Search candidate prefix lengths in [target_tokens - slack, target_tokens
    + slack]. For each candidate length, decode that prefix with
    skip_special_tokens=False and clean_up_tokenization_spaces=False, then
    re-encode (add_special_tokens=False). Return the first text whose
    round-trip length matches `target_tokens`.
  - Raise RuntimeError if no candidate matches within the slack window.

For BS > 1 we cycle through a SEEDS list so different cases get different
prompt content (still exact-token); for now BS=1 is the only path exercised.

The fox sentence comes from the legacy bench reference. Other seeds add
prompt diversity without changing the exactness guarantee.
"""

from typing import Any, Dict, Iterable, List


SEEDS: List[str] = [
    " The quick brown fox jumps over the lazy dog.",
    " Pack my box with five dozen liquor jugs.",
    " Sphinx of black quartz, judge my vow.",
    " How vexingly quick daft zebras jump!",
    " A wizard's job is to vex chumps quickly in fog.",
    " Crazy Fredrick bought many very exquisite opal jewels.",
    " Jaded zombies acted quaintly but kept driving their oxen forward.",
    " The five boxing wizards jump quickly and silently away.",
]


def encode_len(tokenizer: Any, text: str) -> int:
    """Return the number of tokens after tokenizer.encode(text, add_special_tokens=False)."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def _decode(tokenizer: Any, ids: Iterable[int]) -> str:
    return tokenizer.decode(
        list(ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def build_exact_prompt(
    tokenizer: Any,
    target_tokens: int,
    seed_text: str = SEEDS[0],
    slack: int = 512,
) -> str:
    """Return a prompt string that round-trips to exactly `target_tokens` ids.

    Raises RuntimeError if no candidate within +/- slack of `target_tokens`
    in the repeated-seed stream re-encodes to the target length.
    """
    if target_tokens <= 0:
        raise ValueError(f"target_tokens must be > 0; got {target_tokens}")

    seed_ids = tokenizer.encode(seed_text, add_special_tokens=False)
    if not seed_ids:
        raise RuntimeError(f"seed text encodes to 0 tokens: {seed_text!r}")

    # Make sure we have at least target + slack ids available.
    needed = target_tokens + slack
    reps = (needed + len(seed_ids) - 1) // len(seed_ids)
    pool: List[int] = (seed_ids * max(reps, 1))[: max(needed, len(seed_ids))]

    lo = max(1, target_tokens - slack)
    hi = min(len(pool), target_tokens + slack)

    # Walk the candidate-length window starting from the target itself.
    # The exact match is usually within a handful of decode/encode roundtrips,
    # but tokenizer normalization can shift the round-trip length by 1-2.
    order: List[int] = [target_tokens]
    for d in range(1, slack + 1):
        if target_tokens - d >= lo:
            order.append(target_tokens - d)
        if target_tokens + d <= hi:
            order.append(target_tokens + d)

    for cand_len in order:
        prefix = pool[:cand_len]
        text = _decode(tokenizer, prefix)
        if encode_len(tokenizer, text) == target_tokens:
            return text

    raise RuntimeError(
        f"failed to build an exact-length prompt for target_tokens={target_tokens} "
        f"within slack={slack}; seed={seed_text!r}"
    )


def build_synthetic_cases(
    tokenizer: Any,
    lin_tokens: int,
    lout_tokens: int,
    batch_size: int,
    tag_or_label: str,
) -> List[Dict[str, Any]]:
    """Return one synthetic case per batch slot for the gen/bench paths.

    For BS=1 we use SEEDS[0]. For BS>1 we cycle through SEEDS so different
    cases get different prompt content. Each case carries the full
    `prompt_text`, `lin_tokens`, `lout_tokens`, `batch_size`, and a `source`
    dict explaining how the prompt was constructed. `expected_output_token_ids`
    is intentionally absent (None) — verify mode never reads synthetic cases.
    """
    if lin_tokens <= 0:
        raise ValueError(f"lin_tokens must be > 0; got {lin_tokens}")
    if lout_tokens <= 0:
        raise ValueError(f"lout_tokens must be > 0; got {lout_tokens}")
    if batch_size != 1:
        # Real batching is not yet exercised in the runner; surface clearly.
        raise NotImplementedError(
            f"build_synthetic_cases currently supports batch_size=1; got {batch_size}"
        )

    cases: List[Dict[str, Any]] = []
    for slot in range(batch_size):
        seed = SEEDS[slot % len(SEEDS)]
        prompt_text = build_exact_prompt(tokenizer, lin_tokens, seed_text=seed)
        cases.append({
            "case_id": f"synthetic_lin{lin_tokens}_lout{lout_tokens}_bs{batch_size}_case_{slot + 1:04d}",
            "tag": tag_or_label,
            "description": (
                f"Synthetic exact-token prompt (Lin={lin_tokens}, Lout={lout_tokens}, "
                f"slot={slot + 1}/{batch_size})."
            ),
            "prompt_text": prompt_text,
            "lin_tokens": lin_tokens,
            "lout_tokens": lout_tokens,
            "batch_size": batch_size,
            "sampling": {
                "method": "greedy",
                "temperature": 0.0,
                "min_tokens": lout_tokens,
                "max_tokens": lout_tokens,
                "ignore_eos": True,
            },
            "expected_output_token_ids": None,
            "source": {
                "kind": "synthetic_exact_prompt",
                "seed_text": seed,
                "encoded_prompt_length": encode_len(tokenizer, prompt_text),
            },
        })
    return cases


def main(argv: List[str]) -> int:
    """Tiny CLI:
        python -m src.clean_inference.prompting --tokenizer-dir <dir> --target-tokens 100
    Re-encodes the built prompt and prints the verified length + first 80 chars.
    """
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Exact-token prompt CLI smoke.")
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--seed-index", type=int, default=0)
    args = parser.parse_args(argv)

    from transformers import PreTrainedTokenizerFast  # type: ignore

    tok_file = Path(args.tokenizer_dir) / "tokenizer.json"
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tok_file))
    seed = SEEDS[args.seed_index % len(SEEDS)]
    prompt = build_exact_prompt(tokenizer, args.target_tokens, seed_text=seed)
    n = encode_len(tokenizer, prompt)
    print(f"target={args.target_tokens} actual={n} match={n == args.target_tokens}")
    print(f"seed={seed!r}")
    head = prompt[:80].replace("\n", " ")
    print(f"prompt_head={head!r}")
    return 0 if n == args.target_tokens else 1


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
