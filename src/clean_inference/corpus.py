"""InfiniteBench long-context prompt corpus.

Replaces the repeated-pangram seed in `prompting.py` for anything whose result
depends on prompt *content* -- principally MoE expert routing, where repeating
one sentence makes the same handful of experts fire for every token.

Why this corpus, at LIN_TOKENS=120k: chat corpora are the wrong shape. ShareGPT
averages ~80 input tokens, so 120k means concatenating ~1500 unrelated
conversations, manufacturing a semantic transition every ~80 tokens; the
measured expert distribution would then describe the concatenation policy, not a
workload. LMSYS-Chat-1M is longer but still chat-scale, and is HF-gated.
InfiniteBench records are single coherent documents already well past 100k
tokens, so a 120k prompt is one real document truncated. It is ungated.

Record schema (all tasks): {id:int, context:str, input:str, answer:list,
options:list}. `context` is the long document; `input` is the short question.
We use `context` alone -- the task question is irrelevant to a throughput or
routing measurement and would only add a few tokens.

Only natural-language tasks are used. The synthetic ones are excluded on
purpose: `math_find`'s context is a bare list of integers, about as degenerate
for routing as the pangram this replaces.

BATCHING (not yet wired -- BS=1 is the only exercised path today):
`select_records` already takes `n_slots` and assigns slot i a distinct
(task, record) pair, round-robining across TASK FILES first so that slots differ
by *domain* (prose / code / dialogue) before they differ by record. That is the
property a batch needs for routing diversity, and it is why selection is written
slot-wise now rather than being retrofitted later.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

# Natural-language tasks only, ordered so that consecutive batch slots differ by
# domain. Synthetic tasks (math_find, number_string, passkey, kv_retrieval) are
# deliberately absent.
# Deliberate UNDER-estimate of chars/token, used only as a pre-filter bound. The
# measured minimum across these tasks is 3.78 (longdialogue_qa_eng); anything at
# or above that silently drops usable records, and at 120k tokens a bound of 5.0
# drops longdialogue entirely even though its longest records reach ~131k tokens.
# Used ONLY to bound how much of a record gets tokenized, never for
# eligibility -- eligibility comes from the exact counts in index.json. Set
# above the densest observed ratio (4.74) so the first slice normally suffices;
# build_exact_prompt_from_corpus widens and retries if it does not, so a wrong
# value costs time, never correctness. That is the whole point: the three bugs
# this corpus has had all came from a character estimate deciding eligibility.
CHARS_PER_TOKEN_CEIL = 5.0

# Order matters: consecutive batch slots take consecutive entries, so this is
# interleaved by domain (prose, code, prose, dialogue, ...) rather than grouped,
# which keeps a small batch from drawing every slot from the same domain.
TASKS: List[str] = [
    "longbook_sum_eng",     # literary prose (English)
    "code_debug",           # Python source
    "longbook_choice_eng",  # literary prose (English) -- longest records
    "longdialogue_qa_eng",  # screenplay dialogue
    "longbook_qa_chn",      # literary prose (Chinese) -- distinct tokenization
    "longbook_qa_eng",      # literary prose (English) + QA
    "code_run",             # generated Python -- templated, but real syntax
]


INDEX_NAME = "index.json"

# Committed copy of the index, so a fresh clone can use the corpus as soon as
# the .jsonl files are fetched, without the ~15 min indexing pass. It is valid
# because InfiniteBench is a fixed published dataset: record N of a task file is
# the same text for everyone, so the token counts are reproducible rather than
# machine-specific. A locally built index still wins if one is present -- that
# is the one to trust if you ever point PROMPT_CORPUS_PATH at a modified corpus.
VENDORED_INDEX = (
    Path(__file__).resolve().parents[2] / "verification" / "corpus"
    / "infinitebench_index.json"
)


def load_index(corpus_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """Exact per-record token counts, local index preferred over the vendored one."""
    local = os.path.join(corpus_dir, INDEX_NAME)
    path = local if os.path.isfile(local) else str(VENDORED_INDEX)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"no {INDEX_NAME} in {corpus_dir} and no vendored index at "
            f"{VENDORED_INDEX}; run:\n"
            f"  .venv/bin/python scripts/build_corpus_index.py "
            f"--corpus-dir {corpus_dir} --tokenizer-dir <tokenizer dir>"
        )
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh).get("tasks", {})


def _seeded_order(n: int, seed: int, salt: str) -> List[int]:
    """Deterministic permutation of range(n) from (seed, salt).

    Uses blake2b rather than `random` so ordering never depends on interpreter
    version or on any other code having touched the global RNG.
    """
    keyed = [
        (
            hashlib.blake2b(f"{salt}:{seed}:{i}".encode(), digest_size=8).digest(),
            i,
        )
        for i in range(n)
    ]
    keyed.sort()
    return [i for _, i in keyed]


def iter_records(path: str) -> Iterator[Tuple[int, Dict[str, Any]]]:
    """Yield (line_index, record) from one task .jsonl.

    Records are hundreds of KB to ~1.7 MB each, so this streams line by line and
    never holds the file in memory.
    """
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if line:
                yield i, json.loads(line)


def available_tasks(corpus_dir: str) -> List[str]:
    """TASKS present on disk, in TASKS order."""
    return [t for t in TASKS if os.path.isfile(os.path.join(corpus_dir, f"{t}.jsonl"))]


def select_records(
    corpus_dir: str,
    n_slots: int = 1,
    seed: int = 0,
    min_tokens: int = 0,
) -> List[Tuple[str, int]]:
    """Pick one (task, record_index) per batch slot, deterministically.

    Round-robins across task files first, so slot 0 and slot 1 come from
    different domains before any task repeats. Within a task, records are
    visited in a seeded permutation, so repeated runs at the same seed are
    identical while a new seed gives a genuinely different draw.

    Eligibility is `tokens >= min_tokens`, read from the cached index -- an
    EXACT test, not a character estimate. Three character-based filters have
    been wrong here (a 5.0 floor deleted longdialogue_qa_eng, a 3.2 floor
    deletes Chinese, and single-sample per-task ratios miscounted code_debug,
    which varies 1.68x internally), so no estimate is used anywhere.

    The permutation is computed over ALL of a task's records and filtered
    afterwards. That ordering therefore does not depend on `min_tokens`, so a
    sweep over LIN_TOKENS keeps slot i on the same document wherever that
    document still qualifies -- otherwise length and content would vary
    together and neither could be attributed.

    Round-robin walks the task files in TASKS order and DROPS a task once its
    eligible records are used up, rather than wrapping. A batch therefore never
    contains the same document twice.
    """
    index = load_index(corpus_dir)
    tasks = [t for t in TASKS if t in index]
    if not tasks:
        raise FileNotFoundError(f"index in {corpus_dir} lists none of TASKS")

    pools: Dict[str, List[int]] = {}
    for t in tasks:
        rows = index[t]
        # permute over everything, then filter -> order is min_tokens-independent
        order = _seeded_order(len(rows), seed, t)
        eligible = [rows[k]["i"] for k in order if rows[k]["tokens"] >= min_tokens]
        if eligible:
            pools[t] = eligible

    if not pools:
        raise RuntimeError(
            f"no record in {corpus_dir} reaches {min_tokens} tokens"
        )

    out: List[Tuple[str, int]] = []
    cursor = {t: 0 for t in pools}
    live = [t for t in tasks if t in pools]
    while len(out) < n_slots and live:
        for t in list(live):
            if len(out) >= n_slots:
                break
            if cursor[t] >= len(pools[t]):
                live.remove(t)       # exhausted: drop it, never repeat a record
                continue
            out.append((t, pools[t][cursor[t]]))
            cursor[t] += 1
    if len(out) < n_slots:
        raise RuntimeError(
            f"corpus has only {len(out)} distinct records with >= {min_tokens} "
            f"tokens; cannot fill {n_slots} batch slots"
        )
    return out


def load_context(corpus_dir: str, task: str, record_index: int) -> str:
    """Return the `context` string of one record."""
    path = os.path.join(corpus_dir, f"{task}.jsonl")
    for i, rec in iter_records(path):
        if i == record_index:
            return rec.get("context", "")
    raise IndexError(f"record {record_index} not found in {path}")


def build_exact_prompt_from_corpus(
    tokenizer: Any,
    target_tokens: int,
    corpus_dir: str,
    slot: int = 0,
    n_slots: int = 1,
    seed: int = 0,
    slack: int = 64,
    attempts: int = 6,
    task: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Exact-length prompt for one batch slot, plus provenance.

    Mirrors `prompting.build_exact_prompt`'s contract -- the returned text
    re-encodes to exactly `target_tokens` ids -- but draws from a real document
    instead of a repeated seed.

    Tokenizing a 400k-token document to get 120k is wasteful, so the context is
    pre-trimmed by characters first; the round-trip walk then only runs over the
    trimmed text. `slack` is small for the same reason: each probe decodes and
    re-encodes ~120k tokens, so a wide window is expensive, and a real document
    normally lands within a couple of ids.
    """
    from .prompting import encode_len, _decode  # local import: avoids a cycle

    # ~4 chars/token is a deliberate over-estimate so the trim never starves the
    # target; the token-level walk below is what actually enforces exactness.
    # The ONLY corpus-side dependency is target_tokens: a record is eligible iff
    # its exact cached count reaches it. No character heuristics, and therefore
    # no fallbacks needed for "turned out too short" -- that cannot happen now.
    picks = select_records(
        corpus_dir, n_slots=n_slots * attempts, seed=seed, min_tokens=target_tokens
    )
    candidates = [picks[i] for i in range(slot % n_slots, len(picks), n_slots)]
    if task is not None:
        # Pin the domain (e.g. a homogeneous batch, or prose-vs-code at identical
        # Lin). Rotate by slot so slots still get DISTINCT records -- without the
        # rotation every slot takes the same first candidate and a batch becomes
        # N copies of one prompt, which measures nothing.
        pinned = [c for c in picks if c[0] == task]
        if pinned:
            off = slot % len(pinned)
            candidates = pinned[off:] + pinned[:off]

    # `chosen` deliberately differs from the `task` parameter: the parameter is
    # the pin request (may be None), `chosen` is what was actually used.
    chosen = rec_idx = context = None
    ids: List[int] = []
    skipped: List[str] = []
    for chosen, rec_idx in candidates:
        context = load_context(corpus_dir, chosen, rec_idx)
        # Tokenize only what we could need. The index guarantees the FULL record
        # reaches target_tokens, but the slice might not if this record is denser
        # than CHARS_PER_TOKEN_CEIL, so widen and retry rather than trust it.
        for mult in (CHARS_PER_TOKEN_CEIL, 2 * CHARS_PER_TOKEN_CEIL, None):
            head = context if mult is None else context[: int(target_tokens * mult) + 8192]
            ids = tokenizer.encode(head, add_special_tokens=False)
            if len(ids) >= target_tokens:
                break
        if len(ids) >= target_tokens:
            break
        skipped.append(f"{chosen}[{rec_idx}]={len(ids)}tok")
    else:
        raise RuntimeError(
            f"no candidate for slot {slot} reached target_tokens={target_tokens}; "
            f"tried {skipped}"
        )

    order: List[int] = [target_tokens]
    for d in range(1, slack + 1):
        if target_tokens - d >= 1:
            order.append(target_tokens - d)
        if target_tokens + d <= len(ids):
            order.append(target_tokens + d)

    for cand_len in order:
        text = _decode(tokenizer, ids[:cand_len])
        if encode_len(tokenizer, text) == target_tokens:
            return text, {
                "source": "infinitebench",
                "task": chosen,
                "record_index": rec_idx,
                "slot": slot,
                "seed": seed,
                "context_chars": len(context),
                "context_tokens": len(ids),
                "skipped_short": skipped,
                "trim_ids": cand_len,
            }

    raise RuntimeError(
        f"failed to hit target_tokens={target_tokens} from {chosen}[{rec_idx}] "
        f"within slack={slack}"
    )
