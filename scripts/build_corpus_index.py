"""Tokenize the InfiniteBench corpus once and record exact per-record counts.

Why this exists: every character-based estimate of "is this record long enough"
has been wrong, in both directions.

  * a global floor of 5.0 chars/token silently deleted all of longdialogue_qa_eng
  * a global floor of 3.2 silently deletes Chinese records (longbook_qa_chn runs
    1.36-1.77 chars/token, so it needs far fewer characters than the floor demands)
  * single-sample per-task ratios gave wrong eligibility counts, because
    code_debug alone varies 1.68x internally (2.68-4.51 chars/token)

No single number -- global or per-task -- can answer "does this record have at
least N tokens". So we answer it exactly, once, and cache the result.

Output: <corpus_dir>/index.json

    {"cap_chars": 700000,
     "tokenizer_dir": "...",
     "tasks": {"<task>": [{"i": 0, "chars": 632149, "tokens": 134221,
                           "truncated": false}, ...]}}

`tokens` is the token count of the first `cap_chars` characters. The cap bounds
the work per record; it is set so that even the least dense task observed
(4.74 chars/token) still yields more than the model's 131072-token context
limit, so `tokens >= LIN_TOKENS` is an EXACT test for any usable LIN_TOKENS.
`truncated` records whether the cap was hit, i.e. whether the true count is
higher than reported -- which never changes an eligibility answer, only the
reported magnitude.

Run once from a login node after scripts/fetch_infinitebench.sh. Re-run after
adding a task; existing entries are reused unless --force is given.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.clean_inference.corpus import TASKS, iter_records  # noqa: E402
from src.clean_inference.tokenization import load_tokenizer  # noqa: E402

# 131072 (model context) x 4.74 (densest observed chars/token) = 621k; round up.
CAP_CHARS = 700_000
INDEX_NAME = "index.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--force", action="store_true", help="re-tokenize tasks already indexed")
    args = ap.parse_args(argv)

    corpus_dir = Path(args.corpus_dir).resolve()
    out_path = corpus_dir / INDEX_NAME

    tokenizer, _ = load_tokenizer(
        {"CLEAN_ROOT": ".", "TOKENIZER_PATH": args.tokenizer_dir,
         "SHARDED_CKPT_PATH": "", "ACTIVE_MODEL_PATH": ""},
        log_fn=lambda m: None,
    )

    index = {"cap_chars": CAP_CHARS, "tokenizer_dir": args.tokenizer_dir, "tasks": {}}
    if out_path.is_file() and not args.force:
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            if prev.get("cap_chars") == CAP_CHARS:
                index["tasks"] = prev.get("tasks", {})
        except (OSError, json.JSONDecodeError):
            pass

    for task in TASKS:
        path = corpus_dir / f"{task}.jsonl"
        if not path.is_file():
            print(f"[index] skip   {task:22s} (not on disk)")
            continue
        if task in index["tasks"] and not args.force:
            print(f"[index] have   {task:22s} {len(index['tasks'][task])} records")
            continue

        t0 = time.time()
        rows = []
        for i, rec in iter_records(str(path)):
            ctx = rec.get("context", "")
            head = ctx[:CAP_CHARS]
            n = len(tokenizer.encode(head, add_special_tokens=False))
            rows.append({"i": i, "chars": len(ctx), "tokens": n,
                         "truncated": len(ctx) > CAP_CHARS})
        index["tasks"][task] = rows
        ge = sum(1 for r in rows if r["tokens"] >= 120000)
        print(f"[index] built  {task:22s} {len(rows):4d} records, "
              f"{ge:4d} >=120k tokens, {time.time()-t0:6.1f}s")
        out_path.write_text(json.dumps(index), encoding="utf-8")  # checkpoint

    out_path.write_text(json.dumps(index), encoding="utf-8")
    total = sum(len(v) for v in index["tasks"].values())
    ge = sum(1 for v in index["tasks"].values() for r in v if r["tokens"] >= 120000)
    print(f"\n[index] wrote {out_path}  ({total} records, {ge} >=120k tokens)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
