#!/usr/bin/env python3
"""Inspect and validate clean-lane verification reference cases."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional


REQUIRED_CASE_FIELDS = (
    "case_id",
    "tag",
    "description",
    "prompt_text",
    "lin_tokens",
    "lout_tokens",
    "batch_size",
    "sampling",
    "expected_output_token_ids",
    "source",
)

REQUIRED_SAMPLING_FIELDS = (
    "method",
    "temperature",
    "min_tokens",
    "max_tokens",
    "ignore_eos",
)


def fail(path: Optional[Path], message: str) -> None:
    if path is None:
        print(f"ERROR: {message}", file=sys.stderr)
    else:
        print(f"ERROR: {path}: {message}", file=sys.stderr)
    raise SystemExit(1)


def require_type(path: Path, data: Dict[str, Any], field: str, expected_type: type) -> None:
    if not isinstance(data[field], expected_type):
        fail(path, f"{field} must be {expected_type.__name__}")


def validate_case(path: Path, data: Any) -> Dict[str, Any]:
    if not isinstance(data, dict):
        fail(path, "case JSON root must be an object")

    for field in REQUIRED_CASE_FIELDS:
        if field not in data:
            fail(path, f"missing required field: {field}")

    require_type(path, data, "case_id", str)
    require_type(path, data, "tag", str)
    require_type(path, data, "description", str)
    require_type(path, data, "prompt_text", str)
    require_type(path, data, "lin_tokens", int)
    require_type(path, data, "lout_tokens", int)
    require_type(path, data, "batch_size", int)
    require_type(path, data, "expected_output_token_ids", list)
    require_type(path, data, "sampling", dict)
    require_type(path, data, "source", dict)

    sampling = data["sampling"]
    for field in REQUIRED_SAMPLING_FIELDS:
        if field not in sampling:
            fail(path, f"missing required sampling field: {field}")

    if data["lin_tokens"] <= 0:
        fail(path, "lin_tokens must be > 0")
    if data["lout_tokens"] <= 0:
        fail(path, "lout_tokens must be > 0")
    if data["batch_size"] <= 0:
        fail(path, "batch_size must be > 0")

    output_tokens = data["expected_output_token_ids"]
    if not all(isinstance(token, int) for token in output_tokens):
        fail(path, "expected_output_token_ids must be a list of ints")

    if data["batch_size"] == 1 and len(output_tokens) != data["lout_tokens"]:
        fail(path, "len(expected_output_token_ids) must equal lout_tokens for batch_size == 1")

    max_tokens = sampling["max_tokens"]
    min_tokens = sampling["min_tokens"]
    if isinstance(max_tokens, int) and max_tokens != data["lout_tokens"]:
        fail(path, "sampling.max_tokens must equal lout_tokens")
    if isinstance(min_tokens, int) and min_tokens != data["lout_tokens"]:
        fail(path, "sampling.min_tokens must equal lout_tokens")

    return data


def load_case(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        fail(path, f"invalid JSON: {exc}")
    except OSError as exc:
        fail(path, f"could not read file: {exc}")

    return validate_case(path, data)


def inspect_references(reference_root: Path) -> Dict[str, Any]:
    if not reference_root.exists():
        fail(None, f"reference root does not exist: {reference_root}")
    if not reference_root.is_dir():
        fail(None, f"reference root is not a directory: {reference_root}")

    groups = []
    case_count = 0

    for group_dir in sorted(path for path in reference_root.iterdir() if path.is_dir()):
        case_paths = sorted(group_dir.glob("*.json"))
        if not case_paths:
            continue

        cases = []
        for case_path in case_paths:
            case = load_case(case_path)
            cases.append(
                {
                    "path": str(case_path),
                    "case_id": case["case_id"],
                    "tag": case["tag"],
                    "description": case["description"],
                    "prompt_text": case["prompt_text"],
                    "lin_tokens": case["lin_tokens"],
                    "lout_tokens": case["lout_tokens"],
                    "batch_size": case["batch_size"],
                    "sampling": case["sampling"],
                    "expected_output_token_count": len(case["expected_output_token_ids"]),
                }
            )

        case_count += len(cases)
        groups.append(
            {
                "group": group_dir.name,
                "case_count": len(cases),
                "cases": cases,
            }
        )

    return {
        "reference_root": str(reference_root),
        "group_count": len(groups),
        "case_count": case_count,
        "groups": groups,
    }


def print_human(summary: Dict[str, Any]) -> None:
    print("Verification reference summary")
    print("------------------------------")
    print(f"Reference root: {summary['reference_root']}")
    print(f"Groups: {summary['group_count']}")
    print(f"Cases: {summary['case_count']}")
    print()

    for group in summary["groups"]:
        print(f"Group: {group['group']}")
        for case in group["cases"]:
            sampling = case["sampling"]
            ignore_eos = str(sampling["ignore_eos"]).lower()
            print(f"  {Path(case['path']).name}")
            print(f"    case_id: {case['case_id']}")
            print(f"    tag: {case['tag']}")
            print(f"    prompt chars: {len(case['prompt_text'])}")
            print(
                "    lin/lout/batch: "
                f"{case['lin_tokens']} / {case['lout_tokens']} / {case['batch_size']}"
            )
            print(f"    expected output tokens: {case['expected_output_token_count']}")
            print(
                "    sampling: "
                f"{sampling['method']}, temperature={sampling['temperature']}, "
                f"min=max={sampling['max_tokens']}, ignore_eos={ignore_eos}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect clean verification reference cases.")
    parser.add_argument("--reference-root", required=True, help="Reference root directory.")
    parser.add_argument("--format", choices=("human", "json"), default="human")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = inspect_references(Path(args.reference_root))

    if args.format == "json":
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_human(summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
