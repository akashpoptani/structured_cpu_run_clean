#!/usr/bin/env python3
"""Clean verification runner plumbing with mock inference output."""

import argparse
import hashlib
import json
import random
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from inspect_reference_cases import load_case


REQUIRED_CONFIG_KEYS = (
    "TAG",
    "EXPERIMENT_ID",
    "RUN_MODE",
    "RUN_LABEL",
    "CLEAN_ROOT",
    "ACTIVE_MODEL_PATH",
    "GPU_REFERENCE_PATH",
    "OUTPUT_ROOT",
    "LIN_TOKENS",
    "LOUT_TOKENS",
    "BATCH_SIZE",
    "WEIGHTS_PRECISION",
    "KV_CACHE_DTYPE",
    "SHARDING_MODE",
    "DP_SIZE",
    "TP_SIZE",
    "EP_SIZE",
    "PP_SIZE",
)


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def parse_resolved_env(path: Path) -> Dict[str, str]:
    if not path.exists():
        fail(f"resolved config does not exist: {path}")
    if not path.is_file():
        fail(f"resolved config is not a file: {path}")

    config: Dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        fail(f"could not read resolved config {path}: {exc}")

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            fail(f"{path}:{line_number}: expected KEY=VALUE")

        key, value_part = line.split("=", 1)
        key = key.strip()
        value_part = value_part.strip()
        if not key:
            fail(f"{path}:{line_number}: empty key")

        if value_part == "":
            value = ""
        else:
            try:
                parsed = shlex.split(value_part)
            except ValueError as exc:
                fail(f"{path}:{line_number}: could not parse value for {key}: {exc}")
            value = parsed[0] if parsed else ""

        config[key] = value

    for key in REQUIRED_CONFIG_KEYS:
        if key not in config or config[key] == "":
            fail(f"resolved config missing required key: {key}")

    return config


def resolve_path(clean_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return clean_root / path


def load_reference_cases(reference_root: Path) -> List[Dict[str, Any]]:
    if not reference_root.exists():
        fail(f"reference root does not exist: {reference_root}")
    if not reference_root.is_dir():
        fail(f"reference root is not a directory: {reference_root}")

    cases: List[Dict[str, Any]] = []
    for group_dir in sorted(path for path in reference_root.iterdir() if path.is_dir()):
        case_paths = sorted(group_dir.glob("*.json"))
        if not case_paths:
            continue
        for case_path in case_paths:
            case = load_case(case_path)
            cases.append(
                {
                    "group": group_dir.name,
                    "path": str(case_path),
                    "data": case,
                }
            )

    return cases


def random_tokens(group: str, case_id: str, count: int) -> List[int]:
    seed_bytes = hashlib.sha256(f"{group}/{case_id}".encode("utf-8")).digest()
    seed = int.from_bytes(seed_bytes[:8], byteorder="big")
    rng = random.Random(seed)
    return [rng.randint(0, 200000) for _ in range(count)]


def first_mismatch(expected: List[int], generated: List[int]) -> Optional[int]:
    for index, (expected_token, generated_token) in enumerate(zip(expected, generated)):
        if expected_token != generated_token:
            return index
    if len(expected) != len(generated):
        return min(len(expected), len(generated))
    return None


def build_result(config: Dict[str, str], mock_mode: str) -> Dict[str, Any]:
    clean_root = Path(config["CLEAN_ROOT"])
    reference_root = resolve_path(clean_root, config["GPU_REFERENCE_PATH"])
    reference_cases = load_reference_cases(reference_root)

    case_results = []
    passed_count = 0
    failed_count = 0

    for entry in reference_cases:
        group = entry["group"]
        case_path = entry["path"]
        case = entry["data"]
        expected = case["expected_output_token_ids"]

        if mock_mode == "golden":
            generated = list(expected)
        else:
            generated = random_tokens(group, case["case_id"], case["lout_tokens"])

        mismatch = first_mismatch(expected, generated)
        passed = mismatch is None
        if passed:
            passed_count += 1
        else:
            failed_count += 1

        case_results.append(
            {
                "group": group,
                "case_id": case["case_id"],
                "path": case_path,
                "lin_tokens": case["lin_tokens"],
                "lout_tokens": case["lout_tokens"],
                "batch_size": case["batch_size"],
                "expected_output_token_count": len(expected),
                "generated_output_token_count": len(generated),
                "passed": passed,
                "first_mismatch_index": mismatch,
                "expected_prefix": expected[:10],
                "generated_prefix": generated[:10],
            }
        )

    return {
        "tag": config["TAG"],
        "experiment_id": config["EXPERIMENT_ID"],
        "run_mode": config["RUN_MODE"],
        "mock_mode": mock_mode,
        "active_model_path": config["ACTIVE_MODEL_PATH"],
        "reference_root": str(reference_root),
        "case_count": len(case_results),
        "passed": passed_count,
        "failed": failed_count,
        "cases": case_results,
    }


def write_result(config: Dict[str, str], result: Dict[str, Any]) -> Path:
    clean_root = Path(config["CLEAN_ROOT"])
    output_root = resolve_path(clean_root, config["OUTPUT_ROOT"])
    result_path = output_root / "results" / config["TAG"] / "verify_results.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result_path


def print_human(result: Dict[str, Any], result_path: Path) -> None:
    print("Clean verification runner")
    print("-------------------------")
    print(f"TAG: {result['tag']}")
    print(f"RUN_MODE: {result['run_mode']}")
    print(f"Mock mode: {result['mock_mode']}")
    print(f"Reference root: {result['reference_root']}")
    print(f"Cases: {result['case_count']}")
    print(f"Passed: {result['passed']}")
    print(f"Failed: {result['failed']}")
    print(f"Result JSON: {result_path}")
    print()

    for case in result["cases"]:
        status = "PASS" if case["passed"] else "FAIL"
        print(f"{case['group']}/{case['case_id']}: {status}")
        if not case["passed"]:
            print(f"  first mismatch index: {case['first_mismatch_index']}")
        print(f"  expected prefix: {case['expected_prefix']}")
        print(f"  generated prefix: {case['generated_prefix']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run clean verification plumbing with mock output.")
    parser.add_argument("--resolved-config", required=True, help="Resolved config env file.")
    parser.add_argument("--format", choices=("human", "json"), default="human")
    parser.add_argument("--mock-mode", choices=("random", "golden"), default="random")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = parse_resolved_env(Path(args.resolved_config))
    result = build_result(config, args.mock_mode)
    result_path = write_result(config, result)

    if args.format == "json":
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print_human(result, result_path)

    if args.mock_mode == "golden" and result["failed"] != 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
