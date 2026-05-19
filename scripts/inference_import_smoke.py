#!/usr/bin/env python3
"""Smoke test clean override import ordering for DeepSeek inference modules."""

import argparse
import importlib
import sys
from pathlib import Path
from typing import Dict

from run_verify import parse_resolved_env, resolve_path


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def require_dir(path: Path, label: str) -> None:
    if not path.exists():
        fail(f"{label} does not exist: {path}")
    if not path.is_dir():
        fail(f"{label} is not a directory: {path}")


def is_within(path: Path, root: Path) -> bool:
    path_resolved = path.resolve()
    root_resolved = root.resolve()
    return path_resolved == root_resolved or root_resolved in path_resolved.parents


def import_module(name: str):
    try:
        return importlib.import_module(name)
    except Exception as exc:
        fail(f"failed to import {name}: {exc}")


def print_symbol_status(model_module) -> None:
    symbol_options = (
        ("Transformer", ("Transformer",)),
        ("ModelArgs", ("ModelArgs",)),
        ("TransformerBlock or Block", ("TransformerBlock", "Block")),
        ("MLA", ("MLA",)),
        ("MoE", ("MoE",)),
    )

    for label, names in symbol_options:
        found = any(hasattr(model_module, name) for name in names)
        print(f"{label}: {'yes' if found else 'no'}")


def run_smoke(resolved_config_path: Path) -> None:
    config: Dict[str, str] = parse_resolved_env(resolved_config_path)
    clean_root = Path(config["CLEAN_ROOT"]).resolve()
    deepseek_repo = resolve_path(clean_root, config["DEEPSEEK_REPO"]).resolve()
    clean_overrides = clean_root / "src" / "overrides"
    deepseek_inference = deepseek_repo / "inference"

    require_dir(clean_overrides, "CLEAN_OVERRIDES")
    require_dir(deepseek_inference, "DEEPSEEK_INFERENCE")

    sys.path.insert(0, str(deepseek_inference))
    sys.path.insert(0, str(clean_overrides))

    kernel = import_module("kernel")
    fast_hadamard_transform = import_module("fast_hadamard_transform")
    model = import_module("model")

    kernel_file = Path(kernel.__file__).resolve()
    hadamard_file = Path(fast_hadamard_transform.__file__).resolve()
    model_file = Path(model.__file__).resolve()

    if not is_within(kernel_file, clean_overrides):
        fail(f"kernel imported from wrong path: {kernel_file}")
    if not is_within(hadamard_file, clean_overrides):
        fail(f"fast_hadamard_transform imported from wrong path: {hadamard_file}")
    if not is_within(model_file, deepseek_inference):
        fail(f"model imported from wrong path: {model_file}")

    print("Inference import smoke")
    print("----------------------")
    print(f"resolved config path: {resolved_config_path}")
    print(f"CLEAN_ROOT: {clean_root}")
    print(f"DEEPSEEK_REPO: {deepseek_repo}")
    print(f"CLEAN_OVERRIDES: {clean_overrides}")
    print(f"DEEPSEEK_INFERENCE: {deepseek_inference}")
    print(f"kernel.__file__: {kernel_file}")
    print(f"fast_hadamard_transform.__file__: {hadamard_file}")
    print(f"model.__file__: {model_file}")
    print()
    print_symbol_status(model)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test clean DeepSeek import ordering.")
    parser.add_argument("--resolved-config", required=True, help="Resolved config env file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_smoke(Path(args.resolved_config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
