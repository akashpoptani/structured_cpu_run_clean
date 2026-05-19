#!/usr/bin/env python3
"""Smoke test clean override import ordering for DeepSeek inference modules."""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CLEAN_ROOT = SCRIPT_DIR.parent
if str(CLEAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEAN_ROOT))

from src.clean_inference.config import parse_resolved_env, require_config_keys
from src.clean_inference.imports import import_deepseek_modules


REQUIRED_KEYS = ("CLEAN_ROOT", "DEEPSEEK_REPO")


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
    config = parse_resolved_env(resolved_config_path)
    require_config_keys(config, REQUIRED_KEYS)

    bundle = import_deepseek_modules(config)
    paths = bundle["paths"]

    print("Inference import smoke")
    print("----------------------")
    print(f"resolved config path: {resolved_config_path}")
    print(f"CLEAN_ROOT: {paths['clean_root']}")
    print(f"DEEPSEEK_REPO: {paths['deepseek_repo']}")
    print(f"CLEAN_OVERRIDES: {paths['clean_overrides']}")
    print(f"DEEPSEEK_INFERENCE: {paths['deepseek_inference']}")
    print(f"kernel.__file__: {bundle['kernel_file']}")
    print(f"fast_hadamard_transform.__file__: {bundle['fast_hadamard_transform_file']}")
    print(f"model.__file__: {bundle['model_file']}")
    print()
    print_symbol_status(bundle["model"])


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
