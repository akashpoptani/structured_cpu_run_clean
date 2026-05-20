#!/usr/bin/env python3
"""Construct a DeepSeek Transformer from the native ModelArgs JSON.

Reads MODEL_ARGS_CONFIG_PATH (the native DeepSeek `config_671B_v3.2.json`),
instantiates ModelArgs via `ModelArgs(**native_config)`, applies runtime
overrides (dtype from WEIGHTS_PRECISION; max_batch_size and max_seq_len from
CLI), and constructs Transformer.

Does NOT load weights.
Does NOT call forward.
Does NOT call torch.distributed.

The HF-style config.json under ACTIVE_MODEL_PATH is intentionally not used here.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

SCRIPT_DIR = Path(__file__).resolve().parent
CLEAN_ROOT = SCRIPT_DIR.parent
if str(CLEAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEAN_ROOT))

from src.clean_inference.config import parse_resolved_env, require_config_keys, resolve_path
from src.clean_inference.imports import import_deepseek_modules
from src.clean_inference.model_config import (
    load_native_modelargs_config,
    modelargs_from_native_config,
    summarize_modelargs,
)


REQUIRED_KEYS = (
    "TAG",
    "CLEAN_ROOT",
    "DEEPSEEK_REPO",
    "ACTIVE_MODEL_PATH",
    "WEIGHTS_PRECISION",
    "MODEL_ARGS_CONFIG_PATH",
)

PRECISION_TO_DTYPE = {"bf16": "bf16", "fp8": "fp8"}

HIGHLIGHT_FIELDS = (
    "n_layers",
    "dim",
    "n_heads",
    "n_routed_experts",
    "n_shared_experts",
    "n_activated_experts",
    "max_seq_len",
    "max_batch_size",
    "dtype",
    "scale_fmt",
    "vocab_size",
    "q_lora_rank",
    "kv_lora_rank",
    "index_n_heads",
    "index_topk",
)


def _fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def _param_dtype_summary(transformer) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for param in transformer.parameters():
        key = str(param.dtype)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _has_buffers(transformer) -> bool:
    for _ in transformer.buffers():
        return True
    return False


def run_smoke(resolved_config_path: Path, max_batch_size: int, max_seq_len: int) -> int:
    config = parse_resolved_env(resolved_config_path)
    require_config_keys(config, REQUIRED_KEYS)

    bundle = import_deepseek_modules(config)
    model_module = bundle["model"]

    clean_root = Path(config["CLEAN_ROOT"]).resolve()
    native_config_path = resolve_path(clean_root, config["MODEL_ARGS_CONFIG_PATH"]).resolve()
    active_model_path = resolve_path(clean_root, config["ACTIVE_MODEL_PATH"]).resolve()

    weights_precision = config["WEIGHTS_PRECISION"]
    dtype_override = PRECISION_TO_DTYPE.get(weights_precision)
    if dtype_override is None:
        _fail(
            f"WEIGHTS_PRECISION={weights_precision!r} has no dtype mapping; "
            f"expected one of {sorted(PRECISION_TO_DTYPE.keys())}"
        )

    overrides: Dict[str, Any] = {
        "dtype": dtype_override,
        "max_batch_size": max_batch_size,
        "max_seq_len": max_seq_len,
    }

    native_config = load_native_modelargs_config(native_config_path)
    args, report = modelargs_from_native_config(model_module, native_config, overrides=overrides)

    print("Model construction smoke")
    print("------------------------")
    print(f"resolved config: {resolved_config_path}")
    print(f"TAG: {config.get('TAG', '')}")
    print(f"ModelArgs source: native {native_config_path.name}")
    print(f"  Native config path: {native_config_path}")
    print(f"ACTIVE_MODEL_PATH: {active_model_path}")
    print(f"WEIGHTS_PRECISION: {weights_precision}")
    print(f"CLI max_batch_size: {max_batch_size}")
    print(f"CLI max_seq_len: {max_seq_len}")
    print()

    print("Applied overrides (post-construction):")
    for entry in report["overridden"]:
        print(f"  - {entry['field']} = {entry['value']}")
    print(f"Native config fields ({len(report['native_fields'])}): {report['native_fields']}")
    print(f"Defaulted ModelArgs fields: {report['defaulted']}")
    print()

    summary = summarize_modelargs(args)
    print("ModelArgs (highlight fields) before construction:")
    for name in HIGHLIGHT_FIELDS:
        if name in summary:
            print(f"  {name}: {summary[name]}")
    print()

    print("Full ModelArgs summary:")
    for name, value in summary.items():
        print(f"  {name}: {value}")
    print()

    print("Constructing Transformer ...")
    sys.stdout.flush()
    try:
        transformer = model_module.Transformer(args)
    except Exception as exc:
        _fail(f"Transformer construction failed: {exc!r}")

    print("Construction succeeded.")
    print(f"  type: {type(transformer)}")

    try:
        total_params = sum(p.numel() for p in transformer.parameters())
        print(f"  total parameters (numel sum): {total_params:,}")
    except Exception as exc:
        print(f"  total parameters: <unavailable: {exc!r}>")

    try:
        dtype_counts = _param_dtype_summary(transformer)
        if dtype_counts:
            print("  parameter dtype counts:")
            for dtype, count in sorted(dtype_counts.items()):
                print(f"    {dtype}: {count}")
        else:
            print("  parameter dtype counts: <no parameters>")
    except Exception as exc:
        print(f"  parameter dtype counts: <unavailable: {exc!r}>")

    try:
        print(f"  has buffers: {_has_buffers(transformer)}")
    except Exception as exc:
        print(f"  has buffers: <unavailable: {exc!r}>")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construct Transformer from native ModelArgs JSON.")
    parser.add_argument("--resolved-config", required=True, help="Resolved config env file.")
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_smoke(Path(args.resolved_config), args.max_batch_size, args.max_seq_len)


if __name__ == "__main__":
    raise SystemExit(main())
