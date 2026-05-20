#!/usr/bin/env python3
"""Native CPU inference preflight: inspect env, paths, imports, model metadata, signatures.

Does NOT instantiate Transformer.
Does NOT load weights.
Does NOT call torch.distributed.
Does NOT run generation.
"""

import argparse
import dataclasses
import importlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
CLEAN_ROOT = SCRIPT_DIR.parent
if str(CLEAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEAN_ROOT))

from src.clean_inference.config import parse_resolved_env, require_config_keys, resolve_path
from src.clean_inference.imports import import_deepseek_modules
from src.clean_inference.model_files import inspect_model_path
from src.clean_inference.model_config import (
    load_native_modelargs_config,
    modelargs_from_native_config,
    summarize_modelargs,
)


NATIVE_HIGHLIGHT_FIELDS = (
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
    "q_lora_rank",
    "kv_lora_rank",
    "index_n_heads",
    "index_topk",
)


REQUIRED_KEYS = (
    "TAG",
    "CLEAN_ROOT",
    "DEEPSEEK_REPO",
    "ACTIVE_MODEL_PATH",
    "MODEL_ARGS_CONFIG_PATH",
)


SYMBOLS_TO_CHECK = (
    ("Transformer", ("Transformer",)),
    ("ModelArgs", ("ModelArgs",)),
    ("Block or TransformerBlock", ("Block", "TransformerBlock")),
    ("MLA", ("MLA",)),
    ("MoE", ("MoE",)),
)

GLOBAL_NAMES = (
    "world_size",
    "rank",
    "local_rank",
    "block_size",
    "gemm_impl",
    "attn_impl",
)


def _try_import_version(name: str) -> Optional[str]:
    try:
        module = importlib.import_module(name)
    except Exception:
        return None
    return getattr(module, "__version__", None)


def _inspect_env() -> Dict[str, Any]:
    return {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "torch_version": _try_import_version("torch"),
        "transformers_version": _try_import_version("transformers"),
        "safetensors_version": _try_import_version("safetensors"),
    }


def _signature_or_none(obj: Any) -> Optional[str]:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return None


def _inspect_modelargs(model_module: Any) -> Dict[str, Any]:
    cls = getattr(model_module, "ModelArgs", None)
    info: Dict[str, Any] = {"present": cls is not None}
    if cls is None:
        return info

    info["is_dataclass"] = dataclasses.is_dataclass(cls)
    info["annotations"] = list(getattr(cls, "__annotations__", {}).keys())

    if info["is_dataclass"]:
        fields_info: List[Dict[str, Any]] = []
        has_required = False
        for field in dataclasses.fields(cls):
            has_default = field.default is not dataclasses.MISSING
            has_factory = field.default_factory is not dataclasses.MISSING  # type: ignore[misc]
            if not has_default and not has_factory:
                has_required = True
            fields_info.append(
                {
                    "name": field.name,
                    "type": str(field.type),
                    "has_default": has_default,
                    "default": repr(field.default) if has_default else None,
                    "has_default_factory": has_factory,
                }
            )
        info["fields"] = fields_info
        info["has_required_args"] = has_required

        if not has_required:
            try:
                instance = cls()
                info["instantiated"] = True
                info["instance_repr"] = repr(instance)
            except Exception as exc:
                info["instantiated"] = False
                info["instantiation_error"] = repr(exc)
        else:
            info["instantiated"] = False
            info["instantiation_skipped_reason"] = "ModelArgs requires arguments"

    return info


def _inspect_signatures(model_module: Any) -> Dict[str, Any]:
    sigs: Dict[str, Any] = {}

    transformer = getattr(model_module, "Transformer", None)
    if transformer is not None:
        sigs["Transformer.__init__"] = _signature_or_none(transformer.__init__)
        forward = getattr(transformer, "forward", None)
        sigs["Transformer.forward"] = _signature_or_none(forward) if forward else None

    mla = getattr(model_module, "MLA", None)
    if mla is not None:
        forward = getattr(mla, "forward", None)
        sigs["MLA.forward"] = _signature_or_none(forward) if forward else None

    moe = getattr(model_module, "MoE", None)
    if moe is not None:
        forward = getattr(moe, "forward", None)
        sigs["MoE.forward"] = _signature_or_none(forward) if forward else None

    return sigs


def _inspect_globals(model_module: Any) -> Dict[str, Any]:
    globals_info: Dict[str, Any] = {}
    for name in GLOBAL_NAMES:
        if hasattr(model_module, name):
            value = getattr(model_module, name)
            try:
                globals_info[name] = repr(value)
            except Exception as exc:
                globals_info[name] = f"<repr failed: {exc!r}>"
        else:
            globals_info[name] = None
    return globals_info


def _inspect_symbols(model_module: Any) -> Dict[str, bool]:
    out: Dict[str, bool] = {}
    for label, names in SYMBOLS_TO_CHECK:
        out[label] = any(hasattr(model_module, name) for name in names)
    return out


def _inspect_native_modelargs(model_module: Any, native_config_path: Path) -> Dict[str, Any]:
    """Load native ModelArgs JSON and report what it populates / leaves defaulted."""
    if not native_config_path.is_file():
        return {
            "present": False,
            "reason": "native ModelArgs config not found",
            "config_path": str(native_config_path),
        }

    native_config = load_native_modelargs_config(native_config_path)
    args, report = modelargs_from_native_config(model_module, native_config)
    summary = summarize_modelargs(args)

    highlight = {name: summary.get(name) for name in NATIVE_HIGHLIGHT_FIELDS}

    return {
        "present": True,
        "config_path": str(native_config_path),
        "config_filename": native_config_path.name,
        "source": report["source"],
        "native_fields": report["native_fields"],
        "defaulted": report["defaulted"],
        "modelargs_summary": {k: str(v) for k, v in summary.items()},
        "highlight": {k: str(v) for k, v in highlight.items()},
    }


def _inspect_hf_config_note(model_path: Path) -> Dict[str, Any]:
    """Note whether the HF-style checkpoint config.json exists. Not consumed."""
    hf_path = model_path / "config.json"
    return {
        "hf_config_path": str(hf_path),
        "hf_config_exists": hf_path.is_file(),
        "note": (
            "HF-style config.json under ACTIVE_MODEL_PATH is checkpoint metadata "
            "only and is not consumed by the native ModelArgs path."
        ),
    }


CONSTRUCTION_RISK_NOTES = (
    "Transformer construction is not attempted in this script.",
    "Construction may allocate parameters and KV caches depending on ModelArgs.",
    "Next step should be explicit model construction smoke with controlled args and no weight loading.",
)


def build_summary(resolved_config_path: Path) -> Dict[str, Any]:
    config = parse_resolved_env(resolved_config_path)
    require_config_keys(config, REQUIRED_KEYS)

    env_info = _inspect_env()

    bundle = import_deepseek_modules(config)
    paths = bundle["paths"]
    model_module = bundle["model"]

    clean_root = paths["clean_root"]
    active_model_path_raw = config["ACTIVE_MODEL_PATH"]
    active_model_path = Path(active_model_path_raw)
    if not active_model_path.is_absolute():
        active_model_path = clean_root / active_model_path

    paths_summary = {
        "CLEAN_ROOT": str(clean_root),
        "DEEPSEEK_REPO": str(paths["deepseek_repo"]),
        "CLEAN_OVERRIDES": str(paths["clean_overrides"]),
        "DEEPSEEK_INFERENCE": str(paths["deepseek_inference"]),
        "ACTIVE_MODEL_PATH": str(active_model_path),
    }

    module_files = {
        "kernel": str(bundle["kernel_file"]),
        "fast_hadamard_transform": str(bundle["fast_hadamard_transform_file"]),
        "model": str(bundle["model_file"]),
    }

    native_config_path = resolve_path(clean_root, config["MODEL_ARGS_CONFIG_PATH"]).resolve()
    paths_summary["MODEL_ARGS_CONFIG_PATH"] = str(native_config_path)

    model_files_info = inspect_model_path(active_model_path)
    symbols = _inspect_symbols(model_module)
    modelargs = _inspect_modelargs(model_module)
    signatures = _inspect_signatures(model_module)
    globals_info = _inspect_globals(model_module)
    native_modelargs = _inspect_native_modelargs(model_module, native_config_path)
    hf_note = _inspect_hf_config_note(active_model_path)

    return {
        "resolved_config_path": str(resolved_config_path),
        "tag": config.get("TAG", ""),
        "environment": env_info,
        "paths": paths_summary,
        "module_files": module_files,
        "model_files": model_files_info,
        "model_symbols": symbols,
        "modelargs": modelargs,
        "signatures": signatures,
        "model_globals": globals_info,
        "native_modelargs": native_modelargs,
        "hf_config_note": hf_note,
        "construction_risk_notes": list(CONSTRUCTION_RISK_NOTES),
    }


def _print_kv(items: Dict[str, Any]) -> None:
    for key, value in items.items():
        print(f"  {key}: {value}")


def print_human(summary: Dict[str, Any]) -> None:
    print("Model preflight")
    print("---------------")
    print(f"resolved config: {summary['resolved_config_path']}")
    print(f"TAG: {summary['tag']}")
    print()

    print("Environment:")
    _print_kv(summary["environment"])
    print()

    print("Paths:")
    _print_kv(summary["paths"])
    print()

    print("Imported module files:")
    _print_kv(summary["module_files"])
    print()

    mf = summary["model_files"]
    print("Model path inspection:")
    print(f"  path: {mf['path']}")
    print(f"  exists: {mf['exists']}")
    print(f"  is_dir: {mf['is_dir']}")
    print(f"  config-like files present: {mf['config_like_files_present']}")
    print(f"  safetensors count: {mf['safetensors_count']}")
    print(f"  safetensors total GiB: {mf['safetensors_total_gib']}")
    print(f"  index files: {mf['index_files']}")
    print(f"  first 10 safetensors: {mf['safetensors_first_10']}")
    print(f"  tokenizer files present: {mf['tokenizer_files_present']}")
    print(f"  has tokenizer files: {mf['has_tokenizer_files']}")
    print()

    print("Model symbols:")
    for label, found in summary["model_symbols"].items():
        print(f"  {label}: {'yes' if found else 'no'}")
    print()

    ma = summary["modelargs"]
    print("ModelArgs:")
    print(f"  present: {ma['present']}")
    if ma["present"]:
        print(f"  is_dataclass: {ma.get('is_dataclass')}")
        print(f"  annotations: {ma.get('annotations')}")
        if ma.get("is_dataclass"):
            print(f"  has_required_args: {ma.get('has_required_args')}")
            print(f"  instantiated: {ma.get('instantiated')}")
            if "instance_repr" in ma:
                print(f"  instance repr: {ma['instance_repr']}")
            if "instantiation_skipped_reason" in ma:
                print(f"  skipped: {ma['instantiation_skipped_reason']}")
            if "instantiation_error" in ma:
                print(f"  instantiation error: {ma['instantiation_error']}")
            print("  fields:")
            for field in ma.get("fields", []):
                default_repr = (
                    field["default"]
                    if field["has_default"]
                    else ("<factory>" if field["has_default_factory"] else "<required>")
                )
                print(f"    - {field['name']}: {field['type']} = {default_repr}")
    print()

    print("Signatures:")
    for name, sig in summary["signatures"].items():
        print(f"  {name}: {sig}")
    print()

    print("Module globals:")
    for name, value in summary["model_globals"].items():
        if value is None:
            print(f"  {name}: <not present>")
        else:
            print(f"  {name}: {value}")
    print()

    nm = summary["native_modelargs"]
    print(f"ModelArgs source: native {nm.get('config_filename', '<missing>')}")
    print(f"  MODEL_ARGS_CONFIG_PATH: {nm['config_path']}")
    if not nm["present"]:
        print(f"  status: not loaded ({nm.get('reason', 'unknown')})")
    else:
        print(f"  native config fields ({len(nm['native_fields'])}): {nm['native_fields']}")
        print(f"  defaulted ModelArgs fields ({len(nm['defaulted'])}): {nm['defaulted']}")
        print("  highlight ModelArgs fields:")
        for name in NATIVE_HIGHLIGHT_FIELDS:
            print(f"    {name}: {nm['highlight'].get(name)}")
        print("  resulting ModelArgs summary:")
        for name, value in nm["modelargs_summary"].items():
            print(f"    {name}: {value}")
    print()

    hf = summary["hf_config_note"]
    print("HF checkpoint metadata note:")
    print(f"  {hf['hf_config_path']}")
    print(f"  exists: {hf['hf_config_exists']}")
    print(f"  {hf['note']}")
    print()

    print("Construction risk notes:")
    for note in summary["construction_risk_notes"]:
        print(f"  - {note}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Native CPU inference preflight.")
    parser.add_argument("--resolved-config", required=True, help="Resolved config env file.")
    parser.add_argument("--format", choices=("human", "json"), default="human")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_summary(Path(args.resolved_config))

    if args.format == "json":
        print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    else:
        print_human(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
