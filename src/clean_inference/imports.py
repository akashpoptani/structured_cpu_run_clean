"""Set up and validate the clean-override import path before loading DeepSeek."""

import importlib
import sys
from pathlib import Path
from typing import Any, Dict

from .config import resolve_path


def _fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def _require_dir(path: Path, label: str) -> None:
    if not path.exists():
        _fail(f"{label} does not exist: {path}")
    if not path.is_dir():
        _fail(f"{label} is not a directory: {path}")


def _is_within(path: Path, root: Path) -> bool:
    path_resolved = path.resolve()
    root_resolved = root.resolve()
    return path_resolved == root_resolved or root_resolved in path_resolved.parents


def setup_deepseek_imports(config: Dict[str, str]) -> Dict[str, Path]:
    """Prepend src/overrides/ then <DEEPSEEK_REPO>/inference to sys.path.

    Returns a dict of resolved paths used during setup.
    """
    clean_root = Path(config["CLEAN_ROOT"]).resolve()
    deepseek_repo = resolve_path(clean_root, config["DEEPSEEK_REPO"]).resolve()
    clean_overrides = clean_root / "src" / "overrides"
    deepseek_inference = deepseek_repo / "inference"

    _require_dir(clean_overrides, "CLEAN_OVERRIDES")
    _require_dir(deepseek_inference, "DEEPSEEK_INFERENCE")

    # Order matters: overrides must be searched first, so insert them last.
    sys.path.insert(0, str(deepseek_inference))
    sys.path.insert(0, str(clean_overrides))

    return {
        "clean_root": clean_root,
        "deepseek_repo": deepseek_repo,
        "clean_overrides": clean_overrides,
        "deepseek_inference": deepseek_inference,
    }


def _import_module(name: str):
    try:
        return importlib.import_module(name)
    except Exception as exc:
        _fail(f"failed to import {name}: {exc}")


def import_deepseek_modules(config: Dict[str, str]) -> Dict[str, Any]:
    """Set up sys.path, import kernel/fast_hadamard_transform/model, validate origins.

    Returns a dict with both the imported modules and the resolved paths.
    """
    paths = setup_deepseek_imports(config)
    clean_overrides = paths["clean_overrides"]
    deepseek_inference = paths["deepseek_inference"]

    kernel = _import_module("kernel")
    fast_hadamard_transform = _import_module("fast_hadamard_transform")
    model = _import_module("model")

    kernel_file = Path(kernel.__file__).resolve()
    hadamard_file = Path(fast_hadamard_transform.__file__).resolve()
    model_file = Path(model.__file__).resolve()

    if not _is_within(kernel_file, clean_overrides):
        _fail(f"kernel imported from wrong path: {kernel_file}")
    if not _is_within(hadamard_file, clean_overrides):
        _fail(f"fast_hadamard_transform imported from wrong path: {hadamard_file}")
    if not _is_within(model_file, deepseek_inference):
        _fail(f"model imported from wrong path: {model_file}")

    return {
        "paths": paths,
        "kernel": kernel,
        "fast_hadamard_transform": fast_hadamard_transform,
        "model": model,
        "kernel_file": kernel_file,
        "fast_hadamard_transform_file": hadamard_file,
        "model_file": model_file,
    }
