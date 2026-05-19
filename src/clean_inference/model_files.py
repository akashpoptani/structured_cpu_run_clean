"""Lightweight on-disk inspection of a model directory (no weight reads)."""

from pathlib import Path
from typing import Any, Dict, List


CONFIG_LIKE_FILENAMES = (
    "config.json",
    "params.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "generation_config.json",
)

TOKENIZER_HINT_FILENAMES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
)

GIB = 1024 ** 3


def _list_present(path: Path, names: tuple) -> List[str]:
    return [name for name in names if (path / name).is_file()]


def inspect_model_path(path: Path) -> Dict[str, Any]:
    """Return a metadata-only inspection of a model directory.

    Does not read safetensor payloads. Only lists files and sums sizes.
    """
    result: Dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "is_dir": path.is_dir() if path.exists() else False,
        "config_like_files": [],
        "config_like_files_present": [],
        "safetensors_count": 0,
        "safetensors_total_bytes": 0,
        "safetensors_total_gib": 0.0,
        "safetensors_first_10": [],
        "index_files": [],
        "tokenizer_files_present": [],
        "has_tokenizer_files": False,
    }

    if not result["exists"] or not result["is_dir"]:
        return result

    config_present = _list_present(path, CONFIG_LIKE_FILENAMES)
    result["config_like_files"] = list(CONFIG_LIKE_FILENAMES)
    result["config_like_files_present"] = config_present

    safetensors_files = sorted(p for p in path.glob("*.safetensors") if p.is_file())
    result["safetensors_count"] = len(safetensors_files)

    total_bytes = 0
    for entry in safetensors_files:
        try:
            total_bytes += entry.stat().st_size
        except OSError:
            continue
    result["safetensors_total_bytes"] = total_bytes
    result["safetensors_total_gib"] = round(total_bytes / GIB, 3)
    result["safetensors_first_10"] = [entry.name for entry in safetensors_files[:10]]

    index_files = sorted(p.name for p in path.glob("*.index.json") if p.is_file())
    result["index_files"] = index_files

    tokenizer_present = _list_present(path, TOKENIZER_HINT_FILENAMES)
    result["tokenizer_files_present"] = tokenizer_present
    result["has_tokenizer_files"] = bool(tokenizer_present)

    return result
