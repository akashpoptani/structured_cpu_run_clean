"""Resolved-config parsing shared across clean-lane scripts."""

import shlex
import sys
from pathlib import Path
from typing import Dict, Iterable


def _fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def parse_resolved_env(path: Path) -> Dict[str, str]:
    """Parse a resolved env snapshot emitted by scripts/parse_config.sh --format env.

    Each line is KEY=VALUE where VALUE is shell-quoted (printf %q). Blank lines
    and lines starting with '#' are ignored.
    """
    if not path.exists():
        _fail(f"resolved config does not exist: {path}")
    if not path.is_file():
        _fail(f"resolved config is not a file: {path}")

    config: Dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        _fail(f"could not read resolved config {path}: {exc}")

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            _fail(f"{path}:{line_number}: expected KEY=VALUE")

        key, value_part = line.split("=", 1)
        key = key.strip()
        value_part = value_part.strip()
        if not key:
            _fail(f"{path}:{line_number}: empty key")

        if value_part == "":
            value = ""
        else:
            try:
                parsed = shlex.split(value_part)
            except ValueError as exc:
                _fail(f"{path}:{line_number}: could not parse value for {key}: {exc}")
            value = parsed[0] if parsed else ""

        config[key] = value

    return config


def resolve_path(clean_root: Path, raw_path: str) -> Path:
    """Resolve raw_path relative to clean_root unless it is already absolute."""
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return clean_root / path


def require_config_keys(config: Dict[str, str], keys: Iterable[str]) -> None:
    """Fail with SystemExit(1) if any key is missing or empty in config."""
    for key in keys:
        if key not in config or config[key] == "":
            _fail(f"resolved config missing required key: {key}")
