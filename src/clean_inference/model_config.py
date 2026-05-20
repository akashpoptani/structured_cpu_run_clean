"""Load native DeepSeek ModelArgs JSON and instantiate ModelArgs from it.

Native DeepSeek `inference/model.py` consumes a JSON whose keys are exactly
`ModelArgs` field names (no aliases, no nesting). The upstream pattern is:

    args = ModelArgs(**json.load(f))
    model = Transformer(args)

The HF-style checkpoint `config.json` under ACTIVE_MODEL_PATH is NOT the source
of truth for native ModelArgs and is intentionally not consumed here.
"""

import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Fields that must come from explicit runtime overrides, not the native config,
# for the clean lane. These describe runtime allocation/precision choices.
RUNTIME_OVERRIDE_FIELDS = ("dtype", "max_batch_size", "max_seq_len")


def _fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_native_modelargs_config(config_path: Path) -> Dict[str, Any]:
    """Read a native DeepSeek ModelArgs JSON and return its top-level dict."""
    if not config_path.exists():
        _fail(f"native ModelArgs config not found: {config_path}")
    if not config_path.is_file():
        _fail(f"native ModelArgs config is not a file: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        _fail(f"{config_path}: invalid JSON: {exc}")
    except OSError as exc:
        _fail(f"{config_path}: could not read: {exc}")

    if not isinstance(data, dict):
        _fail(f"{config_path}: top-level JSON must be an object")

    return data


def modelargs_from_native_config(
    model_module: Any,
    native_config: Dict[str, Any],
    overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Build ModelArgs from a native ModelArgs JSON and apply runtime overrides.

    Returns (instance, report). The report has shape:
      source: "native"
      native_fields: sorted list of fields populated from native_config
      overridden: list of {"field", "value"} for applied overrides
      defaulted: ModelArgs fields neither in native_config nor in overrides
      unknown_keys: keys in native_config that are not ModelArgs fields
                    (always empty when this function succeeds; failure happens
                    earlier with a clear message)
    """
    cls = getattr(model_module, "ModelArgs", None)
    if cls is None:
        _fail("model module has no ModelArgs class")
    if not dataclasses.is_dataclass(cls):
        _fail("ModelArgs is not a dataclass; cannot instantiate from JSON")

    valid_fields = {f.name for f in dataclasses.fields(cls)}
    unknown_keys = sorted(key for key in native_config if key not in valid_fields)
    if unknown_keys:
        _fail(
            "native ModelArgs config contains keys that are not ModelArgs fields: "
            f"{unknown_keys}"
        )

    try:
        instance = cls(**native_config)
    except Exception as exc:
        _fail(f"could not instantiate ModelArgs(**native_config): {exc!r}")

    overridden: List[Dict[str, Any]] = []
    if overrides:
        for field_name, value in overrides.items():
            if field_name not in valid_fields:
                _fail(f"override field is not a ModelArgs field: {field_name}")
            try:
                setattr(instance, field_name, value)
            except Exception as exc:
                _fail(f"could not set ModelArgs.{field_name} = {value!r}: {exc!r}")
            overridden.append({"field": field_name, "value": value})

    native_fields = sorted(native_config.keys())
    overridden_names = {entry["field"] for entry in overridden}
    defaulted = sorted(
        name for name in valid_fields
        if name not in native_config and name not in overridden_names
    )

    report = {
        "source": "native",
        "native_fields": native_fields,
        "overridden": overridden,
        "defaulted": defaulted,
        "unknown_keys": [],
    }
    return instance, report


def summarize_modelargs(args: Any) -> Dict[str, Any]:
    """Return all dataclass fields and current values as a plain dict."""
    if not dataclasses.is_dataclass(args):
        _fail("summarize_modelargs requires a dataclass instance")
    return {field.name: getattr(args, field.name) for field in dataclasses.fields(args)}
