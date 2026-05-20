"""Tokenizer loading and prompt encoding.

Mirrors the legacy verify_cpu.py loader:
  - Prefer TOKENIZER_PATH from resolved env; fall back to SHARDED_CKPT_PATH;
    then ACTIVE_MODEL_PATH. Each candidate must contain tokenizer.json.
  - PreTrainedTokenizerFast(tokenizer_file=tokenizer.json, **special_kwargs)
    where special-token kwargs are pulled from tokenizer_config.json when
    present (bos/eos/pad/unk + a few cleanup options).
  - Encode the prompt with add_special_tokens=False so the token stream
    matches what the reference capture used.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SPECIAL_TOKEN_KEYS = ("bos_token", "eos_token", "pad_token", "unk_token")
COPY_KEYS = (
    "clean_up_tokenization_spaces",
    "model_max_length",
    "padding_side",
    "truncation_side",
)


def _fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def _resolve_path(clean_root: Path, raw: str) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = clean_root / p
    return p.resolve()


def _has_tokenizer_json(directory: Path) -> bool:
    return directory.is_dir() and (directory / "tokenizer.json").is_file()


def find_tokenizer_dir(config: Dict[str, str]) -> Path:
    """Pick the first directory among (TOKENIZER_PATH, SHARDED_CKPT_PATH,
    ACTIVE_MODEL_PATH) that contains a tokenizer.json."""
    clean_root = Path(config["CLEAN_ROOT"]).resolve()
    candidates = []
    for key in ("TOKENIZER_PATH", "SHARDED_CKPT_PATH", "ACTIVE_MODEL_PATH"):
        raw = (config.get(key) or "").strip()
        if not raw:
            continue
        candidates.append((key, _resolve_path(clean_root, raw)))

    for key, path in candidates:
        if _has_tokenizer_json(path):
            return path

    _fail(
        "no tokenizer.json found in any of "
        f"{[(k, str(p)) for k, p in candidates]}"
    )
    return Path("/dev/null")  # unreachable


def _load_tokenizer_kwargs(tokenizer_dir: Path) -> Dict[str, Any]:
    cfg_path = tokenizer_dir / "tokenizer_config.json"
    if not cfg_path.is_file():
        return {}
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"{cfg_path}: could not parse tokenizer_config.json: {exc}")

    kwargs: Dict[str, Any] = {}
    for key in SPECIAL_TOKEN_KEYS:
        value = cfg.get(key)
        if isinstance(value, dict):
            value = value.get("content")
        if value is not None:
            kwargs[key] = value
    for key in COPY_KEYS:
        value = cfg.get(key)
        if value is not None:
            kwargs[key] = value
    return kwargs


def load_tokenizer(config: Dict[str, str], log_fn=print) -> Tuple[Any, Path]:
    """Return (tokenizer, tokenizer_dir). Fails loudly if none found."""
    from transformers import PreTrainedTokenizerFast  # type: ignore

    tokenizer_dir = find_tokenizer_dir(config)
    tokenizer_file = tokenizer_dir / "tokenizer.json"
    kwargs = _load_tokenizer_kwargs(tokenizer_dir)
    log_fn(f"[tokenizer] dir={tokenizer_dir}")
    log_fn(f"[tokenizer] special kwargs: {sorted(kwargs.keys())}")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file), **kwargs)
    return tokenizer, tokenizer_dir


def encode_prompt(tokenizer: Any, prompt_text: str) -> List[int]:
    """Encode a prompt with add_special_tokens=False (matches reference capture)."""
    return tokenizer.encode(prompt_text, add_special_tokens=False)


def try_decode(tokenizer: Any, token_ids: List[int]) -> Optional[str]:
    """Best-effort decode for human-readable debug output."""
    try:
        return tokenizer.decode(token_ids)
    except Exception:
        return None
