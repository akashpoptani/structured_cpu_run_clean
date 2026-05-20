"""Rank-aware weight loading for native CPU DeepSeek inference.

How weight loading works in the TP2 path
----------------------------------------

DeepSeek-V3.2's `inference/convert.py` (NOT used at runtime; only at
checkpoint-conversion time) takes the HF-style FP8 safetensors and produces
one safetensor file per rank for a target `world_size`:

    model0-mp2.safetensors   # rank 0 shard
    model1-mp2.safetensors   # rank 1 shard

Inside each per-rank file:
  - Every `ColumnParallelLinear` weight is sliced along dim 0 (out_features),
    so the rank-K file contains the K-th `out_features // world_size` slab.
  - Every `RowParallelLinear` weight is sliced along dim 1 (in_features),
    so the rank-K file contains the K-th `in_features // world_size` slab.
  - Replicated tensors (RMSNorm scales, RoPE-related buffers when persisted,
    Linear scales for FP8, embeddings unless ParallelEmbedding splits them)
    are stored unsharded in every file.
  - Parameter names match the constructed model's `state_dict()` keys
    exactly. No alias translation, no key renaming. This is why the legacy
    loader can use `safetensors.torch.load_model(model, path)` directly.

Critical ordering: `torch.distributed.init_process_group("gloo")` must run
BEFORE `Transformer(model_args)` so the module-global `world_size` in
DeepSeek `model.py` is 2 by the time `ColumnParallelLinear.__init__` divides
`out_features // world_size`. If construction happens with `world_size=1`,
the model is built full-size and the rank shards no longer match the
allocated parameter shapes — `load_model` would raise on every parallel
weight.

What this module does
---------------------

For TP2 specifically:
  1. Pick the per-rank shard file `f"model{rank}-mp{world_size}.safetensors"`
     under `SHARDED_CKPT_PATH`.
  2. Verify the file exists (clear failure otherwise).
  3. Call `safetensors.torch.load_model(model, str(shard_path), strict=False)`.
     `strict=False` matches the legacy convention; we additionally report
     any returned `missing_keys` / `unexpected_keys` rather than silently
     ignoring them.
  4. Call `model.eval()`.

There is no key remapping in this clean path because:
  - the constructed parameter names already match the shard's keys, and
  - sharding is already encoded in the file's tensor *shapes*, not its keys.

For the (future) DP modes the path differs: a single full-size shard
`model0-mp1.safetensors` is loaded on every rank without dist initialized
yet, then dist is initialized AFTER construction. That branch is documented
but not implemented here.
"""

import sys
from pathlib import Path
from typing import Any, Dict


def _fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def _resolve_path(clean_root: Path, raw: str) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = clean_root / p
    return p.resolve()


def resolve_shard_path(
    sharded_ckpt_path: Path, sharding_mode: str, rank: int, world_size: int
) -> Path:
    """Return the per-rank safetensor path for the given sharding mode."""
    sharding_mode = sharding_mode.lower()
    if sharding_mode == "tp2":
        return sharded_ckpt_path / f"model{rank}-mp{world_size}.safetensors"
    if sharding_mode in ("dp2", "dp2_epon"):
        return sharded_ckpt_path / "model0-mp1.safetensors"
    _fail(f"unsupported SHARDING_MODE for weight loading: {sharding_mode!r}")
    return Path("/dev/null")  # unreachable; satisfies type checker


def load_weights_into_transformer(
    transformer: Any,
    config: Dict[str, str],
    dist_env: Dict[str, Any],
    log_fn=print,
) -> Dict[str, Any]:
    """Load the per-rank safetensor shard into a constructed Transformer.

    Returns a report dict with the shard path, missing_keys, unexpected_keys,
    and a count summary. Fails loudly on missing files; warns loudly on
    missing/unexpected keys but does not raise (mirrors the legacy
    `strict=False` convention while making it visible).
    """
    sharded_path_raw = config.get("SHARDED_CKPT_PATH", "").strip()
    if not sharded_path_raw:
        _fail("SHARDED_CKPT_PATH is empty; cannot load TP shards")

    clean_root = Path(config["CLEAN_ROOT"]).resolve()
    sharded_ckpt_path = _resolve_path(clean_root, sharded_path_raw)
    if not sharded_ckpt_path.is_dir():
        _fail(f"SHARDED_CKPT_PATH is not a directory: {sharded_ckpt_path}")

    sharding_mode = config.get("SHARDING_MODE", "")
    rank = int(dist_env["rank"])
    world_size = int(dist_env["world_size"])

    shard_path = resolve_shard_path(sharded_ckpt_path, sharding_mode, rank, world_size)
    if not shard_path.is_file():
        _fail(f"shard file not found: {shard_path}")

    log_fn(
        f"[weight-load] sharding_mode={sharding_mode} rank={rank}/{world_size} "
        f"shard={shard_path}"
    )

    from safetensors.torch import load_model

    try:
        result = load_model(transformer, str(shard_path), strict=False)
    except TypeError:
        # Older safetensors lacks the strict kwarg.
        result = load_model(transformer, str(shard_path))

    if isinstance(result, tuple) and len(result) == 2:
        missing_keys, unexpected_keys = result
    else:
        missing_keys, unexpected_keys = [], []

    missing_count = len(missing_keys)
    unexpected_count = len(unexpected_keys)

    log_fn(
        f"[weight-load] loaded: missing_keys={missing_count} "
        f"unexpected_keys={unexpected_count}"
    )
    if missing_count:
        sample = list(missing_keys)[:5]
        log_fn(f"[weight-load]   missing sample (up to 5): {sample}")
    if unexpected_count:
        sample = list(unexpected_keys)[:5]
        log_fn(f"[weight-load]   unexpected sample (up to 5): {sample}")

    transformer.eval()

    return {
        "shard_path": str(shard_path),
        "sharding_mode": sharding_mode,
        "rank": rank,
        "world_size": world_size,
        "missing_count": missing_count,
        "unexpected_count": unexpected_count,
        "missing_keys": list(missing_keys),
        "unexpected_keys": list(unexpected_keys),
    }


def maybe_dequantize_fp8(
    transformer: Any, config: Dict[str, str], log_fn=print
) -> Dict[str, Any]:
    """Optionally pre-dequantize FP8 weights to BF16 in place.

    Scope comes from DEQUANT_FP8_WEIGHTS in the resolved env: 'all' | 'dense'
    | 'none'. For TP2 token-exact baseline 'all' matches the legacy.

    Returns the stats dict from dequantize_fp8_weights, or a sentinel when
    scope='none'.
    """
    scope = (config.get("DEQUANT_FP8_WEIGHTS", "none") or "none").lower()
    if scope == "none":
        log_fn("[weight-load] DEQUANT_FP8_WEIGHTS=none -> skipping pre-dequant")
        return {"scope": "none", "dequant_count": 0}

    # Imported here so the override is loaded via the clean overrides sys.path
    # set up by import_deepseek_modules() — kept lazy to avoid mandatory torch
    # import at module load if a caller only uses other helpers.
    from dequant_weights import dequantize_fp8_weights  # type: ignore

    stats = dequantize_fp8_weights(transformer, scope=scope, log_fn=log_fn)
    stats["scope"] = scope
    return stats
