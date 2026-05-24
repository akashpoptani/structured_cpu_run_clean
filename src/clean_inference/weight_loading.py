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

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional


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


# ---------- Dequant-all BF16 cache ----------
#
# When DEQUANT_FP8_WEIGHTS=all and the operator has paid the ~51 min dequant
# pass cost once, the resulting in-memory BF16 model can be persisted to
# scratch as a per-rank safetensor shard. A later run with the same TP shape
# can `safetensors.torch.load_model` the BF16 cache directly, skipping the
# FP8 load AND the dequant pass entirely.
#
# Important: when reading from the cache, model construction must use
# `dtype="bf16"` (not "fp8") so DeepSeek `model.py`'s `Linear` class does not
# allocate FP8-typed `weight` and `scale` parameters. That's the caller's
# responsibility — `resolve_cache_plan()` reports `override_modelargs_dtype`
# in the plan so the modelargs builder can apply it before construction.

CACHE_FILENAME_TEMPLATE = "model{rank}-mp{world_size}-bf16-dequant-all.safetensors"
CACHE_METADATA_TEMPLATE = "model{rank}-mp{world_size}-bf16-dequant-all.metadata.json"


def _cache_shard_file(cache_dir: Path, rank: int, world_size: int) -> Path:
    return cache_dir / CACHE_FILENAME_TEMPLATE.format(rank=rank, world_size=world_size)


def _cache_metadata_file(cache_dir: Path, rank: int, world_size: int) -> Path:
    return cache_dir / CACHE_METADATA_TEMPLATE.format(rank=rank, world_size=world_size)


def resolve_cache_plan(
    config: Dict[str, str], dist_env: Dict[str, Any], log_fn=print
) -> Dict[str, Any]:
    """Decide the cache action for this rank given DEQUANT_CACHE_MODE.

    Returns a plan dict with:
      mode                       — normalized lowercase mode string.
      cache_dir                  — Path or None.
      cache_file                 — Path or None.
      cache_metadata_file        — Path or None.
      cache_exists               — bool: cache_file existed at plan time.
      do_read_cache              — bool: load weights from BF16 cache (skips
                                   FP8 load + dequant entirely).
      do_fp8_load_then_dequant   — bool: take the FP8 load path and run the
                                   dequant pass.
      do_write_cache             — bool: write BF16 cache after dequant.
      override_modelargs_dtype   — "bf16" when do_read_cache; else None. The
                                   modelargs builder must apply this before
                                   construction so Linear layers are BF16.
    """
    mode = (config.get("DEQUANT_CACHE_MODE") or "off").strip().lower()
    path_raw = (config.get("DEQUANT_CACHE_PATH") or "").strip()
    dequant_scope = (config.get("DEQUANT_FP8_WEIGHTS") or "none").strip().lower()
    rank = int(dist_env["rank"])
    world_size = int(dist_env["world_size"])

    plan: Dict[str, Any] = {
        "mode": mode,
        "cache_dir": None,
        "cache_file": None,
        "cache_metadata_file": None,
        "cache_exists": False,
        "do_read_cache": False,
        "do_fp8_load_then_dequant": True,
        "do_write_cache": False,
        "override_modelargs_dtype": None,
    }

    if mode == "off":
        log_fn("[cache] DEQUANT_CACHE_MODE=off -> no cache action")
        return plan

    if not path_raw:
        _fail(f"DEQUANT_CACHE_MODE={mode!r} but DEQUANT_CACHE_PATH is empty")

    clean_root = Path(config["CLEAN_ROOT"]).resolve()
    cache_dir = _resolve_path(clean_root, path_raw)
    cache_file = _cache_shard_file(cache_dir, rank, world_size)
    cache_metadata = _cache_metadata_file(cache_dir, rank, world_size)
    cache_exists = cache_file.is_file()
    plan.update({
        "cache_dir": cache_dir,
        "cache_file": cache_file,
        "cache_metadata_file": cache_metadata,
        "cache_exists": cache_exists,
    })

    if mode in ("write", "read_or_write") and dequant_scope != "all":
        _fail(
            f"DEQUANT_CACHE_MODE={mode!r} requires DEQUANT_FP8_WEIGHTS=all "
            f"(got {dequant_scope!r}); a BF16 cache only makes sense after a "
            f"scope=all dequant pass."
        )

    if mode == "read":
        if not cache_exists:
            _fail(f"DEQUANT_CACHE_MODE=read but cache shard is missing: {cache_file}")
        plan.update({
            "do_read_cache": True,
            "do_fp8_load_then_dequant": False,
            "do_write_cache": False,
            "override_modelargs_dtype": "bf16",
        })
    elif mode == "write":
        # FP8+dequant; write after dequant if not already on disk.
        plan["do_write_cache"] = not cache_exists
        if cache_exists:
            log_fn(f"[cache] mode=write but cache shard already present: {cache_file}; will not overwrite")
    elif mode == "read_or_write":
        if cache_exists:
            plan.update({
                "do_read_cache": True,
                "do_fp8_load_then_dequant": False,
                "do_write_cache": False,
                "override_modelargs_dtype": "bf16",
            })
        else:
            plan["do_write_cache"] = True
    else:
        _fail(f"unsupported DEQUANT_CACHE_MODE={mode!r}")

    log_fn(
        f"[cache] mode={mode} cache_dir={cache_dir} cache_file={cache_file.name} "
        f"exists={cache_exists} do_read={plan['do_read_cache']} "
        f"do_fp8={plan['do_fp8_load_then_dequant']} do_write={plan['do_write_cache']}"
    )
    return plan


def load_cached_bf16_weights(
    transformer: Any, plan: Dict[str, Any], dist_env: Dict[str, Any], log_fn=print
) -> Dict[str, Any]:
    """Load BF16 weights from the dequant cache. Requires plan["do_read_cache"]."""
    if not plan.get("do_read_cache"):
        _fail("load_cached_bf16_weights called but plan.do_read_cache is False")

    cache_file: Path = plan["cache_file"]
    log_fn(f"[cache] reading BF16 dequant cache: {cache_file}")

    from safetensors.torch import load_model

    t0 = time.perf_counter()
    try:
        result = load_model(transformer, str(cache_file), strict=False)
    except TypeError:
        result = load_model(transformer, str(cache_file))
    elapsed = time.perf_counter() - t0

    if isinstance(result, tuple) and len(result) == 2:
        missing_keys, unexpected_keys = result
    else:
        missing_keys, unexpected_keys = [], []

    log_fn(
        f"[cache] BF16 cache loaded in {elapsed:.2f}s: "
        f"missing_keys={len(missing_keys)} unexpected_keys={len(unexpected_keys)}"
    )
    if missing_keys:
        log_fn(f"[cache]   missing sample (up to 5): {list(missing_keys)[:5]}")
    if unexpected_keys:
        log_fn(f"[cache]   unexpected sample (up to 5): {list(unexpected_keys)[:5]}")

    transformer.eval()

    return {
        "source": "bf16_dequant_cache",
        "shard_path": str(cache_file),
        "rank": int(dist_env["rank"]),
        "world_size": int(dist_env["world_size"]),
        "missing_count": len(missing_keys),
        "unexpected_count": len(unexpected_keys),
        "missing_keys": list(missing_keys),
        "unexpected_keys": list(unexpected_keys),
        "load_seconds": elapsed,
    }


def maybe_write_dequant_cache(
    transformer: Any,
    plan: Dict[str, Any],
    config: Dict[str, str],
    dist_env: Dict[str, Any],
    log_fn=print,
) -> Optional[Dict[str, Any]]:
    """Write the per-rank BF16 cache shard + a small metadata JSON if plan
    says to. Returns the write report or None if nothing was written.
    """
    if not plan.get("do_write_cache"):
        return None

    cache_file: Path = plan["cache_file"]
    cache_metadata: Path = plan["cache_metadata_file"]
    rank = int(dist_env["rank"])
    world_size = int(dist_env["world_size"])

    # Safety: never overwrite an existing cache shard.
    if cache_file.is_file():
        log_fn(f"[cache] cache shard already present, skipping write: {cache_file}")
        return {"status": "skipped_exists", "cache_file": str(cache_file)}

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    log_fn(f"[cache] writing BF16 dequant cache: {cache_file}")

    # Use save_file with .detach() (no full clone — that would peak at ~2x
    # model memory). The dequant pass in src/overrides/dequant_weights.py
    # already forces fresh per-tensor storage for ragged shapes via .clone(),
    # so the shared-storage heuristic in safetensors should be satisfied
    # without further copying here.
    from safetensors.torch import save_file

    t0 = time.perf_counter()
    try:
        state_dict = {name: t.detach() for name, t in transformer.state_dict().items()}
        save_file(state_dict, str(cache_file))
        del state_dict
    except Exception as exc:
        log_fn(f"[cache] WARN: save_file failed: {exc!r}")
        return {"status": "write_failed", "error": repr(exc), "cache_file": str(cache_file)}
    elapsed = time.perf_counter() - t0

    try:
        size_bytes = cache_file.stat().st_size
    except OSError:
        size_bytes = -1
    log_fn(
        f"[cache] BF16 cache written in {elapsed:.2f}s ({size_bytes / 1e9:.1f} GB): "
        f"{cache_file}"
    )

    metadata = {
        "source_sharded_ckpt_path": (config.get("SHARDED_CKPT_PATH") or "").strip(),
        "source_shard_filename": CACHE_FILENAME_TEMPLATE.format(rank=rank, world_size=world_size).replace(
            "-bf16-dequant-all", ""
        ),
        "model_args_config_path": (config.get("MODEL_ARGS_CONFIG_PATH") or "").strip(),
        "dequant_scope": "all",
        "created_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rank": rank,
        "world_size": world_size,
        "dtype": "bf16",
        "cache_filename": cache_file.name,
        "cache_size_bytes": size_bytes,
        "write_seconds": elapsed,
    }
    try:
        cache_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        log_fn(f"[cache] metadata written: {cache_metadata}")
    except OSError as exc:
        log_fn(f"[cache] WARN: metadata write failed: {exc!r}")

    return {
        "status": "written",
        "cache_file": str(cache_file),
        "cache_metadata_file": str(cache_metadata),
        "size_bytes": size_bytes,
        "write_seconds": elapsed,
    }
