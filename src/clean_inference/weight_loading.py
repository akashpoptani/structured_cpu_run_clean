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

import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


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


# ---------- Dequant-all BF16 cache (sharded) ----------
#
# After DEQUANT_FP8_WEIGHTS=all the in-memory BF16 model is ~676 GB per rank.
# An earlier monolithic `safetensors.torch.save_file` write peaked dirty mmap
# pages at the full payload size, crossing the cgroup --mem ceiling and
# triggering OOM-killer. The fix here is a sharded write: chunk the
# state_dict into ~20 GB groups, save each as its own safetensors file, then
# fsync+posix_fadvise(DONTNEED) between shards so dirty pages drop. Peak RSS
# during write stays bounded by ~one shard worth.
#
# Cache compatibility (validated on read by _validate_cache_compat):
#   * matters     : model identity (source_sharded_ckpt_path,
#                   model_args_config_path), sharding mode + world size,
#                   per-rank file (rank), dequant scope, dtype.
#   * does NOT matter: SBATCH_CPUS_PER_TASK, OMP_NUM_THREADS, partition,
#                   account, node names, SLURM job id, thread binding.
# The same TP2 BF16 cache is readable under c=1 or c=96; only performance
# changes. If TP_SIZE / world_size changes, use a different DEQUANT_CACHE_PATH
# and regenerate; the shard slices are world-size-specific by construction.
#
# Layout (per rank):
#   <CACHE_DIR>/model{rank}-mp{ws}-bf16-dequant-all.index.json
#   <CACHE_DIR>/model{rank}-mp{ws}-bf16-dequant-all-00001.safetensors
#   <CACHE_DIR>/model{rank}-mp{ws}-bf16-dequant-all-00002.safetensors
#   ...
#   <CACHE_DIR>/model{rank}-mp{ws}-bf16-dequant-all.metadata.json
#
# The index file is the canonical cache-existence marker. Its body follows
# the HF convention: {"metadata": {...}, "weight_map": {key: shard_filename}}.
#
# Read path: open each shard once and stream-copy each tensor into the
# corresponding model parameter/buffer. Peak extra RSS during load is one
# tensor at a time (a few hundred MB at most), never the full payload.
#
# Important: when reading from the cache, model construction must use
# `dtype="bf16"` (not "fp8") so DeepSeek `model.py`'s `Linear` class does not
# allocate FP8-typed `weight` and `scale` parameters. `resolve_cache_plan()`
# reports `override_modelargs_dtype` in the plan so the modelargs builder
# can apply it before construction.

CACHE_INDEX_TEMPLATE = "model{rank}-mp{world_size}-bf16-dequant-all.index.json"
CACHE_SHARD_TEMPLATE = "model{rank}-mp{world_size}-bf16-dequant-all-{shard:05d}.safetensors"
CACHE_METADATA_TEMPLATE = "model{rank}-mp{world_size}-bf16-dequant-all.metadata.json"
# Legacy single-file naming (predates the sharded layout). Reader falls back
# to this if no index.json exists but the legacy file does.
CACHE_LEGACY_MONO_TEMPLATE = "model{rank}-mp{world_size}-bf16-dequant-all.safetensors"

# Target bytes per shard. The cap is bounded by the cgroup --mem headroom
# above steady-state model RSS. With 800 G cgroup and ~676 G BF16 model,
# headroom is ~120 G; 20 G per shard leaves margin for Python + writer state.
CACHE_SHARD_BYTES = 20 * (1024 ** 3)


def _cache_index_file(cache_dir: Path, rank: int, world_size: int) -> Path:
    return cache_dir / CACHE_INDEX_TEMPLATE.format(rank=rank, world_size=world_size)


def _cache_legacy_mono_file(cache_dir: Path, rank: int, world_size: int) -> Path:
    return cache_dir / CACHE_LEGACY_MONO_TEMPLATE.format(rank=rank, world_size=world_size)


def _cache_metadata_file(cache_dir: Path, rank: int, world_size: int) -> Path:
    return cache_dir / CACHE_METADATA_TEMPLATE.format(rank=rank, world_size=world_size)


def _cache_shard_path(cache_dir: Path, rank: int, world_size: int, shard_id: int) -> Path:
    return cache_dir / CACHE_SHARD_TEMPLATE.format(rank=rank, world_size=world_size, shard=shard_id)


def _drop_page_cache(path: Path, log_fn) -> None:
    """fsync + posix_fadvise(DONTNEED) to flush dirty pages and drop them."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except OSError as exc:
        log_fn(f"[cache] WARN: fsync/fadvise failed for {path.name}: {exc!r}")


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
        "cache_index_file": None,
        "cache_metadata_file": None,
        "cache_legacy_file": None,
        # `cache_file` kept for backward-compat: callers may have logged this
        # path. Points at the index for the sharded layout.
        "cache_file": None,
        "cache_exists": False,
        "cache_layout": None,  # "sharded" | "legacy_monolithic" | None
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
    cache_index = _cache_index_file(cache_dir, rank, world_size)
    cache_metadata = _cache_metadata_file(cache_dir, rank, world_size)
    cache_legacy = _cache_legacy_mono_file(cache_dir, rank, world_size)

    cache_layout: Optional[str] = None
    if cache_index.is_file():
        cache_layout = "sharded"
    elif cache_legacy.is_file():
        cache_layout = "legacy_monolithic"
    cache_exists = cache_layout is not None

    plan.update({
        "cache_dir": cache_dir,
        "cache_index_file": cache_index,
        "cache_metadata_file": cache_metadata,
        "cache_legacy_file": cache_legacy,
        "cache_file": cache_index,  # back-compat alias
        "cache_exists": cache_exists,
        "cache_layout": cache_layout,
    })

    if mode in ("write", "read_or_write") and dequant_scope != "all":
        _fail(
            f"DEQUANT_CACHE_MODE={mode!r} requires DEQUANT_FP8_WEIGHTS=all "
            f"(got {dequant_scope!r}); a BF16 cache only makes sense after a "
            f"scope=all dequant pass."
        )

    if mode == "read":
        if not cache_exists:
            _fail(
                f"DEQUANT_CACHE_MODE=read but no cache found in {cache_dir} "
                f"(neither {cache_index.name} nor {cache_legacy.name})"
            )
        plan.update({
            "do_read_cache": True,
            "do_fp8_load_then_dequant": False,
            "do_write_cache": False,
            "override_modelargs_dtype": "bf16",
        })
    elif mode == "write":
        plan["do_write_cache"] = not cache_exists
        if cache_exists:
            log_fn(
                f"[cache] mode=write but cache already present "
                f"(layout={cache_layout}); will not overwrite"
            )
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
        f"[cache] mode={mode} cache_dir={cache_dir} "
        f"index={cache_index.name} layout={cache_layout} "
        f"exists={cache_exists} do_read={plan['do_read_cache']} "
        f"do_fp8={plan['do_fp8_load_then_dequant']} do_write={plan['do_write_cache']}"
    )
    return plan


def _validate_cache_compat(
    index_path: Path,
    index_metadata: Dict[str, Any],
    sibling_metadata: Optional[Dict[str, Any]],
    config: Dict[str, str],
    dist_env: Dict[str, Any],
    log_fn,
) -> None:
    """Hard-fail if the cache's claimed topology / model identity does not
    match the running config, warn on softer mismatches.

    Compat depends ONLY on model identity and TP topology:
      hard fail : rank, world_size, dtype, dequant_scope — REQUIRED in the
                  cache index metadata. Missing OR mismatched values both
                  abort the read.
      hard fail : sharding_mode WHEN PRESENT in the cache index.
      legacy ok : sharding_mode MISSING (older caches written before that
                  field was recorded). In that case we still accept the
                  cache iff the index filename matches the expected per-rank
                  template AND the running config is SHARDING_MODE=tp2 —
                  that was the only sharding the early cache writer
                  supported. A warning is logged so the operator is aware.
      warn      : source_sharded_ckpt_path / model_args_config_path (the
                  same model can live at multiple paths via symlinks; a
                  mismatch is suspicious but not always fatal)

    Compat is intentionally INDEPENDENT of:
      SBATCH_CPUS_PER_TASK, OMP_NUM_THREADS, partition, account, node names,
      SLURM job id, thread binding. The same TP2 cache is readable under
      c=1 or c=96; only performance changes.
    """
    expected_rank = int(dist_env["rank"])
    expected_world_size = int(dist_env["world_size"])
    expected_sharding_mode = (config.get("SHARDING_MODE") or "").strip().lower()

    incompat: List[str] = []

    _missing = object()

    def _check(field: str, cache_val: Any, expected: Any, *, required: bool = False) -> None:
        if cache_val is _missing or cache_val is None:
            if required:
                incompat.append(f"{field}: missing from cache index metadata (required)")
            return
        if cache_val != expected:
            incompat.append(f"{field}: cache={cache_val!r} expected={expected!r}")

    _check("rank", index_metadata.get("rank", _missing), expected_rank, required=True)
    _check("world_size", index_metadata.get("world_size", _missing), expected_world_size, required=True)
    _check("dtype", index_metadata.get("dtype", _missing), "bf16", required=True)
    _check("dequant_scope", index_metadata.get("dequant_scope", _missing), "all", required=True)

    cache_sharding = index_metadata.get("sharding_mode") or None
    if cache_sharding is not None:
        # New-format cache — sharding_mode is authoritative.
        if expected_sharding_mode:
            _check("sharding_mode", cache_sharding, expected_sharding_mode)
    else:
        # Legacy cache (pre-sharding_mode metadata). Accept iff:
        #   (a) the index filename matches the expected per-rank template,
        #   (b) the running config is SHARDING_MODE=tp2 (the only sharding
        #       supported by the early cache writer).
        expected_index_name = CACHE_INDEX_TEMPLATE.format(
            rank=expected_rank, world_size=expected_world_size
        )
        if index_path.name != expected_index_name:
            incompat.append(
                f"legacy cache filename mismatch: cache={index_path.name!r} "
                f"expected={expected_index_name!r}"
            )
        if expected_sharding_mode and expected_sharding_mode != "tp2":
            incompat.append(
                f"legacy cache without sharding_mode metadata can only be read "
                f"under SHARDING_MODE=tp2; running with {expected_sharding_mode!r}"
            )
        if not incompat:
            log_fn(
                "[cache] metadata missing sharding_mode; accepting legacy TP2 "
                "cache based on filename / rank / world_size / dtype / dequant_scope"
            )

    if incompat:
        _fail(
            "[cache] BF16 dequant cache is incompatible with the running config:\n  "
            + "\n  ".join(incompat)
            + "\n  Note: CPU cores and OMP threads are NOT part of cache compatibility "
            "(same TP cache is readable under c=1 or c=96); the listed fields ARE."
        )

    # Softer warnings using the sibling metadata file.
    if sibling_metadata:
        expected_src_ckpt = (config.get("SHARDED_CKPT_PATH") or "").strip()
        expected_modelargs = (config.get("MODEL_ARGS_CONFIG_PATH") or "").strip()
        cache_src_ckpt = (sibling_metadata.get("source_sharded_ckpt_path") or "").strip()
        cache_modelargs = (sibling_metadata.get("model_args_config_path") or "").strip()
        if cache_src_ckpt and expected_src_ckpt and cache_src_ckpt != expected_src_ckpt:
            log_fn(
                f"[cache] WARN: source_sharded_ckpt_path differs: "
                f"cache={cache_src_ckpt!r} expected={expected_src_ckpt!r} "
                f"(continuing — paths may differ via symlinks but content must match)"
            )
        if cache_modelargs and expected_modelargs and cache_modelargs != expected_modelargs:
            log_fn(
                f"[cache] WARN: model_args_config_path differs: "
                f"cache={cache_modelargs!r} expected={expected_modelargs!r}"
            )

    log_fn(
        f"[cache] compat check OK: rank={expected_rank}/{expected_world_size} "
        f"sharding_mode={expected_sharding_mode or '<unspecified>'} dtype=bf16 scope=all"
    )


def _read_sibling_metadata(cache_dir: Path, rank: int, world_size: int) -> Optional[Dict[str, Any]]:
    """Best-effort read of the per-rank `*.metadata.json`. Missing/invalid is OK."""
    p = _cache_metadata_file(cache_dir, rank, world_size)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_cached_bf16_weights(
    transformer: Any,
    plan: Dict[str, Any],
    dist_env: Dict[str, Any],
    config: Dict[str, str],
    log_fn=print,
) -> Dict[str, Any]:
    """Load BF16 weights from the dequant cache. Requires plan["do_read_cache"].

    Supports both layouts:
      - "sharded"            : index.json + N shard safetensors files.
      - "legacy_monolithic"  : one .safetensors file (pre-sharded layout).

    The sharded path opens each shard exactly once and copies each tensor
    directly into the corresponding model parameter or buffer, so peak extra
    RSS during load is one tensor at a time.
    """
    if not plan.get("do_read_cache"):
        _fail("load_cached_bf16_weights called but plan.do_read_cache is False")

    layout = plan["cache_layout"]
    rank = int(dist_env["rank"])
    world_size = int(dist_env["world_size"])
    cache_dir: Path = plan["cache_dir"]

    if layout == "sharded":
        return _load_sharded_cache(
            transformer, cache_dir, plan["cache_index_file"], rank, world_size,
            config, dist_env, log_fn,
        )
    if layout == "legacy_monolithic":
        return _load_legacy_monolithic_cache(
            transformer, plan["cache_legacy_file"], rank, world_size, log_fn,
        )
    _fail(f"load_cached_bf16_weights: unknown cache_layout={layout!r}")
    return {}  # unreachable


def _stream_copy_one_shard(
    transformer: Any,
    shard_path: Path,
    keys_in_shard: List[str],
    target_params: Dict[str, Any],
    target_buffers: Dict[str, Any],
    log_fn,
) -> Dict[str, int]:
    """Open a shard once and copy each tensor into the model param/buffer in
    place. Returns counts of loaded/unexpected/dtype_mismatch/shape_mismatch.
    """
    import torch
    from safetensors import safe_open

    counts = {"loaded": 0, "unexpected": 0, "dtype_mismatch": 0, "shape_mismatch": 0}
    unexpected_sample: List[str] = []
    with safe_open(str(shard_path), framework="pt", device="cpu") as f:
        shard_keys_available = set(f.keys())
        for key in keys_in_shard:
            if key not in shard_keys_available:
                counts["unexpected"] += 1
                if len(unexpected_sample) < 5:
                    unexpected_sample.append(key + ":not-in-shard")
                continue
            target = target_params.get(key)
            target_kind = "param"
            if target is None:
                target = target_buffers.get(key)
                target_kind = "buffer"
            if target is None:
                counts["unexpected"] += 1
                if len(unexpected_sample) < 5:
                    unexpected_sample.append(key)
                continue
            t = f.get_tensor(key)
            try:
                if t.shape != target.shape:
                    counts["shape_mismatch"] += 1
                    log_fn(f"[cache] WARN: shape mismatch on {key} ({target_kind}): "
                           f"cache={tuple(t.shape)} target={tuple(target.shape)}; skipping")
                    del t
                    continue
                if t.dtype != target.dtype:
                    counts["dtype_mismatch"] += 1
                    log_fn(f"[cache] WARN: dtype mismatch on {key} ({target_kind}): "
                           f"cache={t.dtype} target={target.dtype}; copying with cast")
                with torch.no_grad():
                    target.data.copy_(t)
                counts["loaded"] += 1
            finally:
                del t
    if unexpected_sample:
        log_fn(f"[cache]   unexpected sample from {shard_path.name}: {unexpected_sample}")
    return counts


def _load_sharded_cache(
    transformer: Any,
    cache_dir: Path,
    index_path: Path,
    rank: int,
    world_size: int,
    config: Dict[str, str],
    dist_env: Dict[str, Any],
    log_fn,
) -> Dict[str, Any]:
    log_fn(f"[cache] reading sharded BF16 cache: {index_path}")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"[cache] could not read index {index_path}: {exc!r}")

    # Validate topology / model identity BEFORE touching any shards.
    sibling = _read_sibling_metadata(cache_dir, rank, world_size)
    _validate_cache_compat(index_path, index.get("metadata", {}), sibling, config, dist_env, log_fn)

    weight_map: Dict[str, str] = index.get("weight_map", {})
    if not weight_map:
        _fail(f"[cache] index {index_path} has empty weight_map")

    # Group keys by shard so we open each shard exactly once.
    shard_to_keys: Dict[str, List[str]] = {}
    for key, fname in weight_map.items():
        shard_to_keys.setdefault(fname, []).append(key)

    target_params = dict(transformer.named_parameters())
    target_buffers = dict(transformer.named_buffers())
    target_state_keys = set(target_params) | set(target_buffers)

    totals = {"loaded": 0, "unexpected": 0, "dtype_mismatch": 0, "shape_mismatch": 0}
    t0 = time.perf_counter()
    for shard_fname in sorted(shard_to_keys):
        shard_path = cache_dir / shard_fname
        if not shard_path.is_file():
            _fail(f"[cache] shard listed in index but missing: {shard_path}")
        counts = _stream_copy_one_shard(
            transformer, shard_path, shard_to_keys[shard_fname],
            target_params, target_buffers, log_fn,
        )
        for k, v in counts.items():
            totals[k] += v
        log_fn(
            f"[cache]   shard {shard_fname}: loaded={counts['loaded']} "
            f"unexpected={counts['unexpected']} dtype_mismatch={counts['dtype_mismatch']} "
            f"shape_mismatch={counts['shape_mismatch']}"
        )
        # Drop kernel page cache for this shard so the next shard starts
        # fresh. Helps when several shards are read back-to-back from a
        # bandwidth-limited filesystem.
        _drop_page_cache(shard_path, log_fn)
        gc.collect()
    elapsed = time.perf_counter() - t0

    # Anything in the model but not in the cache index is missing.
    all_cache_keys = set(weight_map.keys())
    missing = sorted(target_state_keys - all_cache_keys)
    log_fn(
        f"[cache] BF16 cache loaded in {elapsed:.2f}s: "
        f"loaded={totals['loaded']} missing={len(missing)} "
        f"unexpected={totals['unexpected']} "
        f"dtype_mismatch={totals['dtype_mismatch']} shape_mismatch={totals['shape_mismatch']}"
    )
    if missing:
        log_fn(f"[cache]   missing sample (up to 5): {missing[:5]}")

    transformer.eval()
    return {
        "source": "bf16_dequant_cache_sharded",
        "index_path": str(index_path),
        "shard_count": len(shard_to_keys),
        "rank": rank,
        "world_size": world_size,
        "loaded": totals["loaded"],
        "missing_count": len(missing),
        "unexpected_count": totals["unexpected"],
        "dtype_mismatch_count": totals["dtype_mismatch"],
        "shape_mismatch_count": totals["shape_mismatch"],
        "missing_keys": missing,
        "load_seconds": elapsed,
    }


def _load_legacy_monolithic_cache(
    transformer: Any,
    cache_file: Path,
    rank: int,
    world_size: int,
    log_fn,
) -> Dict[str, Any]:
    """Back-compat reader for the original single-file cache layout."""
    log_fn(f"[cache] reading legacy monolithic BF16 cache: {cache_file}")
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
        f"[cache] legacy cache loaded in {elapsed:.2f}s: "
        f"missing_keys={len(missing_keys)} unexpected_keys={len(unexpected_keys)}"
    )
    transformer.eval()
    return {
        "source": "bf16_dequant_cache_legacy_mono",
        "shard_path": str(cache_file),
        "rank": rank,
        "world_size": world_size,
        "loaded": -1,  # load_model doesn't report a count
        "missing_count": len(missing_keys),
        "unexpected_count": len(unexpected_keys),
        "missing_keys": list(missing_keys),
        "unexpected_keys": list(unexpected_keys),
        "load_seconds": elapsed,
    }


def _bytes_of(t: Any) -> int:
    return int(t.numel()) * int(t.element_size())


def maybe_write_dequant_cache(
    transformer: Any,
    plan: Dict[str, Any],
    config: Dict[str, str],
    dist_env: Dict[str, Any],
    log_fn=print,
) -> Optional[Dict[str, Any]]:
    """Write a sharded BF16 dequant cache for this rank.

    Each shard caps at ~CACHE_SHARD_BYTES. Between shards we fsync + drop
    page cache so dirty mmap pages don't accumulate (this was the root cause
    of the original monolithic-save OOM at the 800G cgroup ceiling).

    The cache is published atomically by writing the per-rank
    `*.index.json` file LAST. A later run that hits read_or_write will only
    treat the cache as "exists" once the index appears.
    """
    if not plan.get("do_write_cache"):
        return None

    cache_dir: Path = plan["cache_dir"]
    index_path: Path = plan["cache_index_file"]
    cache_metadata: Path = plan["cache_metadata_file"]
    rank = int(dist_env["rank"])
    world_size = int(dist_env["world_size"])

    if index_path.is_file():
        log_fn(f"[cache] index already present, skipping write: {index_path}")
        return {"status": "skipped_exists", "index_path": str(index_path)}

    cache_dir.mkdir(parents=True, exist_ok=True)
    log_fn(
        f"[cache] writing sharded BF16 dequant cache: dir={cache_dir} "
        f"target_shard_bytes={CACHE_SHARD_BYTES / 1e9:.1f} GB"
    )

    from safetensors.torch import save_file

    # state_dict returns references to model storage; iterating is cheap.
    state_items = list(transformer.state_dict().items())
    state_items_sorted = sorted(state_items, key=lambda kv: kv[0])
    log_fn(f"[cache] state_dict has {len(state_items_sorted)} entries")

    weight_map: Dict[str, str] = {}
    written_shards: List[Dict[str, Any]] = []
    current_shard: Dict[str, Any] = {}
    current_size = 0
    shard_id = 0
    total_size = 0
    t0 = time.perf_counter()

    def _flush_current() -> None:
        nonlocal shard_id, current_shard, current_size, total_size
        if not current_shard:
            return
        shard_id += 1
        shard_path = _cache_shard_path(cache_dir, rank, world_size, shard_id)
        try:
            save_file(current_shard, str(shard_path))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"shard {shard_id} write failed: {exc!r}") from exc
        sz = 0
        try:
            sz = shard_path.stat().st_size
        except OSError:
            pass
        for k in current_shard:
            weight_map[k] = shard_path.name
        written_shards.append({
            "shard_id": shard_id,
            "filename": shard_path.name,
            "tensor_count": len(current_shard),
            "size_bytes": sz,
        })
        total_size += sz
        log_fn(
            f"[cache]   shard {shard_id:05d} ({shard_path.name}): "
            f"tensors={len(current_shard)} size={sz / 1e9:.1f} GB"
        )
        # Drop kernel page cache for this shard so RSS doesn't accumulate.
        _drop_page_cache(shard_path, log_fn)
        # Clear our dict (tensors themselves are still owned by the model)
        # and force gc so any transient buffers from save_file's path are
        # reclaimed before the next shard.
        current_shard.clear()
        current_size = 0
        gc.collect()

    try:
        for key, tensor in state_items_sorted:
            # save_file refuses tensors with requires_grad=True; detach.
            t = tensor.detach()
            # Ensure contiguous storage covering exactly this tensor; the
            # dequant pass already clones ragged-padded slabs, but other
            # paths (e.g. buffers) may not.
            if not t.is_contiguous():
                t = t.contiguous()
            sz = _bytes_of(t)
            if current_size + sz > CACHE_SHARD_BYTES and current_shard:
                _flush_current()
            current_shard[key] = t
            current_size += sz
        _flush_current()
    except Exception as exc:  # noqa: BLE001
        log_fn(f"[cache] WARN: sharded write aborted: {exc!r}")
        return {
            "status": "write_failed",
            "error": repr(exc),
            "shards_written": written_shards,
        }

    elapsed = time.perf_counter() - t0
    log_fn(
        f"[cache] all {shard_id} shards written in {elapsed:.2f}s "
        f"({total_size / 1e9:.1f} GB)"
    )

    # Per-rank metadata sibling. Only fields that affect cache *correctness*
    # belong in this file; CPU cores / OMP threads are intentionally omitted
    # because they affect performance only.
    metadata = {
        "source_sharded_ckpt_path": (config.get("SHARDED_CKPT_PATH") or "").strip(),
        "model_args_config_path": (config.get("MODEL_ARGS_CONFIG_PATH") or "").strip(),
        "sharding_mode": (config.get("SHARDING_MODE") or "").strip().lower(),
        "dequant_scope": "all",
        "created_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rank": rank,
        "world_size": world_size,
        "dtype": "bf16",
        "shard_count": shard_id,
        "shard_layout": "sharded",
        "shards": written_shards,
        "total_size_bytes": total_size,
        "write_seconds": elapsed,
        "shard_bytes_target": CACHE_SHARD_BYTES,
    }
    try:
        cache_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        log_fn(f"[cache] metadata written: {cache_metadata}")
    except OSError as exc:
        log_fn(f"[cache] WARN: metadata write failed: {exc!r}")

    # Write the index LAST. Its presence is the cache-existence signal.
    # Compat-relevant fields only — CPU cores / OMP threads are intentionally
    # absent. The reader's _validate_cache_compat checks these against the
    # running config and hard-fails on topology mismatches.
    index_body = {
        "metadata": {
            "format": "pt",
            "rank": rank,
            "world_size": world_size,
            "sharding_mode": (config.get("SHARDING_MODE") or "").strip().lower() or None,
            "dtype": "bf16",
            "dequant_scope": "all",
            "shard_count": shard_id,
            "total_size_bytes": total_size,
        },
        "weight_map": weight_map,
    }
    try:
        # Write to a temp and rename for atomicity (best effort on Lustre).
        tmp_index = index_path.with_name(index_path.name + ".tmp")
        tmp_index.write_text(json.dumps(index_body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(str(tmp_index), str(index_path))
        log_fn(f"[cache] index written: {index_path} ({len(weight_map)} keys)")
    except OSError as exc:
        log_fn(f"[cache] WARN: index write failed: {exc!r}")
        return {
            "status": "index_write_failed",
            "error": repr(exc),
            "shards_written": written_shards,
        }

    return {
        "status": "written",
        "layout": "sharded",
        "index_path": str(index_path),
        "cache_metadata_file": str(cache_metadata),
        "shard_count": shard_id,
        "shards": written_shards,
        "total_size_bytes": total_size,
        "write_seconds": elapsed,
    }
