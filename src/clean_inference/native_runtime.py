"""Native CPU inference runtime: threads, distributed init, ModelArgs build, construction.

Ordering differs by SHARDING_MODE, and the difference is load-bearing:

  1. setup_thread_env  — torch thread count, default dtype, manual seed.
  2. initialize_distributed_before_construction  — TP modes only. Upstream
     ColumnParallel/RowParallel read the module-global world_size in __init__
     to compute per-rank shard shapes, so the process group must exist first.
     No-op for DP modes.
  3. build_modelargs_for_case  — load native ModelArgs JSON, override dtype +
     max_batch_size + max_seq_len from runtime context.
  4. construct_transformer  — model_module.Transformer(args).
  5. initialize_distributed_after_construction  — DP modes only. DP replicates
     the model, so construction must see world_size==1 and build every parallel
     layer at FULL size; initializing earlier would shard them and the
     unsharded mp1 checkpoint would no longer match. No-op for TP modes.
  6. assert_distributed_ready  — fail loudly if world_size>1 and no process
     group exists. Without it, forgetting either hook degrades silently into a
     single-rank run that produces wrong results at full speed.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from .model_config import load_native_modelargs_config, modelargs_from_native_config


PRECISION_TO_DTYPE = {"bf16": "bf16", "fp8": "fp8"}
PRECISION_TO_TORCH_DTYPE = {"bf16": torch.bfloat16, "fp8": torch.bfloat16}


def _fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def setup_thread_env(config: Dict[str, str], log_fn=print) -> Dict[str, Any]:
    """Apply OMP_NUM_THREADS and torch defaults. Mirrors legacy verify_cpu.py.

    Returns a small dict reporting the applied settings.
    """
    omp_threads = int(config.get("OMP_NUM_THREADS", "0") or 0)
    if omp_threads > 0:
        torch.set_num_threads(omp_threads)

    weights_precision = config.get("WEIGHTS_PRECISION", "bf16")
    torch_default_dtype = PRECISION_TO_TORCH_DTYPE.get(weights_precision, torch.bfloat16)
    torch.set_default_dtype(torch_default_dtype)

    torch.manual_seed(0)

    info = {
        "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS", ""),
        "omp_proc_bind_env": os.environ.get("OMP_PROC_BIND", ""),
        "omp_places_env": os.environ.get("OMP_PLACES", ""),
        "torch_num_threads": torch.get_num_threads(),
        "torch_default_dtype": str(torch.get_default_dtype()),
        "manual_seed": 0,
    }
    log_fn(f"[runtime] threads: torch.get_num_threads()={info['torch_num_threads']}")
    log_fn(f"[runtime] default dtype: {info['torch_default_dtype']}")
    log_fn(
        f"[runtime] OMP_NUM_THREADS={info['omp_num_threads_env']} "
        f"OMP_PROC_BIND={info['omp_proc_bind_env']} "
        f"OMP_PLACES={info['omp_places_env']}"
    )
    return info


def detect_distributed_env() -> Dict[str, Any]:
    """Read torchrun/srun-injected env. Returns rank/world/local + flags."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    master_addr = os.environ.get("MASTER_ADDR", "")
    master_port = os.environ.get("MASTER_PORT", "")
    return {
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "master_addr": master_addr,
        "master_port": master_port,
        "is_distributed": world_size > 1,
        "is_root": rank == 0,
    }


TP_MODES = ("tp2",)
DP_MODES = ("dp2", "dp2_epon")


def _init_gloo(dist_env: Dict[str, Any], when: str, mode: str, log_fn) -> bool:
    import torch.distributed as dist

    if not dist_env["is_distributed"]:
        log_fn("[runtime] world_size=1 -> skipping dist init")
        return False
    if dist.is_initialized():
        log_fn("[runtime] dist already initialized")
        return False
    log_fn(
        f"[runtime] dist.init_process_group('gloo') {when} construction "
        f"({mode}; world_size={dist_env['world_size']}, rank={dist_env['rank']})"
    )
    dist.init_process_group("gloo")
    return True


def initialize_distributed_before_construction(
    config: Dict[str, str], dist_env: Dict[str, Any], log_fn=print
) -> bool:
    """Init dist BEFORE Transformer(args). TP modes only.

    Upstream ColumnParallelLinear/RowParallelLinear read the module-global
    world_size in __init__ to compute per-rank shard shapes, so it must be
    visible before construction. No-op for DP modes.
    """
    mode = config.get("SHARDING_MODE", "").lower()
    if mode not in TP_MODES:
        log_fn(
            f"[runtime] sharding_mode={mode!r} is not a TP mode -> dist init "
            f"deferred to initialize_distributed_after_construction()"
        )
        return False
    return _init_gloo(dist_env, "BEFORE", mode, log_fn)


def initialize_distributed_after_construction(
    config: Dict[str, str], dist_env: Dict[str, Any], log_fn=print
) -> bool:
    """Init dist AFTER Transformer(args). DP modes only.

    DP replicates the model, so construction must see world_size==1 and build
    every parallel layer at FULL size. Initializing earlier would shard them
    and the unsharded mp1 checkpoint would no longer match. No-op for TP modes.
    """
    mode = config.get("SHARDING_MODE", "").lower()
    if mode not in DP_MODES:
        return False
    return _init_gloo(dist_env, "AFTER", mode, log_fn)


def assert_distributed_ready(dist_env: Dict[str, Any], log_fn=print) -> None:
    """Fail loudly if this is a multi-rank run with no process group.

    Without this, forgetting one of the two init hooks above degrades silently
    into a single-rank run that produces wrong results at full speed.
    """
    import torch.distributed as dist

    if not dist_env["is_distributed"]:
        return
    if not dist.is_initialized():
        _fail(
            f"world_size={dist_env['world_size']} but torch.distributed is NOT "
            f"initialized. Neither dist-init hook fired -- check SHARDING_MODE."
        )
    log_fn(
        f"[runtime] dist ready: backend={dist.get_backend()} "
        f"rank={dist.get_rank()}/{dist.get_world_size()}"
    )


def build_modelargs_for_case(
    model_module: Any,
    config: Dict[str, str],
    reference_case: Dict[str, Any],
    max_seq_len_pad: int = 0,
) -> Tuple[Any, Dict[str, Any], Path]:
    """Load native ModelArgs JSON and apply runtime overrides for one case.

    Overrides:
      dtype           = WEIGHTS_PRECISION -> ModelArgs literal
      max_batch_size  = reference_case["batch_size"]
      max_seq_len     = reference_case["lin_tokens"] + reference_case["lout_tokens"]
                        + max_seq_len_pad

    Returns (args, report, native_config_path).
    """
    clean_root = Path(config["CLEAN_ROOT"]).resolve()
    raw = config["MODEL_ARGS_CONFIG_PATH"]
    native_path = Path(raw)
    if not native_path.is_absolute():
        native_path = clean_root / native_path
    native_path = native_path.resolve()

    weights_precision = config["WEIGHTS_PRECISION"]
    dtype_override = PRECISION_TO_DTYPE.get(weights_precision)
    if dtype_override is None:
        _fail(
            f"WEIGHTS_PRECISION={weights_precision!r} has no dtype mapping; "
            f"expected one of {sorted(PRECISION_TO_DTYPE)}"
        )

    lin = int(reference_case["lin_tokens"])
    lout = int(reference_case["lout_tokens"])
    batch_size = int(reference_case["batch_size"])
    max_seq_len = lin + lout + max_seq_len_pad

    overrides = {
        "dtype": dtype_override,
        "max_batch_size": batch_size,
        "max_seq_len": max_seq_len,
    }

    native_config = load_native_modelargs_config(native_path)
    args, report = modelargs_from_native_config(model_module, native_config, overrides=overrides)
    report["native_config_path"] = str(native_path)
    return args, report, native_path


def construct_transformer(model_module: Any, args: Any, log_fn=print) -> Any:
    """Instantiate model_module.Transformer(args). No weight loading here."""
    log_fn(f"[runtime] constructing Transformer({type(args).__name__}) ...")
    sys.stdout.flush()
    transformer = model_module.Transformer(args)
    log_fn(f"[runtime] Transformer constructed: {type(transformer).__name__}")
    return transformer
