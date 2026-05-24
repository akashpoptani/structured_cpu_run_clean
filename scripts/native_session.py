#!/usr/bin/env python3
"""Session runner. Loads the model ONCE, then runs each child config's
RUN_MODE phase sequentially in the same torch.distributed process.

The child resolved-env paths are passed via repeated --child flags. Each
child carries its own TAG, RUN_MODE, LIN_TOKENS, LOUT_TOKENS, BATCH_SIZE,
and `reference_group` (only consumed by verify/both). The session sbatch
overrides DEQUANT_CACHE_MODE/PATH so the cache decision is made once.

All children must already share SHARDING_MODE, TP/DP/EP/PP, WEIGHTS_PRECISION,
SHARDED_CKPT_PATH, MODEL_ARGS_CONFIG_PATH, DEQUANT_FP8_WEIGHTS (verified by
submit_session.sh). The model envelope is the max(Lin+Lout, BS) across all
children.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
CLEAN_ROOT = SCRIPT_DIR.parent
if str(CLEAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEAN_ROOT))

# Reuse helpers from native_run; this is the cleanest way to avoid duplicating
# ~200 lines of setup. native_run.py is import-safe (main is guarded).
import native_run as nr

from src.clean_inference.config import parse_resolved_env, resolve_path
from src.clean_inference.imports import import_deepseek_modules
from src.clean_inference.model_config import summarize_modelargs
from src.clean_inference.native_runtime import (
    build_modelargs_for_case,
    construct_transformer,
    detect_distributed_env,
    initialize_distributed_if_needed,
    setup_thread_env,
)
from src.clean_inference.weight_loading import (
    load_cached_bf16_weights,
    load_weights_into_transformer,
    maybe_dequantize_fp8,
    maybe_write_dequant_cache,
    resolve_cache_plan,
)
from src.clean_inference.tokenization import load_tokenizer
from src.clean_inference.prompting import build_synthetic_cases, encode_len


def _fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def _load_child(path: Path) -> Dict[str, str]:
    if not path.is_file():
        _fail(f"child resolved-config not found: {path}")
    return parse_resolved_env(path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Native CPU TP-aware session runner.")
    p.add_argument("--session-tag", required=True)
    p.add_argument("--child", action="append", required=True,
                   help="Path to a child resolved-env (one per child). Repeatable.")
    p.add_argument("--dequant-cache-mode", default="off")
    p.add_argument("--dequant-cache-path", default="")
    p.add_argument("--reference-group", default="prompt1_bs1_lin10_lout15",
                   help="Reference group consumed by children whose RUN_MODE is verify or both.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    children_paths = [Path(p) for p in args.child]
    children = [_load_child(p) for p in children_paths]
    if not children:
        _fail("no children passed to session")

    session_tag = args.session_tag
    dist_env = detect_distributed_env()
    is_root = dist_env["is_root"]
    log = nr._make_logger(is_root)

    log(f"[session] SESSION_TAG: {session_tag}")
    log(f"[session] children ({len(children)}):")
    for cp, cfg in zip(children_paths, children):
        log(f"  - {cfg.get('TAG')} mode={cfg.get('RUN_MODE')} lin={cfg.get('LIN_TOKENS')} "
            f"lout={cfg.get('LOUT_TOKENS')} bs={cfg.get('BATCH_SIZE')}  [{cp}]")
    log(
        f"[session] dist env: rank={dist_env['rank']} world={dist_env['world_size']} "
        f"local_rank={dist_env['local_rank']} "
        f"master={dist_env['master_addr']}:{dist_env['master_port']}"
    )

    # First child acts as the "base" config for setup-shared fields. Session
    # cache settings override the base.
    base_config = dict(children[0])
    base_config["TAG"] = session_tag
    base_config["DEQUANT_CACHE_MODE"] = args.dequant_cache_mode
    base_config["DEQUANT_CACHE_PATH"] = args.dequant_cache_path

    # Envelope across all children.
    lin = max(int(c["LIN_TOKENS"]) for c in children)
    lout = max(int(c["LOUT_TOKENS"]) for c in children)
    bs = max(int(c["BATCH_SIZE"]) for c in children)
    log(f"[session] model envelope: lin={lin} lout={lout} bs={bs}")
    envelope = {"lin_tokens": lin, "lout_tokens": lout, "batch_size": bs}

    setup_thread_env(base_config, log_fn=log)
    initialize_distributed_if_needed(base_config, dist_env, log_fn=log)

    bundle = import_deepseek_modules(base_config)
    model_module = bundle["model"]
    log(f"[session] model.__file__ = {bundle['model_file']}")

    cache_plan = resolve_cache_plan(base_config, dist_env, log_fn=log)
    if cache_plan["override_modelargs_dtype"]:
        log(f"[session] cache plan overrides ModelArgs.dtype -> {cache_plan['override_modelargs_dtype']}")

    margs, modelargs_report, native_path = build_modelargs_for_case(
        model_module, base_config, envelope, max_seq_len_pad=0
    )
    if cache_plan["override_modelargs_dtype"]:
        new_dtype = cache_plan["override_modelargs_dtype"]
        margs.dtype = new_dtype
        margs.scale_fmt = None
        modelargs_report["overridden"].append({"field": "dtype", "value": new_dtype, "source": "cache_plan"})
    log(f"[session] ModelArgs source: native {native_path.name}")
    summary = summarize_modelargs(margs)
    for k in ("n_layers", "dim", "n_heads", "n_routed_experts", "n_activated_experts",
              "max_seq_len", "max_batch_size", "dtype", "scale_fmt"):
        log(f"[session]   {k} = {summary[k]}")

    transformer = construct_transformer(model_module, margs, log_fn=log)
    try:
        total_params = sum(p.numel() for p in transformer.parameters())
        log(f"[session] total parameters (numel sum): {total_params:,}")
    except Exception as exc:
        log(f"[session] total parameters: <unavailable: {exc!r}>")

    # ----- weight load (BF16 cache OR FP8+dequant+maybe write) -----
    cache_read_report = None
    load_report = None
    dequant_report = None
    cache_write_report = None
    if cache_plan["do_read_cache"]:
        cache_read_report = load_cached_bf16_weights(transformer, cache_plan, dist_env, log_fn=log)
    else:
        load_report = load_weights_into_transformer(transformer, base_config, dist_env, log_fn=log)
        dequant_report = maybe_dequantize_fp8(transformer, base_config, log_fn=log)
        cache_write_report = maybe_write_dequant_cache(transformer, cache_plan, base_config, dist_env, log_fn=log)

    # ----- tokenizer once -----
    tokenizer, tokenizer_dir = load_tokenizer(base_config, log_fn=log)

    shared = {
        "load_report": load_report,
        "dequant_report": dequant_report,
        "cache_read_report": cache_read_report,
        "cache_write_report": cache_write_report,
        "tokenizer_dir": str(tokenizer_dir),
    }
    session_modelargs_summary = {k: str(v) for k, v in summary.items()}

    # ----- iterate children -----
    session_results: List[Dict[str, Any]] = []
    session_start = time.perf_counter()
    overall_rc = 0
    for cfg in children:
        child_tag = cfg["TAG"]
        run_mode = cfg["RUN_MODE"].strip().lower()
        log(f"[session] >>>> child={child_tag} run_mode={run_mode} <<<<")

        base_result = {
            "tag": child_tag,
            "run_mode": run_mode,
            "session_tag": session_tag,
            "reference_group": args.reference_group if run_mode in ("verify", "both") else None,
            "case_id_filter": None,
            "modelargs_report": modelargs_report,
            "modelargs_summary": session_modelargs_summary,
            "cache_plan": {
                "mode": cache_plan["mode"],
                "cache_dir": str(cache_plan["cache_dir"]) if cache_plan["cache_dir"] else None,
                "cache_file": str(cache_plan["cache_file"]) if cache_plan["cache_file"] else None,
                "cache_existed_at_plan": cache_plan["cache_exists"],
                "do_read_cache": cache_plan["do_read_cache"],
                "do_write_cache": cache_plan["do_write_cache"],
            },
            "synthetic_plan": (
                {"lin_tokens": int(cfg["LIN_TOKENS"]),
                 "lout_tokens": int(cfg["LOUT_TOKENS"]),
                 "batch_size": int(cfg["BATCH_SIZE"])}
                if run_mode in ("generate", "bench", "both") else None
            ),
        }

        try:
            child_rc, child_payload = _run_child(
                cfg, run_mode, transformer, tokenizer, base_result, shared,
                is_root, log, args.reference_group,
            )
        except SystemExit as exc:
            log(f"[session] child {child_tag} FAILED via SystemExit({exc.code!r})")
            child_rc, child_payload = (int(exc.code) if isinstance(exc.code, int) else 1), {"error": "SystemExit"}
        except Exception as exc:  # noqa: BLE001
            log(f"[session] child {child_tag} FAILED via exception: {type(exc).__name__}: {exc}")
            child_rc, child_payload = 1, {"error": f"{type(exc).__name__}: {exc}"}

        session_results.append({
            "tag": child_tag,
            "run_mode": run_mode,
            "rc": child_rc,
            **child_payload,
        })
        if child_rc != 0:
            overall_rc = child_rc

    session_elapsed = time.perf_counter() - session_start
    session_summary = {
        "session_tag": session_tag,
        "session_elapsed_seconds": session_elapsed,
        "children": session_results,
        "modelargs_summary": session_modelargs_summary,
        "load_report": load_report,
        "dequant_report": dequant_report,
        "cache_read_report": cache_read_report,
        "cache_write_report": cache_write_report,
        "tokenizer_dir": str(tokenizer_dir),
        "rc": overall_rc,
    }
    out_dir = Path(base_config["CLEAN_ROOT"]).resolve() / base_config["OUTPUT_ROOT"] / "results" / session_tag
    if is_root:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "session_results.json"
        out_path.write_text(json.dumps(session_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        log(f"[session] session summary JSON: {out_path}")

    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass
    return overall_rc


def _run_child(
    cfg: Dict[str, str],
    run_mode: str,
    transformer,
    tokenizer,
    base_result: Dict[str, Any],
    shared: Dict[str, Any],
    is_root: bool,
    log,
    reference_group: str,
):
    """Run the per-mode decode phase for one child config. Returns (rc, payload)."""
    tag = cfg["TAG"]
    lin = int(cfg["LIN_TOKENS"])
    lout = int(cfg["LOUT_TOKENS"])
    bs = int(cfg["BATCH_SIZE"])

    if run_mode == "verify":
        ref_cases = nr._enumerate_reference_cases(cfg, reference_group)
        v = nr._run_verify_phase(
            transformer, tokenizer, ref_cases,
            base_result, shared, cfg, is_root, log,
            nr.RUN_MODE_RESULT_FILENAME["verify"],
        )
        return (0 if v["passed"] else 1), {"out_path": str(v["out_path"]),
                                            "passed_cases": v["passed_cases"],
                                            "failed_cases": v["failed_cases"]}

    if run_mode == "generate":
        synth = build_synthetic_cases(tokenizer, lin, lout, bs, tag)
        log(f"[session]   synth lin_actual={encode_len(tokenizer, synth[0]['prompt_text'])}")
        out = nr._run_generate_phase(
            transformer, tokenizer, synth,
            base_result, shared, cfg, is_root, log,
            nr.RUN_MODE_RESULT_FILENAME["generate"],
        )
        return 0, {"out_path": str(out["out_path"])}

    if run_mode == "bench":
        synth = build_synthetic_cases(tokenizer, lin, lout, bs, tag)
        log(f"[session]   synth lin_actual={encode_len(tokenizer, synth[0]['prompt_text'])}")
        out = nr._run_bench_phase(
            transformer, tokenizer, synth,
            base_result, shared, cfg, is_root, log,
            nr.RUN_MODE_RESULT_FILENAME["bench"],
        )
        return 0, {"out_path": str(out["out_path"]), "aggregate": out["aggregate"]}

    if run_mode == "both":
        ref_cases = nr._enumerate_reference_cases(cfg, reference_group)
        verify_base = dict(base_result)
        verify_base["total_cases"] = len(ref_cases)
        v = nr._run_verify_phase(
            transformer, tokenizer, ref_cases,
            verify_base, shared, cfg, is_root, log,
            nr.RUN_MODE_RESULT_FILENAME["verify"],
        )
        both_summary = dict(base_result)
        both_summary["verify_prompt_source"] = "reference_cases"
        both_summary["bench_prompt_source"] = "synthetic_exact_prompt"
        both_summary["verify_passed"] = v["passed"]
        both_summary["verify_path"] = str(v["out_path"])
        both_summary["verify_passed_cases"] = v["passed_cases"]
        both_summary["verify_failed_cases"] = v["failed_cases"]

        if not v["passed"]:
            both_summary["status"] = "verify_failed"
            both_summary["bench_path"] = None
            nr._write_result(cfg, both_summary, is_root, nr.RUN_MODE_RESULT_FILENAME["both"])
            return 1, {"verify_path": str(v["out_path"]), "bench_path": None}

        synth = build_synthetic_cases(tokenizer, lin, lout, bs, tag)
        bench_base = dict(base_result)
        bench_base["total_cases"] = len(synth)
        b = nr._run_bench_phase(
            transformer, tokenizer, synth,
            bench_base, shared, cfg, is_root, log,
            nr.RUN_MODE_RESULT_FILENAME["bench"],
        )
        both_summary["status"] = "both_passed"
        both_summary["bench_path"] = str(b["out_path"])
        both_summary["bench_aggregate"] = b["aggregate"]
        nr._write_result(cfg, both_summary, is_root, nr.RUN_MODE_RESULT_FILENAME["both"])
        return 0, {
            "verify_path": str(v["out_path"]),
            "bench_path": str(b["out_path"]),
            "aggregate": b["aggregate"],
        }

    _fail(f"unsupported child run_mode={run_mode!r}")
    return 1, {"error": f"unsupported run_mode={run_mode!r}"}


if __name__ == "__main__":
    raise SystemExit(main())
