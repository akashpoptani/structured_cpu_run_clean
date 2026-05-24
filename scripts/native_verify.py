#!/usr/bin/env python3
"""Native CPU TP-aware runner. Despite the historical name, this script
dispatches on the resolved-config RUN_MODE and supports:

  RUN_MODE=verify    -> greedy-decode each case, compare against
                        expected_output_token_ids, write
                        native_verify_results.json (token-exact PASS/FAIL).
  RUN_MODE=generate  -> greedy-decode each case, do not compare; write
                        native_generate_results.json (tokens + decoded text).
  RUN_MODE=bench     -> greedy-decode each case, emit timing/throughput
                        metrics (TTFT, TPOT, tokens/sec); write
                        native_bench_results.json. No correctness check.
  RUN_MODE=both      -> verify first; if any case fails, write verify JSON
                        and exit 1 without benchmarking. If verify passes,
                        re-use the same case timings to write bench JSON and
                        write native_both_results.json with paths to both.

Setup (dist init, Transformer construction, weight load, optional dequant,
tokenizer load) is identical across modes and only runs once per job.

Safety flags (apply regardless of RUN_MODE):
  --no-load-weights : construct only; skip weight load and the decode loop.
                      Safe on a login node.
  --no-generate     : load weights but skip decode. Useful as a Slurm
                      weight-load smoke before a real run.

Both flags write a result JSON that records what was skipped.

Without those flags the full pipeline runs; that should be launched inside
a Slurm allocation via scripts/run_native_distributed.sh.
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

from src.clean_inference.config import parse_resolved_env, require_config_keys, resolve_path
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
    load_weights_into_transformer,
    maybe_dequantize_fp8,
)
from src.clean_inference.tokenization import encode_prompt, load_tokenizer, try_decode
from src.clean_inference.generation import greedy_decode

from inspect_reference_cases import load_case


REQUIRED_KEYS = (
    "TAG",
    "CLEAN_ROOT",
    "DEEPSEEK_REPO",
    "ACTIVE_MODEL_PATH",
    "WEIGHTS_PRECISION",
    "MODEL_ARGS_CONFIG_PATH",
    "SHARDING_MODE",
    "RUN_MODE",
    "GPU_REFERENCE_PATH",
    "OUTPUT_ROOT",
)

RUN_MODE_RESULT_FILENAME = {
    "verify": "native_verify_results.json",
    "generate": "native_generate_results.json",
    "bench": "native_bench_results.json",
    "both": "native_both_results.json",
}


def _fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def _reference_group_dir(config: Dict[str, str], group: str) -> Path:
    clean_root = Path(config["CLEAN_ROOT"]).resolve()
    reference_root = resolve_path(clean_root, config["GPU_REFERENCE_PATH"])
    group_dir = reference_root / group
    if not group_dir.is_dir():
        _fail(f"reference group dir not found: {group_dir}")
    return group_dir


def _resolve_reference_case(config: Dict[str, str], group: str, case_id: str) -> Dict[str, Any]:
    case_path = _reference_group_dir(config, group) / f"{case_id}.json"
    if not case_path.is_file():
        _fail(f"reference case not found: {case_path}")
    return load_case(case_path)


def _enumerate_cases(config: Dict[str, str], group: str) -> List[Dict[str, Any]]:
    """Return all reference cases in the group sorted by filename."""
    group_dir = _reference_group_dir(config, group)
    case_paths = sorted(group_dir.glob("*.json"))
    if not case_paths:
        _fail(f"no *.json cases in {group_dir}")
    return [load_case(p) for p in case_paths]


def _envelope_case(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Synthesize a `case`-shaped dict whose lin/lout/batch envelope every case.

    Used to size the model's max_seq_len / max_batch_size up front so a single
    construction can host all cases in the group.
    """
    max_lin = max(int(c["lin_tokens"]) for c in cases)
    max_lout = max(int(c["lout_tokens"]) for c in cases)
    max_bs = max(int(c["batch_size"]) for c in cases)
    return {
        "lin_tokens": max_lin,
        "lout_tokens": max_lout,
        "batch_size": max_bs,
    }


def _make_logger(is_root: bool):
    if is_root:
        def log(*args, **kwargs):
            print(*args, **kwargs, flush=True)
        return log
    def silent(*_args, **_kwargs):
        return None
    return silent


def _result_dir(config: Dict[str, str]) -> Path:
    clean_root = Path(config["CLEAN_ROOT"]).resolve()
    output_root = resolve_path(clean_root, config["OUTPUT_ROOT"])
    return output_root / "results" / config["TAG"]


def _write_result(
    config: Dict[str, str],
    result: Dict[str, Any],
    is_root: bool,
    filename: str = "native_verify_results.json",
) -> Path:
    out_path = _result_dir(config) / filename
    if is_root:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _first_mismatch(expected: List[int], generated: List[int]) -> int:
    for i, (e, g) in enumerate(zip(expected, generated)):
        if e != g:
            return i
    if len(expected) != len(generated):
        return min(len(expected), len(generated))
    return -1


def _decode_one_case(
    transformer,
    tokenizer,
    case: Dict[str, Any],
    log_fn,
) -> Dict[str, Any]:
    """Tokenize + greedy decode a single case. Mode-agnostic.

    Returns a base per-case dict (case_id, prompt info, generated tokens,
    decoded text, timing). Verify scoring is added separately by
    _score_verify(). The reference's expected_output_token_ids are *carried
    through* in the returned dict for diagnostics regardless of mode.
    """
    case_id = case["case_id"]
    prompt_text = case["prompt_text"]
    prompt_tokens = encode_prompt(tokenizer, prompt_text)
    log_fn(f"[case {case_id}] prompt: {prompt_text!r}")
    log_fn(f"[case {case_id}] prompt tokens ({len(prompt_tokens)}): {prompt_tokens}")

    lout = int(case["lout_tokens"])
    expected = list(case.get("expected_output_token_ids", []))

    t0 = time.perf_counter()
    generated, prefill_s, decode_s = greedy_decode(transformer, prompt_tokens, lout, log_fn=log_fn)
    elapsed = time.perf_counter() - t0

    log_fn(f"[case {case_id}] generated tokens ({len(generated)}): {generated}")
    decoded = try_decode(tokenizer, generated)
    if decoded is not None:
        log_fn(f"[case {case_id}] decoded text: {decoded!r}")

    return {
        "case_id": case_id,
        "prompt_text": prompt_text,
        "prompt_tokens": prompt_tokens,
        "prompt_token_count": len(prompt_tokens),
        "expected_tokens": expected,
        "reference_expected_tokens_available": bool(expected),
        "generated_tokens": generated,
        "decoded_text": decoded,
        "timing": {
            "prefill_seconds": prefill_s,
            "decode_seconds_total": decode_s,
            "total_seconds": elapsed,
        },
    }


def _score_verify(case_result: Dict[str, Any], log_fn) -> Dict[str, Any]:
    """Add token-exact pass/fail fields to a base case result."""
    case_id = case_result["case_id"]
    expected = case_result["expected_tokens"]
    generated = case_result["generated_tokens"]

    log_fn(f"[case {case_id}] expected tokens  ({len(expected)}): {expected}")
    mismatch_index = _first_mismatch(expected, generated)
    passed = mismatch_index == -1 and len(generated) == len(expected) and len(expected) > 0
    log_fn(
        f"[case {case_id}] {'PASS' if passed else 'FAIL'} "
        f"(first mismatch index={mismatch_index})"
    )
    case_result["passed"] = passed
    case_result["first_mismatch_index"] = mismatch_index
    return case_result


def _bench_stats_for_case(case_result: Dict[str, Any]) -> Dict[str, Any]:
    """Derive TTFT / TPOT / throughput from a decoded case result.

    TTFT  = prefill_seconds. The first output token is emitted at the end of
            the prefill forward pass, so this is "time to first emitted token".
    TPOT  = decode_seconds_total / (lout - 1). Average inter-token latency
            across the lout-1 single-token decode steps. Undefined when
            lout == 1.
    tokens_per_second = lout / (prefill_seconds + decode_seconds_total).
    """
    t = case_result["timing"]
    lout = len(case_result["generated_tokens"])
    prefill_s = float(t["prefill_seconds"])
    decode_s = float(t["decode_seconds_total"])
    total_s = float(t["total_seconds"])
    decode_steps = max(lout - 1, 0)
    tpot_seconds = (decode_s / decode_steps) if decode_steps > 0 else None
    tokens_per_second = (lout / total_s) if total_s > 0 else None
    return {
        "case_id": case_result["case_id"],
        "lout_tokens": lout,
        "prompt_token_count": case_result["prompt_token_count"],
        "ttft_seconds": prefill_s,
        "tpot_seconds": tpot_seconds,
        "decode_seconds_total": decode_s,
        "total_seconds": total_s,
        "tokens_per_second": tokens_per_second,
    }


def _aggregate_bench(case_bench: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-case bench stats across the group."""
    if not case_bench:
        return {"case_count": 0}
    ttft = [c["ttft_seconds"] for c in case_bench]
    tpot = [c["tpot_seconds"] for c in case_bench if c["tpot_seconds"] is not None]
    tps = [c["tokens_per_second"] for c in case_bench if c["tokens_per_second"] is not None]
    return {
        "case_count": len(case_bench),
        "ttft_seconds_mean": sum(ttft) / len(ttft),
        "ttft_seconds_min": min(ttft),
        "ttft_seconds_max": max(ttft),
        "tpot_seconds_mean": (sum(tpot) / len(tpot)) if tpot else None,
        "tokens_per_second_mean": (sum(tps) / len(tps)) if tps else None,
    }


def run(
    resolved_config_path: Path,
    reference_group: str,
    case_id: Optional[str],
    no_load_weights: bool,
    no_generate: bool,
) -> int:
    config = parse_resolved_env(resolved_config_path)
    require_config_keys(config, REQUIRED_KEYS)

    run_mode = config["RUN_MODE"].strip().lower()
    if run_mode not in RUN_MODE_RESULT_FILENAME:
        _fail(f"unsupported RUN_MODE={run_mode!r}; expected one of {sorted(RUN_MODE_RESULT_FILENAME)}")

    dist_env = detect_distributed_env()
    is_root = dist_env["is_root"]
    log = _make_logger(is_root)

    log(f"[native-verify] resolved_config: {resolved_config_path}")
    log(f"[native-verify] TAG: {config['TAG']}")
    log(f"[native-verify] RUN_MODE: {run_mode}")
    log(
        f"[native-verify] dist env: rank={dist_env['rank']} world={dist_env['world_size']} "
        f"local_rank={dist_env['local_rank']} "
        f"master={dist_env['master_addr']}:{dist_env['master_port']}"
    )

    if case_id:
        cases = [_resolve_reference_case(config, reference_group, case_id)]
    else:
        cases = _enumerate_cases(config, reference_group)
    log(
        f"[native-verify] reference group: {reference_group} "
        f"({len(cases)} case{'s' if len(cases) != 1 else ''})"
    )
    for c in cases:
        log(
            f"[native-verify]   case {c['case_id']}: "
            f"lin={c['lin_tokens']} lout={c['lout_tokens']} bs={c['batch_size']}"
        )

    setup_thread_env(config, log_fn=log)
    initialize_distributed_if_needed(config, dist_env, log_fn=log)

    bundle = import_deepseek_modules(config)
    model_module = bundle["model"]
    log(f"[native-verify] model.__file__ = {bundle['model_file']}")

    # Size the model envelope to host every case in the group.
    envelope = _envelope_case(cases)
    args, modelargs_report, native_path = build_modelargs_for_case(
        model_module, config, envelope, max_seq_len_pad=0
    )
    log(f"[native-verify] ModelArgs source: native {native_path.name}")
    log(f"[native-verify]   Native config path: {native_path}")
    log(
        "[native-verify]   Applied overrides: "
        + ", ".join(f"{e['field']}={e['value']}" for e in modelargs_report["overridden"])
    )
    log(f"[native-verify]   Native config fields ({len(modelargs_report['native_fields'])}): "
        f"{modelargs_report['native_fields']}")
    log(f"[native-verify]   Defaulted: {modelargs_report['defaulted']}")
    summary = summarize_modelargs(args)
    for k in ("n_layers", "dim", "n_heads", "n_routed_experts", "n_activated_experts",
              "max_seq_len", "max_batch_size", "dtype", "scale_fmt"):
        log(f"[native-verify]   {k} = {summary[k]}")

    transformer = construct_transformer(model_module, args, log_fn=log)
    try:
        total_params = sum(p.numel() for p in transformer.parameters())
        log(f"[native-verify] total parameters (numel sum): {total_params:,}")
    except Exception as exc:
        log(f"[native-verify] total parameters: <unavailable: {exc!r}>")

    base_result: Dict[str, Any] = {
        "tag": config["TAG"],
        "run_mode": run_mode,
        "reference_group": reference_group,
        "case_id_filter": case_id,
        "total_cases": len(cases),
        "modelargs_report": modelargs_report,
        "modelargs_summary": {k: str(v) for k, v in summary.items()},
    }

    planned_cases = [
        {
            "case_id": c["case_id"],
            "planned": True,
            "lin_tokens": c["lin_tokens"],
            "lout_tokens": c["lout_tokens"],
            "batch_size": c["batch_size"],
        }
        for c in cases
    ]

    if no_load_weights:
        log(
            "[native-verify] --no-load-weights set; constructing once and reporting "
            "planned cases without loading weights. Exiting 0."
        )
        result = dict(base_result)
        result["status"] = "construct_only"
        result["cases"] = planned_cases
        result["skipped"] = ["weight_load", "tokenization", "generation"]
        _write_result(config, result, is_root, RUN_MODE_RESULT_FILENAME[run_mode])
        return 0

    load_report = load_weights_into_transformer(transformer, config, dist_env, log_fn=log)
    dequant_report = maybe_dequantize_fp8(transformer, config, log_fn=log)

    if no_generate:
        log(
            "[native-verify] --no-generate set; weights loaded. "
            f"Reporting {len(cases)} skipped case(s). Exiting 0."
        )
        result = dict(base_result)
        result["status"] = "weights_loaded_no_generation"
        result["load_report"] = load_report
        result["dequant_report"] = dequant_report
        result["cases"] = planned_cases
        _write_result(config, result, is_root, RUN_MODE_RESULT_FILENAME[run_mode])
        return 0

    tokenizer, tokenizer_dir = load_tokenizer(config, log_fn=log)

    # Decode every case once. All four modes share this loop; mode-specific
    # post-processing (verify scoring / bench aggregation / both summary)
    # happens after.
    case_results: List[Dict[str, Any]] = []
    overall_start = time.perf_counter()
    for case in cases:
        log(f"[native-verify] >>> running case {case['case_id']} <<<")
        case_results.append(_decode_one_case(transformer, tokenizer, case, log))
    overall_seconds = time.perf_counter() - overall_start

    shared: Dict[str, Any] = {
        "load_report": load_report,
        "dequant_report": dequant_report,
        "tokenizer_dir": str(tokenizer_dir),
        "timing_overall_seconds": overall_seconds,
    }

    rc = _dispatch_run_mode(
        run_mode, base_result, shared, case_results, config, is_root, log
    )

    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass
    return rc


def _dispatch_run_mode(
    run_mode: str,
    base_result: Dict[str, Any],
    shared: Dict[str, Any],
    case_results: List[Dict[str, Any]],
    config: Dict[str, str],
    is_root: bool,
    log,
) -> int:
    """Write mode-appropriate result JSON(s) and return the process exit code."""
    if run_mode == "verify":
        return _run_verify_dispatch(base_result, shared, case_results, config, is_root, log)
    if run_mode == "generate":
        return _run_generate_dispatch(base_result, shared, case_results, config, is_root, log)
    if run_mode == "bench":
        return _run_bench_dispatch(base_result, shared, case_results, config, is_root, log)
    if run_mode == "both":
        return _run_both_dispatch(base_result, shared, case_results, config, is_root, log)
    _fail(f"unreachable: RUN_MODE={run_mode!r}")
    return 1


def _run_verify_dispatch(
    base_result: Dict[str, Any],
    shared: Dict[str, Any],
    case_results: List[Dict[str, Any]],
    config: Dict[str, str],
    is_root: bool,
    log,
) -> int:
    passed_count = 0
    failed_count = 0
    for cr in case_results:
        _score_verify(cr, log)
        if cr["passed"]:
            passed_count += 1
        else:
            failed_count += 1
    log(
        f"[native-verify] verify summary: total={len(case_results)} "
        f"passed={passed_count} failed={failed_count} "
        f"in {shared['timing_overall_seconds']:.2f}s"
    )
    result = dict(base_result)
    result.update(shared)
    result["status"] = "verified" if failed_count == 0 else "mismatch"
    result["passed_cases"] = passed_count
    result["failed_cases"] = failed_count
    result["cases"] = case_results
    out_path = _write_result(config, result, is_root, RUN_MODE_RESULT_FILENAME["verify"])
    log(f"[native-verify] result JSON: {out_path}")
    return 0 if failed_count == 0 else 1


def _run_generate_dispatch(
    base_result: Dict[str, Any],
    shared: Dict[str, Any],
    case_results: List[Dict[str, Any]],
    config: Dict[str, str],
    is_root: bool,
    log,
) -> int:
    log(
        f"[native-verify] generate summary: total={len(case_results)} cases "
        f"in {shared['timing_overall_seconds']:.2f}s"
    )
    result = dict(base_result)
    result.update(shared)
    result["status"] = "generated"
    result["cases"] = case_results
    out_path = _write_result(config, result, is_root, RUN_MODE_RESULT_FILENAME["generate"])
    log(f"[native-verify] result JSON: {out_path}")
    return 0


def _run_bench_dispatch(
    base_result: Dict[str, Any],
    shared: Dict[str, Any],
    case_results: List[Dict[str, Any]],
    config: Dict[str, str],
    is_root: bool,
    log,
) -> int:
    bench_cases = [_bench_stats_for_case(cr) for cr in case_results]
    agg = _aggregate_bench(bench_cases)
    log(
        f"[native-verify] bench summary: total={len(case_results)} cases "
        f"in {shared['timing_overall_seconds']:.2f}s; "
        f"ttft_mean={agg.get('ttft_seconds_mean')} "
        f"tpot_mean={agg.get('tpot_seconds_mean')} "
        f"tps_mean={agg.get('tokens_per_second_mean')}"
    )
    result = dict(base_result)
    result.update(shared)
    result["status"] = "benchmarked"
    result["cases"] = case_results
    result["bench_per_case"] = bench_cases
    result["bench_aggregate"] = agg
    out_path = _write_result(config, result, is_root, RUN_MODE_RESULT_FILENAME["bench"])
    log(f"[native-verify] result JSON: {out_path}")
    return 0


def _run_both_dispatch(
    base_result: Dict[str, Any],
    shared: Dict[str, Any],
    case_results: List[Dict[str, Any]],
    config: Dict[str, str],
    is_root: bool,
    log,
) -> int:
    # Score verify first.
    passed_count = 0
    failed_count = 0
    for cr in case_results:
        _score_verify(cr, log)
        if cr["passed"]:
            passed_count += 1
        else:
            failed_count += 1

    verify_result = dict(base_result)
    verify_result.update(shared)
    verify_result["status"] = "verified" if failed_count == 0 else "mismatch"
    verify_result["passed_cases"] = passed_count
    verify_result["failed_cases"] = failed_count
    verify_result["cases"] = case_results
    verify_path = _write_result(
        config, verify_result, is_root, RUN_MODE_RESULT_FILENAME["verify"]
    )
    log(f"[native-verify] (both) verify JSON: {verify_path}")

    if failed_count != 0:
        log(
            f"[native-verify] (both) verify FAILED (passed={passed_count} "
            f"failed={failed_count}); skipping bench. Exiting 1."
        )
        summary = {
            **base_result,
            "status": "verify_failed",
            "verify_passed": False,
            "verify_path": str(verify_path),
            "bench_path": None,
        }
        _write_result(config, summary, is_root, RUN_MODE_RESULT_FILENAME["both"])
        return 1

    # verify passed; reuse decoded timings for bench.
    bench_cases = [_bench_stats_for_case(cr) for cr in case_results]
    agg = _aggregate_bench(bench_cases)
    bench_result = dict(base_result)
    bench_result.update(shared)
    bench_result["status"] = "benchmarked"
    bench_result["cases"] = case_results
    bench_result["bench_per_case"] = bench_cases
    bench_result["bench_aggregate"] = agg
    bench_path = _write_result(
        config, bench_result, is_root, RUN_MODE_RESULT_FILENAME["bench"]
    )
    log(f"[native-verify] (both) bench JSON: {bench_path}")

    summary = {
        **base_result,
        "status": "both_passed",
        "verify_passed": True,
        "verify_path": str(verify_path),
        "bench_path": str(bench_path),
        "passed_cases": passed_count,
        "failed_cases": failed_count,
        "bench_aggregate": agg,
    }
    out_path = _write_result(config, summary, is_root, RUN_MODE_RESULT_FILENAME["both"])
    log(f"[native-verify] (both) summary JSON: {out_path}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Native CPU TP-aware verification.")
    p.add_argument("--resolved-config", required=True)
    p.add_argument("--reference-group", required=True)
    p.add_argument(
        "--case-id",
        default=None,
        help="Optional. If omitted, every *.json in the reference group is run, sorted by filename.",
    )
    p.add_argument("--no-load-weights", action="store_true",
                   help="Construct only; skip weight load and generation.")
    p.add_argument("--no-generate", action="store_true",
                   help="Load weights but skip the decode loop.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    return run(
        Path(args.resolved_config),
        args.reference_group,
        args.case_id,
        args.no_load_weights,
        args.no_generate,
    )


if __name__ == "__main__":
    raise SystemExit(main())
