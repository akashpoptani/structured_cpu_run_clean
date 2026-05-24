#!/usr/bin/env python3
"""Native CPU TP-aware runner. Dispatches on the resolved-config RUN_MODE.

Case sources by mode
--------------------
  verify   -> reference JSON cases under
              GPU_REFERENCE_PATH/<reference_group>/*.json.
              Generated tokens are compared against expected_output_token_ids.
  generate -> synthetic exact-token prompts built from LIN_TOKENS / LOUT_TOKENS
              / BATCH_SIZE via src.clean_inference.prompting.
  bench    -> same synthetic prompts as generate.
  both     -> verify on reference cases first; if every case passes, run a
              second decode pass on synthetic prompts (Lin/Lout from config)
              and report bench metrics. Two distinct decode passes.

Setup (dist init, Transformer construction, weight load, optional dequant
or BF16-cache read, tokenizer load) runs once per process; per-mode
dispatch follows.

Safety flags (apply regardless of RUN_MODE):
  --no-load-weights : construct only; skip weight load and the decode loop.
                      Safe on a login node.
  --no-generate     : load weights but skip decode. Useful as a Slurm
                      weight-load smoke before a real run.

Both flags write a result JSON that records what was skipped (file naming
follows RUN_MODE_RESULT_FILENAME).
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
    load_cached_bf16_weights,
    load_weights_into_transformer,
    maybe_dequantize_fp8,
    maybe_write_dequant_cache,
    resolve_cache_plan,
)
from src.clean_inference.tokenization import encode_prompt, load_tokenizer, try_decode
from src.clean_inference.generation import greedy_decode
from src.clean_inference.prompting import build_synthetic_cases, encode_len

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
    "LIN_TOKENS",
    "LOUT_TOKENS",
    "BATCH_SIZE",
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


# ---------- reference / synthetic case loading ----------

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


def _enumerate_reference_cases(config: Dict[str, str], group: str) -> List[Dict[str, Any]]:
    group_dir = _reference_group_dir(config, group)
    case_paths = sorted(group_dir.glob("*.json"))
    if not case_paths:
        _fail(f"no *.json cases in {group_dir}")
    return [load_case(p) for p in case_paths]


def _envelope_lin_lout_bs(cases: List[Dict[str, Any]]) -> Dict[str, int]:
    """Maximum (lin, lout, bs) across a case list — used to size the model."""
    return {
        "lin_tokens": max(int(c["lin_tokens"]) for c in cases),
        "lout_tokens": max(int(c["lout_tokens"]) for c in cases),
        "batch_size": max(int(c["batch_size"]) for c in cases),
    }


def _planned_cases_summary(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "case_id": c["case_id"],
            "planned": True,
            "lin_tokens": int(c["lin_tokens"]),
            "lout_tokens": int(c["lout_tokens"]),
            "batch_size": int(c["batch_size"]),
            "prompt_source": c.get("source", {}).get("kind", "unknown"),
        }
        for c in cases
    ]


# ---------- logging / result writers ----------

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
    filename: str,
) -> Path:
    out_path = _result_dir(config) / filename
    if is_root:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


# ---------- per-case decode + scoring ----------

def _first_mismatch(expected: List[int], generated: List[int]) -> int:
    for i, (e, g) in enumerate(zip(expected, generated)):
        if e != g:
            return i
    if len(expected) != len(generated):
        return min(len(expected), len(generated))
    return -1


def _decode_one_case(transformer, tokenizer, case: Dict[str, Any], log_fn) -> Dict[str, Any]:
    """Tokenize + greedy decode a single case. Mode-agnostic.

    Carries the reference's expected_output_token_ids through (when present)
    for diagnostics regardless of mode.
    """
    case_id = case["case_id"]
    prompt_text = case["prompt_text"]
    prompt_tokens = encode_prompt(tokenizer, prompt_text)
    log_fn(f"[case {case_id}] prompt: {prompt_text[:120]!r}{'...' if len(prompt_text) > 120 else ''}")
    log_fn(f"[case {case_id}] prompt token count: {len(prompt_tokens)}")

    lout = int(case["lout_tokens"])
    raw_expected = case.get("expected_output_token_ids") or []
    expected = list(raw_expected) if raw_expected else []

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
    expected = case_result["expected_tokens"]
    generated = case_result["generated_tokens"]
    log_fn(f"[case {case_result['case_id']}] expected tokens  ({len(expected)}): {expected}")
    mismatch_index = _first_mismatch(expected, generated)
    passed = mismatch_index == -1 and len(generated) == len(expected) and len(expected) > 0
    log_fn(
        f"[case {case_result['case_id']}] {'PASS' if passed else 'FAIL'} "
        f"(first mismatch index={mismatch_index})"
    )
    case_result["passed"] = passed
    case_result["first_mismatch_index"] = mismatch_index
    return case_result


def _bench_stats_for_case(case_result: Dict[str, Any]) -> Dict[str, Any]:
    """TTFT/TPOT/throughput per case (see docs/INFERENCE.md for definitions)."""
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


# ---------- per-mode decode + result writing ----------

def _decode_cases(transformer, tokenizer, cases: List[Dict[str, Any]], log_fn) -> List[Dict[str, Any]]:
    """Greedy-decode each case in `cases`. Returns the per-case result dicts."""
    results: List[Dict[str, Any]] = []
    for case in cases:
        log_fn(f"[native-run] >>> running case {case['case_id']} <<<")
        results.append(_decode_one_case(transformer, tokenizer, case, log_fn))
    return results


def _run_verify_phase(
    transformer, tokenizer, cases: List[Dict[str, Any]],
    base_result: Dict[str, Any], shared: Dict[str, Any],
    config: Dict[str, str], is_root: bool, log_fn, write_filename: str,
) -> Dict[str, Any]:
    """Decode verify cases, score, write the verify JSON, return summary."""
    overall_start = time.perf_counter()
    case_results = _decode_cases(transformer, tokenizer, cases, log_fn)
    elapsed = time.perf_counter() - overall_start

    passed_count = 0
    failed_count = 0
    for cr in case_results:
        _score_verify(cr, log_fn)
        if cr["passed"]:
            passed_count += 1
        else:
            failed_count += 1

    log_fn(
        f"[native-run] verify summary: total={len(case_results)} "
        f"passed={passed_count} failed={failed_count} in {elapsed:.2f}s"
    )

    result = dict(base_result)
    result.update(shared)
    result["status"] = "verified" if failed_count == 0 else "mismatch"
    result["prompt_source"] = "reference_cases"
    result["passed_cases"] = passed_count
    result["failed_cases"] = failed_count
    result["cases"] = case_results
    result["timing_phase_seconds"] = elapsed
    out_path = _write_result(config, result, is_root, write_filename)
    log_fn(f"[native-run] result JSON: {out_path}")

    return {
        "passed": failed_count == 0,
        "passed_cases": passed_count,
        "failed_cases": failed_count,
        "out_path": out_path,
        "elapsed_seconds": elapsed,
    }


def _run_generate_phase(
    transformer, tokenizer, cases: List[Dict[str, Any]],
    base_result: Dict[str, Any], shared: Dict[str, Any],
    config: Dict[str, str], is_root: bool, log_fn, write_filename: str,
) -> Dict[str, Any]:
    overall_start = time.perf_counter()
    case_results = _decode_cases(transformer, tokenizer, cases, log_fn)
    elapsed = time.perf_counter() - overall_start
    log_fn(f"[native-run] generate summary: total={len(case_results)} cases in {elapsed:.2f}s")
    result = dict(base_result)
    result.update(shared)
    result["status"] = "generated"
    result["prompt_source"] = "synthetic_exact_prompt"
    result["cases"] = case_results
    result["timing_phase_seconds"] = elapsed
    out_path = _write_result(config, result, is_root, write_filename)
    log_fn(f"[native-run] result JSON: {out_path}")
    return {"out_path": out_path, "elapsed_seconds": elapsed}


def _run_bench_phase(
    transformer, tokenizer, cases: List[Dict[str, Any]],
    base_result: Dict[str, Any], shared: Dict[str, Any],
    config: Dict[str, str], is_root: bool, log_fn, write_filename: str,
) -> Dict[str, Any]:
    overall_start = time.perf_counter()
    case_results = _decode_cases(transformer, tokenizer, cases, log_fn)
    elapsed = time.perf_counter() - overall_start

    bench_cases = [_bench_stats_for_case(cr) for cr in case_results]
    agg = _aggregate_bench(bench_cases)
    log_fn(
        f"[native-run] bench summary: total={len(case_results)} cases in {elapsed:.2f}s; "
        f"ttft_mean={agg.get('ttft_seconds_mean')} "
        f"tpot_mean={agg.get('tpot_seconds_mean')} "
        f"tps_mean={agg.get('tokens_per_second_mean')}"
    )

    result = dict(base_result)
    result.update(shared)
    result["status"] = "benchmarked"
    result["prompt_source"] = "synthetic_exact_prompt"
    result["cases"] = case_results
    result["bench_per_case"] = bench_cases
    result["bench_aggregate"] = agg
    result["timing_phase_seconds"] = elapsed
    out_path = _write_result(config, result, is_root, write_filename)
    log_fn(f"[native-run] result JSON: {out_path}")
    return {"out_path": out_path, "elapsed_seconds": elapsed, "aggregate": agg}


# ---------- main run() ----------

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

    log(f"[native-run] resolved_config: {resolved_config_path}")
    log(f"[native-run] TAG: {config['TAG']}")
    log(f"[native-run] RUN_MODE: {run_mode}")
    log(
        f"[native-run] dist env: rank={dist_env['rank']} world={dist_env['world_size']} "
        f"local_rank={dist_env['local_rank']} "
        f"master={dist_env['master_addr']}:{dist_env['master_port']}"
    )

    needs_reference = run_mode in ("verify", "both")
    needs_synthetic = run_mode in ("generate", "bench", "both")
    lin_tokens = int(config["LIN_TOKENS"])
    lout_tokens = int(config["LOUT_TOKENS"])
    batch_size = int(config["BATCH_SIZE"])

    # Reference cases are file-driven and don't need the tokenizer.
    reference_cases: List[Dict[str, Any]] = []
    if needs_reference:
        if case_id:
            reference_cases = [_resolve_reference_case(config, reference_group, case_id)]
        else:
            reference_cases = _enumerate_reference_cases(config, reference_group)
        log(f"[native-run] reference group: {reference_group} ({len(reference_cases)} case(s))")
        for c in reference_cases:
            log(
                f"[native-run]   ref case {c['case_id']}: "
                f"lin={c['lin_tokens']} lout={c['lout_tokens']} bs={c['batch_size']}"
            )
    if needs_synthetic:
        log(
            f"[native-run] synthetic prompt plan: Lin={lin_tokens} Lout={lout_tokens} "
            f"BS={batch_size}"
        )

    # Decide the envelope max across all case sources to size the model once.
    envelope_sources: List[Dict[str, int]] = []
    if reference_cases:
        envelope_sources.append(_envelope_lin_lout_bs(reference_cases))
    if needs_synthetic:
        envelope_sources.append({
            "lin_tokens": lin_tokens,
            "lout_tokens": lout_tokens,
            "batch_size": batch_size,
        })
    if not envelope_sources:
        _fail(f"no cases planned for RUN_MODE={run_mode!r}")
    envelope = {
        "lin_tokens": max(e["lin_tokens"] for e in envelope_sources),
        "lout_tokens": max(e["lout_tokens"] for e in envelope_sources),
        "batch_size": max(e["batch_size"] for e in envelope_sources),
    }
    log(
        f"[native-run] model envelope (max across sources): "
        f"lin={envelope['lin_tokens']} lout={envelope['lout_tokens']} bs={envelope['batch_size']}"
    )

    setup_thread_env(config, log_fn=log)
    initialize_distributed_if_needed(config, dist_env, log_fn=log)

    bundle = import_deepseek_modules(config)
    model_module = bundle["model"]
    log(f"[native-run] model.__file__ = {bundle['model_file']}")

    # Cache plan must be decided BEFORE construction so we know whether to
    # construct Linear layers as BF16 (cache-read path) or FP8.
    cache_plan = resolve_cache_plan(config, dist_env, log_fn=log)
    if cache_plan["override_modelargs_dtype"]:
        log(
            f"[native-run] cache plan overrides ModelArgs.dtype -> "
            f"{cache_plan['override_modelargs_dtype']} (cache-read path)"
        )

    args, modelargs_report, native_path = build_modelargs_for_case(
        model_module, config, envelope, max_seq_len_pad=0
    )
    # Apply cache-read dtype override on top of the existing overrides.
    if cache_plan["override_modelargs_dtype"]:
        new_dtype = cache_plan["override_modelargs_dtype"]
        args.dtype = new_dtype
        args.scale_fmt = None
        modelargs_report["overridden"].append({"field": "dtype", "value": new_dtype, "source": "cache_plan"})
        modelargs_report["overridden"].append({"field": "scale_fmt", "value": None, "source": "cache_plan"})

    log(f"[native-run] ModelArgs source: native {native_path.name}")
    log(
        "[native-run]   Applied overrides: "
        + ", ".join(f"{e['field']}={e['value']}" for e in modelargs_report["overridden"])
    )
    log(f"[native-run]   Native config fields ({len(modelargs_report['native_fields'])}): "
        f"{modelargs_report['native_fields']}")
    log(f"[native-run]   Defaulted: {modelargs_report['defaulted']}")
    summary = summarize_modelargs(args)
    for k in ("n_layers", "dim", "n_heads", "n_routed_experts", "n_activated_experts",
              "max_seq_len", "max_batch_size", "dtype", "scale_fmt"):
        log(f"[native-run]   {k} = {summary[k]}")

    transformer = construct_transformer(model_module, args, log_fn=log)
    try:
        total_params = sum(p.numel() for p in transformer.parameters())
        log(f"[native-run] total parameters (numel sum): {total_params:,}")
    except Exception as exc:
        log(f"[native-run] total parameters: <unavailable: {exc!r}>")

    base_result: Dict[str, Any] = {
        "tag": config["TAG"],
        "run_mode": run_mode,
        "reference_group": reference_group if needs_reference else None,
        "case_id_filter": case_id if needs_reference else None,
        "modelargs_report": modelargs_report,
        "modelargs_summary": {k: str(v) for k, v in summary.items()},
        "cache_plan": {
            "mode": cache_plan["mode"],
            "cache_dir": str(cache_plan["cache_dir"]) if cache_plan["cache_dir"] else None,
            "cache_file": str(cache_plan["cache_file"]) if cache_plan["cache_file"] else None,
            "cache_existed_at_plan": cache_plan["cache_exists"],
            "do_read_cache": cache_plan["do_read_cache"],
            "do_write_cache": cache_plan["do_write_cache"],
        },
        "synthetic_plan": (
            {"lin_tokens": lin_tokens, "lout_tokens": lout_tokens, "batch_size": batch_size}
            if needs_synthetic else None
        ),
    }

    # --no-load-weights: no shard read, no tokenizer required for verify, but
    # generate/bench/both want the synthetic plan reported. We don't have a
    # tokenizer here, so we report planned shape instead of materialized prompts.
    if no_load_weights:
        log(
            "[native-run] --no-load-weights set; constructing once and reporting "
            "planned cases without loading weights. Exiting 0."
        )
        result = dict(base_result)
        result["status"] = "construct_only"
        result["cases"] = _planned_cases_summary(reference_cases) if needs_reference else []
        if needs_synthetic:
            result["planned_synthetic"] = {
                "lin_tokens": lin_tokens,
                "lout_tokens": lout_tokens,
                "batch_size": batch_size,
                "prompt_source": "synthetic_exact_prompt",
            }
        result["skipped"] = ["weight_load", "tokenization", "generation"]
        _write_result(config, result, is_root, RUN_MODE_RESULT_FILENAME[run_mode])
        return 0

    # ---------- weight load: FP8+dequant or BF16 cache read ----------
    cache_read_report: Optional[Dict[str, Any]] = None
    load_report: Optional[Dict[str, Any]] = None
    dequant_report: Optional[Dict[str, Any]] = None
    cache_write_report: Optional[Dict[str, Any]] = None

    if cache_plan["do_read_cache"]:
        cache_read_report = load_cached_bf16_weights(transformer, cache_plan, dist_env, log_fn=log)
    else:
        load_report = load_weights_into_transformer(transformer, config, dist_env, log_fn=log)
        dequant_report = maybe_dequantize_fp8(transformer, config, log_fn=log)
        cache_write_report = maybe_write_dequant_cache(transformer, cache_plan, config, dist_env, log_fn=log)

    if no_generate:
        log(
            f"[native-run] --no-generate set; weights loaded. "
            f"Reporting planned case(s). Exiting 0."
        )
        result = dict(base_result)
        result["status"] = "weights_loaded_no_generation"
        result["load_report"] = load_report
        result["dequant_report"] = dequant_report
        result["cache_read_report"] = cache_read_report
        result["cache_write_report"] = cache_write_report
        result["cases"] = _planned_cases_summary(reference_cases) if needs_reference else []
        if needs_synthetic:
            result["planned_synthetic"] = {
                "lin_tokens": lin_tokens,
                "lout_tokens": lout_tokens,
                "batch_size": batch_size,
                "prompt_source": "synthetic_exact_prompt",
            }
        _write_result(config, result, is_root, RUN_MODE_RESULT_FILENAME[run_mode])
        return 0

    # ---------- tokenizer + synthetic case materialization ----------
    tokenizer, tokenizer_dir = load_tokenizer(config, log_fn=log)

    synthetic_cases: List[Dict[str, Any]] = []
    if needs_synthetic:
        log(
            f"[native-run] building synthetic exact-token prompt(s): "
            f"Lin={lin_tokens} Lout={lout_tokens} BS={batch_size}"
        )
        synthetic_cases = build_synthetic_cases(
            tokenizer,
            lin_tokens=lin_tokens,
            lout_tokens=lout_tokens,
            batch_size=batch_size,
            tag_or_label=config["TAG"],
        )
        for sc in synthetic_cases:
            actual = encode_len(tokenizer, sc["prompt_text"])
            log(
                f"[native-run]   synth {sc['case_id']}: "
                f"lin_target={lin_tokens} lin_actual={actual} lout={sc['lout_tokens']} "
                f"seed={sc['source']['seed_text']!r}"
            )

    shared: Dict[str, Any] = {
        "load_report": load_report,
        "dequant_report": dequant_report,
        "cache_read_report": cache_read_report,
        "cache_write_report": cache_write_report,
        "tokenizer_dir": str(tokenizer_dir),
    }

    rc = 0

    if run_mode == "verify":
        base_result["total_cases"] = len(reference_cases)
        v = _run_verify_phase(
            transformer, tokenizer, reference_cases, base_result, shared,
            config, is_root, log, RUN_MODE_RESULT_FILENAME["verify"],
        )
        rc = 0 if v["passed"] else 1

    elif run_mode == "generate":
        base_result["total_cases"] = len(synthetic_cases)
        _run_generate_phase(
            transformer, tokenizer, synthetic_cases, base_result, shared,
            config, is_root, log, RUN_MODE_RESULT_FILENAME["generate"],
        )

    elif run_mode == "bench":
        base_result["total_cases"] = len(synthetic_cases)
        _run_bench_phase(
            transformer, tokenizer, synthetic_cases, base_result, shared,
            config, is_root, log, RUN_MODE_RESULT_FILENAME["bench"],
        )

    elif run_mode == "both":
        # Phase 1: verify on reference cases.
        verify_base = dict(base_result)
        verify_base["total_cases"] = len(reference_cases)
        verify_summary = _run_verify_phase(
            transformer, tokenizer, reference_cases, verify_base, shared,
            config, is_root, log, RUN_MODE_RESULT_FILENAME["verify"],
        )
        both_summary = dict(base_result)
        both_summary["verify_prompt_source"] = "reference_cases"
        both_summary["bench_prompt_source"] = "synthetic_exact_prompt"
        both_summary["verify_passed"] = verify_summary["passed"]
        both_summary["verify_path"] = str(verify_summary["out_path"])
        both_summary["verify_passed_cases"] = verify_summary["passed_cases"]
        both_summary["verify_failed_cases"] = verify_summary["failed_cases"]

        if not verify_summary["passed"]:
            both_summary["status"] = "verify_failed"
            both_summary["bench_path"] = None
            _write_result(config, both_summary, is_root, RUN_MODE_RESULT_FILENAME["both"])
            rc = 1
        else:
            # Phase 2: bench on synthetic prompts.
            bench_base = dict(base_result)
            bench_base["total_cases"] = len(synthetic_cases)
            bench_summary = _run_bench_phase(
                transformer, tokenizer, synthetic_cases, bench_base, shared,
                config, is_root, log, RUN_MODE_RESULT_FILENAME["bench"],
            )
            both_summary["status"] = "both_passed"
            both_summary["bench_path"] = str(bench_summary["out_path"])
            both_summary["bench_aggregate"] = bench_summary["aggregate"]
            both_summary["bench_elapsed_seconds"] = bench_summary["elapsed_seconds"]
            _write_result(config, both_summary, is_root, RUN_MODE_RESULT_FILENAME["both"])
            rc = 0
    else:
        _fail(f"unreachable: RUN_MODE={run_mode!r}")

    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass

    return rc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Native CPU TP-aware runner.")
    p.add_argument("--resolved-config", required=True)
    p.add_argument("--reference-group", required=True)
    p.add_argument(
        "--case-id",
        default=None,
        help="Optional. Only used by RUN_MODE=verify and =both. If omitted, every *.json in the reference group is run.",
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
