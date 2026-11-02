#!/usr/bin/env python3
"""Native CPU sharding-aware runner. Dispatches on the resolved-config RUN_MODE.

Supported SHARDING_MODE values (enforced by parse_config.sh):
  tp2       — tensor parallel. dist init BEFORE construction so world_size is
              baked into ColumnParallel/RowParallel layers; each rank loads
              model{rank}-mp2.safetensors (or the BF16 equivalent).
  dp2_epon  — data parallel with expert parallel. Model is REPLICATED, so dist
              init runs AFTER construction (layers must be built full size) and
              src/overrides/ep_moe.py replaces MoE.forward per instance.

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
import os
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
    initialize_distributed_before_construction,
    initialize_distributed_after_construction,
    assert_distributed_ready,
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


# Set once from config["PREFILL_CHUNK"] before any decode. Module-level rather
# than threaded through _decode_case_group / _decode_case_group_sharded and
# their callers, which would be a wider change for no extra clarity.
_PREFILL_CHUNK = 0


def _decode_case_group(transformer, tokenizer, cases: List[Dict[str, Any]], log_fn) -> List[Dict[str, Any]]:
    """Tokenize + greedy decode `cases` TOGETHER as one batch of B=len(cases).

    One forward per step covers every row, which is the point: it is what makes
    a batch-scaling curve meaningful and what exposes MoE experts to several
    sequences at once. Decoding the cases in a loop instead would measure B
    independent runs.

    All rows must share a prompt length and a `lout_tokens`; both prompt sources
    are exact-length by construction, so this holds, and greedy_decode asserts
    the first. Timing is per batch, so each row's result carries the SAME
    prefill/decode seconds -- divide by B for a per-sequence figure.
    """
    prompt_tokens = [encode_prompt(tokenizer, c["prompt_text"]) for c in cases]
    B = len(cases)
    for c, toks in zip(cases, prompt_tokens):
        text = c["prompt_text"]
        log_fn(f"[case {c['case_id']}] prompt: {text[:120]!r}{'...' if len(text) > 120 else ''}")
        log_fn(f"[case {c['case_id']}] prompt token count: {len(toks)}")

    louts = {int(c["lout_tokens"]) for c in cases}
    if len(louts) != 1:
        raise ValueError(f"batched cases must share lout_tokens; got {sorted(louts)}")
    lout = louts.pop()

    t0 = time.perf_counter()
    # Zero the kernel's stage accumulators so the breakdown below covers only
    # this group's prefill+decode. This is the gather-vs-rest split for TPOT.
    try:
        import flash_attn_cpu as _fa
        _fa.reset_timings()
    except Exception:
        _fa = None
    try:
        import kernel as _kn
        _kn.reset_index_timer()
    except Exception:
        _kn = None

    generated_rows, prefill_s, decode_s = greedy_decode(
        transformer, prompt_tokens, lout, log_fn=log_fn,
        prefill_chunk=_PREFILL_CHUNK,
    )

    if _fa is not None:
        _t = _fa.get_timings()
        if _t.get("calls", 0):
            # lout counts the token produced by PREFILL, so decode steps are
            # lout-1 (confirmed: 427 kernel calls / 61 layers = 7 at lout=8).
            steps = max(1, lout - 1)
            log_fn("[kernel] decode stage breakdown, ms per decode step "
                   f"(B={B}, {int(_t['calls'])} kernel calls total):")
            for _k in ("gather", "kpack", "qkt", "softmax", "vpack", "amxv",
                       "barrier", "reduce"):
                log_fn(f"[kernel]   {_k:9s} {_t.get(_k, 0.0) / steps * 1e3:9.2f}")
            if _kn is not None:
                _it = _kn.get_index_timer()
                if _it["calls"]:
                    # NOT per-step: this accumulator is reset before
                    # greedy_decode, so it spans PREFILL + decode, and the
                    # prefill call dominates it. Call count gives it away --
                    # attention logs 61*steps calls (decode only) while the
                    # indexer logs 61*(steps+1), the extra one per layer being
                    # the prefill chunk at M=Lin,N=Lin against M=1,N=ctx for
                    # each decode step. At Lin=31744 that one call is ~99.8%
                    # of the total, so dividing by steps and calling it
                    # "per decode step" overstates decode indexer by ~500x.
                    log_fn(f"[kernel]   {'INDEXER':9s} {_it['seconds'] * 1e3:9.2f} ms TOTAL"
                           f" over prefill+decode ({int(_it['calls'])} fp8_index calls)")
            log_fn(f"[kernel]   {'TOTAL':9s} "
                   f"{sum(_t.get(k, 0.0) for k in ('gather','kpack','qkt','softmax','vpack','amxv','barrier','reduce')) / steps * 1e3:9.2f}")
    elapsed = time.perf_counter() - t0

    results: List[Dict[str, Any]] = []
    for idx, (case, toks, generated) in enumerate(zip(cases, prompt_tokens, generated_rows)):
        raw_expected = case.get("expected_output_token_ids") or []
        expected = list(raw_expected) if raw_expected else []
        log_fn(f"[case {case['case_id']}] generated tokens ({len(generated)}): {generated}")
        decoded = try_decode(tokenizer, generated)
        if decoded is not None:
            log_fn(f"[case {case['case_id']}] decoded text: {decoded!r}")
        results.append({
            "case_id": case["case_id"],
            "batch_index": idx,
            "batch_size": B,
            "prompt_source": case.get("source", {}),
            "prompt_text": case["prompt_text"],
            "prompt_tokens": toks,
            "prompt_token_count": len(toks),
            "expected_tokens": expected,
            "reference_expected_tokens_available": bool(expected),
            "generated_tokens": generated,
            "decoded_text": decoded,
            "timing": {
                # batch-level: identical across rows, by construction
                "prefill_seconds": prefill_s,
                "decode_seconds_total": decode_s,
                "total_seconds": elapsed,
                "batch_size": B,
                "per_sequence_total_seconds": elapsed / B,
            },
        })
    return results


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

def _decode_cases(transformer, tokenizer, cases: List[Dict[str, Any]], log_fn,
                  config: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """Decode `cases`, batching together those that share a prompt length.

    Synthetic/corpus cases for one run are all LIN_TOKENS long, so they form a
    single batch of B=BATCH_SIZE. Reference cases may differ in length, so they
    are grouped and each group is run as its own batch -- that keeps verify
    working unchanged while letting bench/generate batch properly.
    """
    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for case in cases:
        key = (len(encode_prompt(tokenizer, case["prompt_text"])), int(case["lout_tokens"]))
        groups.setdefault(key, []).append(case)

    sharded = (config or {}).get("SHARDING_MODE", "").lower() == "dp2_true"
    results: List[Dict[str, Any]] = []
    for (plen, lout), group in groups.items():
        ids = ", ".join(c["case_id"] for c in group)
        log_fn(
            f"[native-run] >>> running batch of {len(group)} "
            f"(Lin={plen} Lout={lout}): {ids} <<<"
        )
        if sharded and len(group) > 1:
            results.extend(_decode_case_group_sharded(transformer, tokenizer, group, log_fn))
        else:
            results.extend(_decode_case_group(transformer, tokenizer, group, log_fn))
    return results


def _decode_case_group_sharded(transformer, tokenizer, cases, log_fn):
    """dp2_true: rank r decodes cases[r::W], then token ids are gathered.

    Each rank attends only over its own sequences and holds only their KV --
    that is the whole point of the mode. The MoE's all_gather reconstitutes the
    full batch inside every layer, so routing is unchanged; only attention and
    the KV buffers are sharded.

    Every rank must make the SAME NUMBER of forward calls with the SAME shapes,
    or the collectives inside MoE deadlock. Guaranteed here because the parser
    requires B % DP_SIZE == 0, so every rank gets an equal row count, and lout
    is uniform across the group.

    Results are gathered at the end rather than written per rank: rank 0 writes
    the result JSON but only generated its own slice. The payload is tiny
    (B x lout ints), so a single all_gather of token ids is simpler and cheaper
    than merging per-rank files.
    """
    import torch
    import torch.distributed as dist

    world = dist.get_world_size()
    rank = dist.get_rank()
    local = cases[rank::world]
    log_fn(
        f"[true-dp] rank {rank}/{world} decoding {len(local)} of {len(cases)} "
        f"sequences: {', '.join(c['case_id'] for c in local)}"
    )

    local_results = _decode_case_group(transformer, tokenizer, local, log_fn)

    # all_gather the generated ids; fixed shape [n_local, lout] on every rank.
    lout = int(cases[0]["lout_tokens"])
    mine = torch.tensor([r["generated_tokens"] for r in local_results], dtype=torch.long)
    bufs = [torch.empty_like(mine) for _ in range(world)]
    dist.all_gather(bufs, mine.contiguous())

    # rank r held cases[r::world], so local index i is global index r + world*i
    merged: List[Dict[str, Any]] = [None] * len(cases)  # type: ignore[list-item]
    for r in range(world):
        for i in range(bufs[r].shape[0]):
            g = r + world * i
            if r == rank:
                res = dict(local_results[i])
            else:
                case = cases[g]
                toks = encode_prompt(tokenizer, case["prompt_text"])
                res = {
                    "case_id": case["case_id"],
                    "prompt_source": case.get("source", {}),
                    "prompt_text": case["prompt_text"],
                    "prompt_tokens": toks,
                    "prompt_token_count": len(toks),
                    "expected_tokens": list(case.get("expected_output_token_ids") or []),
                    "reference_expected_tokens_available": bool(case.get("expected_output_token_ids")),
                    "timing": dict(local_results[0]["timing"]),
                    "decoded_by_rank": r,
                }
            res["generated_tokens"] = bufs[r][i].tolist()
            res["decoded_text"] = try_decode(tokenizer, res["generated_tokens"])
            res["batch_index"] = g
            res["batch_size"] = len(cases)
            res["attention_mode"] = "sequence_sharded"
            merged[g] = res
    log_fn(f"[true-dp] gathered {len(cases)} sequences' tokens across {world} ranks")
    return merged


def _run_verify_phase(
    transformer, tokenizer, cases: List[Dict[str, Any]],
    base_result: Dict[str, Any], shared: Dict[str, Any],
    config: Dict[str, str], is_root: bool, log_fn, write_filename: str,
) -> Dict[str, Any]:
    """Decode verify cases, score, write the verify JSON, return summary."""
    overall_start = time.perf_counter()
    case_results = _decode_cases(transformer, tokenizer, cases, log_fn, config=config)
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
    case_results = _decode_cases(transformer, tokenizer, cases, log_fn, config=config)
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
    case_results = _decode_cases(transformer, tokenizer, cases, log_fn, config=config)
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
    initialize_distributed_before_construction(config, dist_env, log_fn=log)

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

    # DP modes build every parallel layer at FULL size (world_size==1 during
    # construction), so the process group comes up only now. TP modes already
    # initialized before construction; both paths are then asserted ready so a
    # missed hook fails loudly instead of running silently as a single rank.
    initialize_distributed_after_construction(config, dist_env, log_fn=log)
    assert_distributed_ready(dist_env, log_fn=log)
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
                "prompt_source": (config.get("PROMPT_SOURCE") or "synthetic"),
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
        cache_read_report = load_cached_bf16_weights(transformer, cache_plan, dist_env, config, log_fn=log)
    else:
        load_report = load_weights_into_transformer(transformer, config, dist_env, log_fn=log)
        dequant_report = maybe_dequantize_fp8(transformer, config, log_fn=log)
        cache_write_report = maybe_write_dequant_cache(transformer, cache_plan, config, dist_env, log_fn=log)

    indexer_bcast_report: Optional[Dict[str, Any]] = None
    if (config.get("SKIP_INDEXER_BROADCAST") or "1").strip() == "1":
        from indexer_broadcast import install_indexer_broadcast_skip  # src/overrides

        indexer_bcast_report = install_indexer_broadcast_skip(model_module, log_fn=log)
        base_result["indexer_bcast_report"] = indexer_bcast_report

    global _PREFILL_CHUNK
    # Chunked prefill must be installed BEFORE any forward: it rebinds
    # MLA.forward onto the absorbed path so prefill reads the full kv cache.
    # generation.py's chunking is unsafe without it.
    # ORDER MATTERS: flash_attn_cpu sets chunked_prefill.ATTN_IMPL, which
    # install_chunked_prefill reads when it binds MLA.forward. Installed the
    # other way round, the kernel is loaded but never used.
    if (config.get("USE_BF16_ALLREDUCE") or "").strip() == "1":
        from bf16_allreduce import install_bf16_allreduce  # src/overrides

        base_result["bf16_allreduce_report"] = install_bf16_allreduce(
            model_module, log_fn=log)

    # kernel.py (the override model.py imports) reads USE_AMX_INDEXER at call
    # time, so it must be in the env regardless of the attention flag.
    if (config.get("USE_AMX_INDEXER") or "").strip():
        os.environ["USE_AMX_INDEXER"] = config["USE_AMX_INDEXER"].strip()

    # moe_batched.py reads this at IMPORT time, so it must be in the env before
    # install_batched_moe runs below.
    if (config.get("MOE_BARRIER_PROBE") or "").strip():
        os.environ["MOE_BARRIER_PROBE"] = config["MOE_BARRIER_PROBE"].strip()

    use_amx_attention = (config.get("USE_AMX_ATTENTION") or "").strip() == "1"
    if use_amx_attention:
        # Kernel tunables come from the config but are read by flash_attn_cpu at
        # import time, so they must be in os.environ BEFORE that import.
        for _k in ("FLASHMLA_DECODE", "FLASHMLA_THREADS", "FLASHMLA_BLOCK_SIZE",
                   "FLASHMLA_MOE_MIN_TOKENS", "USE_AMX_INDEXER"):
            _v = (config.get(_k) or "").strip()
            if _v:
                os.environ[_k] = _v
        from flash_attn_cpu import install_flash_attention  # src/overrides

        base_result["flash_attn_report"] = install_flash_attention(
            so_path=(config.get("AMX_ATTENTION_SO") or "").strip() or None,
            log_fn=log)

    # Install whenever EITHER chunking or the AMX kernels are wanted. The two
    # are independent: chunking is an INDEXER workaround (its index_score is
    # [bsz, seqlen, end_pos] -- 57.6 GB single-shot at Lin=120k, and chunking
    # also halves its work, Lin^2 -> Lin^2/2), while the kernels are flash-style
    # and need no chunking of their own.
    if (int(config.get("PREFILL_CHUNK") or 0) > 0
            or (config.get("USE_AMX_ATTENTION") or "").strip() == "1"):
        from chunked_prefill import install_chunked_prefill  # src/overrides

        _PREFILL_CHUNK = int(config["PREFILL_CHUNK"])

        base_result["chunked_prefill_report"] = install_chunked_prefill(
            transformer, model_module, log_fn=log)

    # AMX prefill attention: sets mla_absorbed.PREFILL_IMPL, and raises if the
    # .so is missing rather than silently measuring the torch core.
    use_amx_prefill_attention = (config.get("USE_AMX_PREFILL_ATTENTION") or "0").strip() == "1"
    if use_amx_prefill_attention:
        from amx_prefill_attention import install_amx_prefill_attention  # src/overrides

        base_result["amx_prefill_attention_report"] = install_amx_prefill_attention(
            so_path=(config.get("AMX_PREFILL_ATTENTION_SO") or "").strip() or None, log_fn=log)

    # AMX prefill indexer scores: sets kernel.INDEX_IMPL (fp8_index, M > 1 only).
    # Independent of the absorbed install: both prefill paths call fp8_index.
    if (config.get("USE_AMX_PREFILL_INDEXER") or "0").strip() == "1":
        from amx_prefill_indexer import install_amx_prefill_indexer  # src/overrides

        base_result["amx_prefill_indexer_report"] = install_amx_prefill_indexer(
            so_path=(config.get("AMX_PREFILL_INDEXER_SO") or "").strip() or None, log_fn=log)

    # Chunked prefill must be installed BEFORE any forward: it rebinds
    # MLA.forward onto the absorbed path so prefill reads the full kv cache.
    # generation.py's chunking is unsafe without it. The AMX kernel runs inside
    # that same forward, so it needs the install even without chunking.
    # KV_CACHE_DTYPE=bf16 also needs both overrides: upstream quantises both
    # caches inside MLA.forward and Indexer.forward, with no switch.
    #
    # On top of chunked_prefill when USE_AMX_ATTENTION=1: mla_absorbed then takes
    # every prefill forward and hands fp8 decode to the forward it replaced, so
    # decode keeps the flash_attn_cpu kernels. Configs that only set
    # USE_AMX_ATTENTION + PREFILL_CHUNK keep chunked_prefill for prefill too.
    kv_cache_dtype = config["KV_CACHE_DTYPE"]
    if (use_amx_prefill_attention or kv_cache_dtype == "bf16"
            or (int(config.get("PREFILL_CHUNK") or 0) > 0 and not use_amx_attention)):
        from mla_absorbed import install_mla_absorbed  # src/overrides
        from indexer_forward import install_indexer_forward  # src/overrides

        _PREFILL_CHUNK = int(config["PREFILL_CHUNK"])
        base_result["mla_absorbed_report"] = install_mla_absorbed(
            transformer, model_module, log_fn=log, kv_cache_dtype=kv_cache_dtype)
        base_result["indexer_forward_report"] = install_indexer_forward(
            transformer, model_module, kv_cache_dtype=kv_cache_dtype, log_fn=log)

    # Optional: capture real Indexer top-k from the prefill pass. Needed to pick
    # the sparse-prefill kernel structure (see src/overrides/indexer_dump.py).
    # Only meaningful at LIN_TOKENS > 2048, where topk actually selects a subset.
    if (config.get("INDEXER_DUMP_DIR") or "").strip():
        from indexer_dump import install_indexer_dump  # src/overrides

        _layers = [int(x) for x in (config.get("INDEXER_DUMP_LAYERS") or "0").split(",") if x.strip()]
        base_result["indexer_dump_report"] = install_indexer_dump(
            transformer, config["INDEXER_DUMP_DIR"].strip(),
            layers=_layers, rank=int(dist_env["rank"]), log_fn=log,
        )

    ep_moe_report: Optional[Dict[str, Any]] = None
    sharding_mode = config.get("SHARDING_MODE", "").lower()
    if sharding_mode in ("dp2_epon", "dp2_true"):
        # Both modes prune non-local experts identically; they differ only in
        # which MoE forward gets bound. install_true_dp_moe wraps install_ep_moe
        # and then decides from batch_size: B=1 keeps the replicated forward
        # (and logs that the run is NOT measuring true DP), B>1 binds the
        # all_gather/reduce_scatter one.
        #
        # This decision MUST agree with _decode_case_group_sharded, which shards
        # the batch on the same `len(group) > 1` condition. If one shards and the
        # other does not, the collectives inside MoE mismatch and the ranks
        # deadlock. Both key off batch_size; they live in different files.
        if sharding_mode == "dp2_true":
            from ep_moe import install_true_dp_moe  # src/overrides

            ep_moe_report = install_true_dp_moe(
                transformer, model_module, dist_env,
                n_routed_experts=int(args.n_routed_experts),
                ep_size=int(config.get("EP_SIZE") or 0),
                batch_size=int(config.get("BATCH_SIZE") or 1), log_fn=log,
            )
        else:
            from ep_moe import install_ep_moe  # src/overrides

            ep_moe_report = install_ep_moe(
                transformer, model_module, dist_env,
                n_routed_experts=int(args.n_routed_experts),
                ep_size=int(config.get("EP_SIZE") or 0), log_fn=log,
            )
        base_result["ep_moe_report"] = ep_moe_report

    # AFTER any EP/DP install (and outside it, since tp2 does no EP install and
    # would otherwise never get here). Off by default; opt-in per run.
    if (config.get("USE_BATCHED_MOE") or "").strip() == "1":
        from moe_batched import install_batched_moe  # src/overrides

        base_result["moe_batched_report"] = install_batched_moe(
            transformer, model_module=model_module, log_fn=log)

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
            config=config,
        )
        for sc in synthetic_cases:
            actual = encode_len(tokenizer, sc["prompt_text"])
            src = sc["source"]
            # Provenance differs by PROMPT_SOURCE: the synthetic path carries
            # seed_text, the corpus path carries task/record_index. Never index
            # either key directly -- that is what broke job 27841357.
            if "seed_text" in src:
                origin = f"seed={src['seed_text']!r}"
            else:
                origin = (
                    f"corpus={src.get('source', '?')}:{src.get('task', '?')}"
                    f"[{src.get('record_index', '?')}] seed={src.get('seed', '?')}"
                )
            log(
                f"[native-run]   synth {sc['case_id']}: "
                f"lin_target={lin_tokens} lin_actual={actual} lout={sc['lout_tokens']} "
                f"{origin}"
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
        # BS_SWEEP: measure several batch sizes in ONE process. The weight load
        # is ~15 min, so a 9-point sweep run as 9 jobs spends 2.2 h just
        # loading. The KV cache is allocated once for BATCH_SIZE (set to the
        # max of the sweep) and a smaller batch simply uses the first rows;
        # cases already overwrite the cache from position 0, which is why
        # multi-case groups work today.
        #
        # dp2_true: install_true_dp_moe bound its forward from BATCH_SIZE (the
        # max, so >1 -> the sharded forward) while _decode_case_group_sharded
        # shards on len(group)>1. Sweeping BS=1 under dp2_true would make those
        # two disagree and DEADLOCK the ranks, so BS=1 is skipped there.
        _sweep = [int(x) for x in (config.get("BS_SWEEP") or "").replace(",", " ").split()]
        # Under SYNTH_CTX the tiled path consumes only the first SYNTH_PREFILL
        # tokens of each prompt, so building prompts at full LIN_TOKENS is pure
        # waste -- at Lin=120k over a 5-point sweep that is 31 prompts of
        # 120k tokens built to use 4k of each. LIN_TOKENS still sets
        # max_seq_len (which must cover SYNTH_CTX + LOUT_TOKENS); only the
        # prompt LENGTH drops.
        _mkln = int(config.get("SYNTH_PREFILL") or 4096) if int(config.get("SYNTH_CTX") or 0) else lin_tokens
        for _bs in (_sweep or [batch_size]):
            if sharding_mode == "dp2_true" and _bs == 1:
                log("[bs-sweep] BS=1 skipped under dp2_true (would deadlock vs the sharded MoE forward)")
                continue
            if _sweep:
                log(f"[bs-sweep] ===== BATCH_SIZE={_bs} (Lin={lin_tokens} Lout={lout_tokens}) =====")
                synthetic_cases = build_synthetic_cases(
                    tokenizer, lin_tokens=_mkln, lout_tokens=lout_tokens,
                    batch_size=_bs, tag_or_label=config["TAG"], config=config)
            base_result["total_cases"] = len(synthetic_cases)
            base_result["swept_batch_size"] = _bs
            _ctx = int(config.get("SYNTH_CTX") or 0)
            if _ctx:
                # Long-context TPOT without a long prefill. See ctx_tile.py:
                # a real SYNTH_PREFILL-token prefill is tiled out to _ctx so the
                # decode sees a full-size cache. Measures TPOT only -- the
                # generated tokens are meaningless.
                import torch
                from ctx_tile import tile_to_context, timed_decode  # src/overrides

                _pf = int(config.get("SYNTH_PREFILL") or 4096)
                # Cases carry prompt_text, NOT token ids -- encode here, the
                # same way _decode_case_group does.
                _rows = [encode_prompt(tokenizer, c["prompt_text"])[:_pf]
                         for c in synthetic_cases]
                _lens = {len(r) for r in _rows}
                if _lens != {_pf}:
                    raise ValueError(f"prefill rows must all be {_pf} tokens; got {sorted(_lens)}")
                _tok = torch.tensor(_rows, dtype=torch.long)
                # dp2_true: each rank prefills, tiles and decodes only its own
                # rows -- the same cases[r::W] split as _decode_case_group_sharded.
                # Without this the full batch went through every rank (27933614).
                _local_bs = _bs
                if sharding_mode == "dp2_true":
                    import torch.distributed as dist
                    _w, _rk = dist.get_world_size(), dist.get_rank()
                    _tok = _tok[_rk::_w].contiguous()
                    _local_bs = _tok.shape[0]
                    log(f"[true-dp] tiled path: rank {_rk}/{_w} holds {_local_bs} of {_bs} sequences")
                _t0 = time.perf_counter()
                _chunk = int(config.get("PREFILL_CHUNK") or 0) or _pf
                with torch.no_grad():
                    for _p0 in range(0, _pf, _chunk):
                        transformer.forward(_tok[:, _p0:_p0 + _chunk], _p0)
                _pfs = time.perf_counter() - _t0
                log(f"[bs-sweep] bs={_bs} real prefill {_pf} tok in {_pfs:.1f}s")
                tile_to_context(transformer, _local_bs, _pf, _ctx, log_fn=log)
                _ms = timed_decode(transformer, _tok, _ctx, lout_tokens, log_fn=log)
                _warm = _ms[1:] or _ms
                _tpot = sum(_warm) / len(_warm) / 1000.0
                log(f"[bs-sweep] RESULT bs={_bs} ctx={_ctx} tpot_s={_tpot:.4f} "
                    f"tok_s={_bs / _tpot:.3f} steps={len(_ms)} "
                    f"min_ms={min(_warm):.0f} max_ms={max(_warm):.0f}")
                _out = dict(base_result)
                _out.update({"status": "tpot_tiled_context", "synth_ctx": _ctx,
                             "synth_prefill": _pf, "prefill_seconds": _pfs,
                             "step_ms": _ms,
                             "bench_aggregate": {"tpot_seconds_mean": _tpot,
                                                 "tokens_per_second_mean": _bs / _tpot,
                                                 "ttft_seconds_mean": None}})
                _write_result(config, _out, is_root, f"bench_bs{_bs}.json")
                continue
            try:
                _r = _run_bench_phase(
                    transformer, tokenizer, synthetic_cases, base_result, shared,
                    config, is_root, log,
                    f"bench_bs{_bs}.json" if _sweep else RUN_MODE_RESULT_FILENAME["bench"],
                )
                _a = _r["aggregate"]
                log(f"[bs-sweep] RESULT bs={_bs} tpot_s={_a.get('tpot_seconds_mean')} "
                    f"ttft_s={_a.get('ttft_seconds_mean')} tok_s={_a.get('tokens_per_second_mean')}")
            except Exception as exc:
                # OOM at the top of the sweep is the ANSWER (max batch size),
                # not a failure -- record it and stop climbing.
                log(f"[bs-sweep] RESULT bs={_bs} FAILED: {type(exc).__name__}: {exc}")
                raise

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
