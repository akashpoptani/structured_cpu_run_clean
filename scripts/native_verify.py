#!/usr/bin/env python3
"""Native CPU verification: TP-aware distributed init -> Transformer construction ->
weight load -> tokenize prompt -> greedy decode -> compare to reference tokens.

Designed for the first real TP2 verification run. Driven by a resolved env
file (output of scripts/parse_config.sh --format env) and a selected
reference case under verification/references/<group>/<case>.json.

Safety flags:
  --no-load-weights : construct only; skip weight load and generation. Safe
                      to run on a login node (only reserves virtual memory).
  --no-generate     : load weights but skip the decode loop. Useful for a
                      pure weight-loading smoke under Slurm.

Without those flags the full pipeline runs; that should be launched inside
a Slurm allocation via scripts/run_native_distributed.sh.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

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
    "GPU_REFERENCE_PATH",
    "OUTPUT_ROOT",
)


def _fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def _resolve_reference_case(config: Dict[str, str], group: str, case_id: str) -> Dict[str, Any]:
    clean_root = Path(config["CLEAN_ROOT"]).resolve()
    reference_root = resolve_path(clean_root, config["GPU_REFERENCE_PATH"])
    case_path = reference_root / group / f"{case_id}.json"
    if not case_path.is_file():
        _fail(f"reference case not found: {case_path}")
    return load_case(case_path)


def _make_logger(is_root: bool):
    if is_root:
        def log(*args, **kwargs):
            print(*args, **kwargs, flush=True)
        return log
    def silent(*_args, **_kwargs):
        return None
    return silent


def _write_result(
    config: Dict[str, str], result: Dict[str, Any], is_root: bool
) -> Path:
    clean_root = Path(config["CLEAN_ROOT"]).resolve()
    output_root = resolve_path(clean_root, config["OUTPUT_ROOT"])
    out_path = output_root / "results" / config["TAG"] / "native_verify_results.json"
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


def run(
    resolved_config_path: Path,
    reference_group: str,
    case_id: str,
    no_load_weights: bool,
    no_generate: bool,
) -> int:
    config = parse_resolved_env(resolved_config_path)
    require_config_keys(config, REQUIRED_KEYS)

    dist_env = detect_distributed_env()
    is_root = dist_env["is_root"]
    log = _make_logger(is_root)

    log(f"[native-verify] resolved_config: {resolved_config_path}")
    log(f"[native-verify] TAG: {config['TAG']}")
    log(
        f"[native-verify] dist env: rank={dist_env['rank']} world={dist_env['world_size']} "
        f"local_rank={dist_env['local_rank']} "
        f"master={dist_env['master_addr']}:{dist_env['master_port']}"
    )

    case = _resolve_reference_case(config, reference_group, case_id)
    log(
        f"[native-verify] reference case: group={reference_group} case_id={case_id} "
        f"lin={case['lin_tokens']} lout={case['lout_tokens']} bs={case['batch_size']}"
    )

    setup_thread_env(config, log_fn=log)
    initialize_distributed_if_needed(config, dist_env, log_fn=log)

    bundle = import_deepseek_modules(config)
    model_module = bundle["model"]
    log(f"[native-verify] model.__file__ = {bundle['model_file']}")

    args, modelargs_report, native_path = build_modelargs_for_case(
        model_module, config, case, max_seq_len_pad=0
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

    if no_load_weights:
        log("[native-verify] --no-load-weights set; skipping weight load and generation. Exiting 0.")
        result = {
            "tag": config["TAG"],
            "reference_group": reference_group,
            "case_id": case_id,
            "status": "construct_only",
            "modelargs_report": modelargs_report,
            "modelargs_summary": {k: str(v) for k, v in summary.items()},
            "skipped": ["weight_load", "tokenization", "generation"],
        }
        _write_result(config, result, is_root)
        return 0

    load_report = load_weights_into_transformer(transformer, config, dist_env, log_fn=log)
    dequant_report = maybe_dequantize_fp8(transformer, config, log_fn=log)

    if no_generate:
        log("[native-verify] --no-generate set; skipping decode. Exiting 0 after weight load.")
        result = {
            "tag": config["TAG"],
            "reference_group": reference_group,
            "case_id": case_id,
            "status": "weights_loaded_no_generation",
            "modelargs_report": modelargs_report,
            "load_report": load_report,
            "dequant_report": dequant_report,
        }
        _write_result(config, result, is_root)
        return 0

    tokenizer, tokenizer_dir = load_tokenizer(config, log_fn=log)
    prompt_text = case["prompt_text"]
    prompt_tokens = encode_prompt(tokenizer, prompt_text)
    log(f"[native-verify] prompt: {prompt_text!r}")
    log(f"[native-verify] prompt tokens ({len(prompt_tokens)}): {prompt_tokens}")

    expected = list(case["expected_output_token_ids"])
    lout = int(case["lout_tokens"])

    t0 = time.perf_counter()
    generated, prefill_s, decode_s = greedy_decode(transformer, prompt_tokens, lout, log_fn=log)
    elapsed = time.perf_counter() - t0

    log(f"[native-verify] generated tokens ({len(generated)}): {generated}")
    log(f"[native-verify] expected tokens  ({len(expected)}): {expected}")

    decoded = try_decode(tokenizer, generated)
    if decoded is not None:
        log(f"[native-verify] decoded generated text: {decoded!r}")

    mismatch_index = _first_mismatch(expected, generated)
    passed = mismatch_index == -1 and len(generated) == len(expected)
    log(
        f"[native-verify] {'PASS' if passed else 'FAIL'} "
        f"(first mismatch index={mismatch_index})"
    )

    result = {
        "tag": config["TAG"],
        "reference_group": reference_group,
        "case_id": case_id,
        "status": "verified" if passed else "mismatch",
        "passed": passed,
        "first_mismatch_index": mismatch_index,
        "prompt_text": prompt_text,
        "prompt_tokens": prompt_tokens,
        "prompt_token_count": len(prompt_tokens),
        "expected_tokens": expected,
        "generated_tokens": generated,
        "decoded_text": decoded,
        "modelargs_report": modelargs_report,
        "load_report": load_report,
        "dequant_report": dequant_report,
        "tokenizer_dir": str(tokenizer_dir),
        "timing": {
            "prefill_seconds": prefill_s,
            "decode_seconds_total": decode_s,
            "total_seconds": elapsed,
        },
    }
    out_path = _write_result(config, result, is_root)
    log(f"[native-verify] result JSON: {out_path}")

    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass

    return 0 if passed else 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Native CPU TP-aware verification.")
    p.add_argument("--resolved-config", required=True)
    p.add_argument("--reference-group", required=True)
    p.add_argument("--case-id", required=True)
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
