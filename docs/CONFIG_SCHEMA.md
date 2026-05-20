# Config Schema

Purpose: Define the future clean configuration format.

This document describes the intended configuration model and the first local-only parser.

## 1. Config Model

The clean lane uses a flat env-file design.

- `scripts/configs/_baseline.env` contains defaults.
- Override env files should contain only changed values plus experiment identity.
- `scripts/parse_config.sh` parses, validates, and prints the resolved config.

No runnable job submission flow exists yet.

## 2. Naming Convention

Experiment override files should use this naming shape:

```text
<TAG>_<runmode>_<sharding>_lin<LIN>_lout<LOUT>_bs<BS>_n<NODES>_c<CORES>_mem<MEM>_tprof<0|1>_mprof<0|1>.env
```

Example:

```text
TPCHECK_verify_tp2_lin10_lout15_bs1_n2_c32_mem600g_tprof0_mprof0.env
```

The tag prefix is the user-facing config name used by commands like:

```bash
bash scripts/submit_experiment.sh TPCHECK
```

The command exists only as a non-submitting skeleton right now.

## 3. Parser Behavior

`scripts/parse_config.sh` is local-only and does not submit jobs.

- It sources `scripts/configs/_baseline.env` first.
- Then it sources exactly one `TAG_*.env` override.
- Override files should contain only changed values plus experiment identity.
- `ACTIVE_MODEL_PATH` is derived from `WEIGHTS_PRECISION`.
- `PP_SIZE` must remain `1`.
- The parser validates known enum values and required fields.
- It has human-readable output by default.
- `scripts/parse_config.sh --format env <TAG>` emits shell-safe resolved env.
- Future sbatch generation will consume the env output.
- The parser still does not create files or submit jobs.

`scripts/submit_experiment.sh` consumes `scripts/parse_config.sh --format env <TAG>`. Dry-run sbatch generation is the first consumer of machine-readable env output. The generated sbatch is not yet a real runner.

`scripts/submit_experiment.sh` generates the sbatch. The generated sbatch delegates execution to `scripts/run_case.sh`, and `scripts/run_case.sh` consumes the resolved config snapshot.

## 4. RUN_MODE

Supported schema values:

- `verify`: run correctness verification against `GPU_REFERENCE_PATH`.
- `bench`: run timing benchmark only.
- `both`: run a small verification case first, then run benchmark.
- `generate`: run inference and print generated output without verification or benchmark aggregation.

`RUN_MODE` controls the high-level behavior. A separate `VERIFY_ENABLED` field is intentionally not included.

## 5. INFERENCE_ARCHITECTURE

Supported schema values:

- `direct_native`: direct PyTorch `model.forward` execution path.
- `server_client`: future mode for a persistent service-style execution path.

`server_client` is future-facing and not implemented yet.

## 6. STREAMING

`STREAMING` means token/output streaming: exposing tokens or results incrementally as they are generated instead of only after the full decode finishes.

For `direct_native`, `STREAMING` should currently remain `0`.

## 7. SESSION_MODE

`SESSION_MODE` means one allocation/session can run multiple cases without restarting everything.

This is an important future feature for reducing repeated setup cost, but it is not implemented yet.

## 8. BATCH_SIZE

`BATCH_SIZE` is the number of prompts processed together.

It is not the same thing as concurrency.

`CONCURRENCY` is intentionally not included right now. The first clean config layer should describe the model workload, not a future serving/request scheduler.

## 9. Verification Controls

Verification behavior is controlled by `RUN_MODE` plus `GPU_REFERENCE_PATH` for now.

The following fields are intentionally not included:

- `VERIFY_ENABLED`
- `COMPARE_TOKENS`
- `COMPARE_LOGITS`

Token comparison is implied by `RUN_MODE=verify` or `RUN_MODE=both` until the clean verification design becomes more explicit.

## 10. Parallelism Scope

Pipeline parallelism is schema-visible but out of scope.

- `PP_SIZE` must remain `1`.
- `TP_SIZE`, `DP_SIZE`, and `EP_SIZE` describe the intended CPU sharding shape.
- `SHARDING_MODE` is the high-level mode name used to select behavior.

## 11. REAL_RUN gate and sharded checkpoint path

`REAL_RUN` controls whether `submit_experiment.sh` generates a dry-run sbatch (calls `run_case.sh` placeholder) or a real distributed sbatch (calls `scripts/run_native_distributed.sh`). Default is `0`. TPCHECKREAL sets `REAL_RUN=1`.

`SHARDED_CKPT_PATH` is the directory containing per-rank converted safetensor shards (filenames `model{rank}-mp{world_size}.safetensors`), produced offline by `../DeepSeek-V3.2/inference/convert.py`. Required when `REAL_RUN=1`. Should also contain `tokenizer.json` / `tokenizer_config.json`. Empty in the baseline.

`DEQUANT_FP8_WEIGHTS` controls optional pre-dequantization of FP8 weights to BF16 in place once at load. Supported values are `all` (matches the legacy TP2 token-exact convention) and `none` (keeps FP8; the per-call FP32 fallback in `src/overrides/kernel.py` runs instead). The legacy `dense` scope is intentionally not exposed in the clean lane until DP2 EP-off support lands.

## 12. Native ModelArgs config

`MODEL_ARGS_CONFIG_PATH` points at the native DeepSeek ModelArgs JSON consumed directly by `../DeepSeek-V3.2/inference/model.py`. Default: `../DeepSeek-V3.2/inference/config_671B_v3.2.json`.

- Keys in this JSON must be `ModelArgs` field names. No alias mapping. Unknown keys are rejected loudly.
- Paths are resolved relative to `CLEAN_ROOT` when not absolute.
- `MODEL_ARGS_CONFIG_PATH` is *not* the HF-style `<ACTIVE_MODEL_PATH>/config.json`. The HF file is checkpoint metadata only and is not used by the native ModelArgs path.
- `dtype`, `max_batch_size`, and `max_seq_len` must come from runtime/experiment configuration (resolved env + CLI), not from this JSON. `max_seq_len` in particular is a runtime KV/RoPE allocation limit and must not be auto-mapped from any checkpoint `max_position_embeddings` value.

## 13. Precision Scope

`WEIGHTS_PRECISION` and `KV_CACHE_DTYPE` are separate concepts.

bf16 KV cache is not implemented end-to-end yet. For now, `KV_CACHE_DTYPE="fp8"` remains the expected baseline value.

Do not treat this document as complete yet.
