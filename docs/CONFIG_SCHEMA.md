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

## 11. Precision Scope

`WEIGHTS_PRECISION` and `KV_CACHE_DTYPE` are separate concepts.

bf16 KV cache is not implemented end-to-end yet. For now, `KV_CACHE_DTYPE="fp8"` remains the expected baseline value.

Do not treat this document as complete yet.
