# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`structured_cpu_run_clean` is a **supervised, incremental clean-lane reimplementation** of CPU DeepSeek-V3.2 inference. The original `../structured_cpu_run/` is the historical source of truth and remains the only working CPU inference path — this lane is being built from scratch around a better config + execution shape and has not yet copied the actual model-driving code over.

What exists today is a skeleton:

- Env-file config schema + parser (`scripts/parse_config.sh`).
- Non-submitting `submit_experiment.sh` that resolves the config and generates a placeholder sbatch.
- `scripts/run_case.sh` placeholder that dispatches by `RUN_MODE` but does not run inference.
- Reference case JSON inspector and a **mock** verification runner (`scripts/run_verify.py`).
- Clean override modules in `src/overrides/` (only `kernel.py`, `fast_hadamard_transform.py` so far) plus an import-order smoke test.

What is **not** implemented: model construction, weight loading, `torch.distributed` launch, real token generation. `run_case.sh` and the generated sbatch are intentionally non-functional placeholders.

Read `docs/CURRENT_INVENTORY.md` to understand what lives in the legacy `../structured_cpu_run/` lane and what is intended for future copy.

## Project rules (load-bearing — see `AGENT_RULES_FOR_THE_PROJECT.md`)

- **No giant rewrites.** Every change must be small and reviewable, target **<20 lines** of code.
- Each step must fit one of: inspect only, create new file, copy existing file in, add documentation, move one small piece of logic behind a wrapper, add one test/check, run a diff and explain.
- Every step ends with **"Do not proceed further until Akash approves."** Do not chain multiple steps together autonomously.
- Do not modify `../FlashMLA`, `../FlashMLA_CPU`, `../DeepSeek-V3.2`, or `../structured_cpu_run`.
- Pipeline parallelism is out of scope — `PP_SIZE` must remain `1`.

After making a change, also update `MIGRATION.md` (phase tracker) and `COMMANDS.md` (command list) when the change introduces or alters either.

## Common commands

Environment setup (Python 3.12+ required; default login-node `python3` is too old):

```bash
module load python/3.12.1            # or: PYTHON_BIN=/path/to/python3.12
bash scripts/setup_venv.sh           # refuses to run if .venv already exists; rm -rf .venv to retry
```

Dry-run an experiment end-to-end (writes resolved env + placeholder sbatch, does **not** submit):

```bash
bash scripts/submit_experiment.sh TPCHECK
bash scripts/run_case.sh results_clean/resolved_configs/TPCHECK_resolved.env
.venv/bin/python scripts/inference_import_smoke.py \
    --resolved-config results_clean/resolved_configs/TPCHECK_resolved.env
```

Inspect / parse only:

```bash
bash scripts/parse_config.sh TPCHECK                 # human-readable
bash scripts/parse_config.sh --format env TPCHECK    # machine-readable, sourceable
python scripts/inspect_reference_cases.py --reference-root verification/references/
python3 scripts/run_verify.py --resolved-config results_clean/resolved_configs/TPCHECK_resolved.env --mock-mode golden
```

There is no test suite or linter yet. `tests/` is empty.

Temporary fallback when the clean `.venv` is not yet built: the old known-good venv lives at `/home/akashpt/DeepSeekRun/structured_cpu_run/without_vllm/.venv/`.

## Architecture

### Config flow (the only "real" flow today)

1. `scripts/configs/_baseline.env` holds all defaults.
2. A `<TAG>_*.env` override file (naming: `<TAG>_<runmode>_<sharding>_lin<LIN>_lout<LOUT>_bs<BS>_n<NODES>_c<CORES>_mem<MEM>_tprof<0|1>_mprof<0|1>.env`) holds only changed values plus experiment identity.
3. `parse_config.sh` sources baseline then the single matching override, derives `ACTIVE_MODEL_PATH` from `WEIGHTS_PRECISION`, validates enums and required fields, and emits either human or `KEY=$'value'` env output (consumed via `eval`).
4. `submit_experiment.sh` calls the parser, snapshots the resolved env to `results_clean/resolved_configs/<TAG>_resolved.env`, and writes a placeholder sbatch under `tmp/sbatch/` that would `bash scripts/run_case.sh <resolved_config>`.
5. `run_case.sh` sources the resolved config and dispatches by `RUN_MODE` (`verify`/`bench`/`both`/`generate`) and `INFERENCE_ARCHITECTURE` (`direct_native` only; `server_client` is schema-visible but exits non-zero). `verify` currently calls the mock `run_verify.py`.

The resolved-config env file is the **handoff contract** between shell and Python — `run_verify.py` and `inference_import_smoke.py` both parse it via `parse_resolved_env` (`scripts/run_verify.py:43`).

See `docs/CONFIG_SCHEMA.md` for field semantics: `RUN_MODE`, `INFERENCE_ARCHITECTURE`, `STREAMING`, `SESSION_MODE`, `BATCH_SIZE` (not concurrency), precision (`WEIGHTS_PRECISION` ≠ `KV_CACHE_DTYPE`), and sharding (`TP_SIZE`/`DP_SIZE`/`EP_SIZE`, `SHARDING_MODE`).

### Override import ordering (the key inference-bring-up invariant)

CPU inference depends on injecting clean override modules **before** the upstream DeepSeek inference modules in `sys.path`. The intended order is:

1. `src/overrides/` (clean-owned)
2. `<DEEPSEEK_REPO>/inference` (from `_baseline.env`, currently `../DeepSeek-V3.2`)

`scripts/inference_import_smoke.py` enforces this: it confirms `kernel` and `fast_hadamard_transform` resolve to `src/overrides/` while `model` resolves to the DeepSeek inference dir. It deliberately stops short of instantiating `Transformer` or loading weights. This is the test gate for every future override copy.

`src/overrides/kernel.py` carries optimization history in its docstrings (FP8 dequant path tradeoffs across Step 7a / 7b iterations) — preserve those notes when editing.

### Verification (mock today, real later)

Reference cases live under `verification/references/<group>/<case>.json` with the schema enforced by `scripts/inspect_reference_cases.py:load_case`. Required fields include `case_id`, `prompt_text`, `lin_tokens`/`lout_tokens`, `batch_size`, `sampling`, `expected_output_token_ids`, `source`.

`run_verify.py` loads cases, generates mock tokens (`random` mode = deterministic hash-seeded random; `golden` mode = echoes expected, only useful for plumbing checks), compares to `expected_output_token_ids`, and writes `results_clean/results/<TAG>/verify_results.json`. **No real model inference yet.**

Future verification targets (per `docs/CURRENT_INVENTORY.md`): >1 prompt, batch size 4, logits, layer outputs, attention outputs, MoE outputs.

## Directory map

- `scripts/` — parser, dry-run entrypoint, placeholders, smoke tests. `scripts/lib/` is reserved/empty.
- `scripts/configs/` — `_baseline.env` plus override env files (one per experiment tag).
- `src/overrides/` — clean-owned CPU override modules (loaded before DeepSeek inference).
- `src/checkpoint/` — reserved/empty (future weight loading).
- `verification/references/` — committed GPU reference case data, grouped by `<promptN>_bs<B>_lin<L>_lout<L>` naming.
- `examples/generated/` — committed examples of what `submit_experiment.sh` produces (the live versions go to gitignored locations).
- `results_clean/` — runtime outputs: `logs/`, `results/`, `profiles/`, `resolved_configs/`, `sbatch/` (all gitignored).
- `tmp/sbatch/` — gitignored generated dry-run sbatch files.
- `docs/` — design docs. `CONFIG_SCHEMA.md`, `CURRENT_INVENTORY.md`, `VERIFICATION.md`, `OVERRIDES.md`, `ENVIRONMENT.md` have content; `EXECUTION_FLOW.md`, `SHARDING.md`, `PROFILING.md`, `RESULTS_FORMAT.md` are TODO stubs.

`MIGRATION.md` is the running phase tracker. `COMMANDS.md` is the running command index. `diff.diff` is a working-directory scratch file, not load-bearing.
