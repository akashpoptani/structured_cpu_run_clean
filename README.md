# structured_cpu_run_clean

A clean, supervised re-implementation of CPU DeepSeek-V3.2 inference. The clean lane drives upstream `../DeepSeek-V3.2/inference/model.py` through clean-owned overrides and a small runtime, with no Python imported at runtime from the legacy `../structured_cpu_run/` lane.

## Current milestone

First real native TP2 token-exact verification PASSED. The `TPCHECKREAL` config (2 nodes × 16 cores × 400 GB, fp8 weights, `DEQUANT_FP8_WEIGHTS=none`) ran greedy decode for the committed reference prompt and produced the expected 15/15 output token IDs against `verification/references/prompt1_bs1_lin10_lout15/case_0001.json`. The `DEQUANT_FP8_WEIGHTS=none` per-call FP32 fallback is correctness-proven but ~10× slower than the legacy `=all` BF16 pre-dequant fast path.

## One-line command

Submit a real experiment:

```bash
bash scripts/submit_experiment.sh TPCHECKREAL
```

`submit_experiment.sh` validates the config, writes a resolved env snapshot, writes an sbatch, calls `sbatch`, captures the job id, and records run metadata. There is no manual `sbatch` step.

Check a job's status by id (planned, not yet implemented):

```bash
bash scripts/check_run.sh <job_id>     # LATER, not supported right now
```

For now: `squeue -j <job_id>` and `tail -F results_clean/logs/<RUN_LABEL>_<job_id>.out`.

## Setup

Python 3.12 is required. On lighthouse:

```bash
module load python/3.12.1
bash scripts/setup_venv.sh --reset
```

- `--reset` first deletes any existing `.venv` and rebuilds it. Use it when you want a known-clean environment or after `requirements.txt` changed.
- Without `--reset`, the script refuses to overwrite an existing `.venv` and prints how to clean up.
- If Lmod is unavailable: `PYTHON_BIN=/path/to/python3.12 bash scripts/setup_venv.sh --reset`.

## Config model

- `scripts/configs/_baseline.env` defines the defaults for every field. `REAL_RUN=1` is the default; submissions go through the real distributed path.
- Each experiment is one override file `scripts/configs/<TAG>_<runmode>_<sharding>_lin<LIN>_lout<LOUT>_bs<BS>_n<NODES>_c<CORES>_mem<MEM>_tprof<0|1>_mprof<0|1>.env` containing only the values that differ from baseline plus experiment identity.
- `scripts/parse_config.sh` enforces the schema (required fields, enum values, `PP_SIZE=1` invariant) and emits either human-readable output or `KEY=$'value'` env lines.
- To run a new experiment: copy an existing override, edit fields, then `bash scripts/submit_experiment.sh <TAG>`. Do not edit Python scripts for normal experiment-shape changes.

## Current real configs

| Tag | Purpose | Behavior |
|---|---|---|
| `TPCHECKREAL_NOLOAD` | TP2 construct-only smoke | dist init + `Transformer(args)`; skip weight load and decode. ~few seconds wall. |
| `TPCHECKREAL_NOGEN` | TP2 weight-load smoke | dist init + construct + load `model{rank}-mp2.safetensors`; skip decode. ~5–10 min wall. |
| `TPCHECKREAL` | TP2 verify (token-exact) | full pipeline + compare against `expected_output_token_ids`. Known-good against `case_0001`. ~60 min wall at `DEQUANT_FP8_WEIGHTS=none`. |
| `TPGEN` | TP2 generate | full pipeline; emits tokens + decoded text, no compare. ~60 min wall. |
| `TPBENCH` | TP2 bench | full pipeline; emits TTFT / TPOT / tokens-per-second. ~60 min wall. |
| `TPBOTH` | TP2 verify then bench | verify first; if every case passes, reuse decode timings to compute bench. ~60 min wall (one decode pass). |

All three use: `SHARDING_MODE=tp2`, `TP_SIZE=2`, `WEIGHTS_PRECISION=fp8`, `DEQUANT_FP8_WEIGHTS=none`, `SHARDED_CKPT_PATH=/scratch/.../deepseek-v3.2-mp2-rerun`, `SBATCH_PARTITION=project_l`, `SBATCH_ACCOUNT=kdur`, 2 nodes × 16 cpus × 400 GB.

The legacy `TPCHECK` config is from an earlier mock/dry-run bring-up stage and is not the main launcher target anymore. The mock path through `run_case.sh` still exists as internal/debug code.

## Execution flow

What happens when you type `bash scripts/submit_experiment.sh <TAG>`:

```
scripts/submit_experiment.sh <TAG>
  └─ runs scripts/parse_config.sh <TAG>            # validates + prints config
  └─ runs scripts/parse_config.sh --format env <TAG>
       └─ writes results_clean/resolved_configs/<TAG>_resolved.env
  └─ writes tmp/sbatch/<TAG>_..._.sbatch
  └─ runs `sbatch <sbatch>`
       └─ captures Submitted batch job <JOB_ID>
  └─ writes .last_job, .last_sbatch, .last_resolved_config
  └─ writes results_clean/runs/<TAG>/<JOB_ID>/run_metadata.env

Generated sbatch (inside Slurm):
  #SBATCH … (partition, account, nodes, cpus, mem, time)
  srun --nodes=N --ntasks=N --ntasks-per-node=1 \
      bash scripts/run_native_distributed.sh <resolved_config>

scripts/run_native_distributed.sh (per node):
  └─ sources <resolved_config>
  └─ activates .venv (or honors $PYTHON / falls back to .venv/bin/python)
  └─ exports OMP_NUM_THREADS / OMP_PROC_BIND / OMP_PLACES / MKL_NUM_THREADS
  └─ computes MASTER_ADDR / MASTER_PORT from SLURM_JOB_NODELIST / SLURM_JOB_ID
  └─ translates NATIVE_NO_LOAD_WEIGHTS / NATIVE_NO_GENERATE into CLI flags
  └─ exec python -m torch.distributed.run \
         --nnodes=N --nproc-per-node=1 --node-rank=$SLURM_NODEID \
         --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT \
         scripts/native_verify.py --resolved-config <…> --reference-group <…>
             [--no-load-weights | --no-generate]

scripts/native_verify.py (per rank):
  └─ parses resolved config + detects rank/world from env
  └─ enumerates reference cases in --reference-group (loops if --case-id omitted)
  └─ setup_thread_env: torch.set_num_threads / default dtype / manual_seed(0)
  └─ initialize_distributed_if_needed: dist.init_process_group("gloo")
      BEFORE Transformer construction when SHARDING_MODE=tp2 (so model.world_size
      is baked into ColumnParallel/RowParallel layers)
  └─ import_deepseek_modules: prepends src/overrides/ before
      <DEEPSEEK_REPO>/inference on sys.path, then imports model/kernel/
      fast_hadamard_transform
  └─ build_modelargs_for_case: loads MODEL_ARGS_CONFIG_PATH (native
      config_671B_v3.2.json), instantiates ModelArgs(**native_config), overrides
      dtype (from WEIGHTS_PRECISION), max_batch_size (from case), max_seq_len
      (= Lin + Lout)
  └─ construct_transformer: model.Transformer(args)
  └─ unless NATIVE_NO_LOAD_WEIGHTS=1:
       load_weights_into_transformer: safetensors.torch.load_model(
         model, SHARDED_CKPT_PATH/model{rank}-mp{world_size}.safetensors,
         strict=False); reports missing/unexpected keys
       maybe_dequantize_fp8: optional DEQUANT_FP8_WEIGHTS pass
  └─ unless NATIVE_NO_GENERATE=1 (or NATIVE_NO_LOAD_WEIGHTS=1):
       load_tokenizer + encode_prompt(add_special_tokens=False)
       greedy_decode(prompt_tokens, lout)
       compare generated vs expected_output_token_ids
  └─ rank 0 writes results_clean/results/<TAG>/native_verify_results.json
```

## Files and directories created by runs

| Path | Lifetime |
|---|---|
| `tmp/sbatch/<TAG>_*.sbatch` | per submit (gitignored) |
| `results_clean/resolved_configs/<TAG>_resolved.env` | per submit (gitignored) |
| `results_clean/logs/<RUN_LABEL>_<JOB_ID>.{out,err}` | per job (gitignored) |
| `results_clean/results/<TAG>/native_verify_results.json` | per job, rank 0 only (gitignored) |
| `results_clean/runs/<TAG>/<JOB_ID>/run_metadata.env` | per submit (gitignored) |
| `.last_job` / `.last_sbatch` / `.last_resolved_config` | latest-only pointers (gitignored) |

Reference cases under `verification/references/<group>/<case>.json` are committed. `examples/generated/` contains a checked-in example resolved env + sbatch for documentation.

## Important: do not run native_verify.py directly for TP2

Calling `.venv/bin/python scripts/native_verify.py --resolved-config <TPCHECKREAL_resolved.env> --reference-group <…>` from a normal shell starts a **single-rank** process with `WORLD_SIZE=1`. The weight loader would then look for `model0-mp1.safetensors` (which does not exist for the TP2 shards) and the model would not be sharded across ranks. Always go through `submit_experiment.sh` for real TP2 runs. The construct-only `--no-load-weights` path is the only safe direct-Python invocation, and even then it only validates the build pipeline, not the distributed runtime.

## Next work

- Additional reference cases (more prompts, batch size 4, logit/layer/attention/MoE comparisons).
- Fast path: `DEQUANT_FP8_WEIGHTS=all` on a slot with ≥750 GB per rank (AMX BF16 brgemm).
- DP2 and DP2 EP-on clean paths (DP modes invert the dist-init ordering; not yet implemented).
- Memory and time profiling lanes.
- `scripts/check_run.sh` helper to report job state + tail logs by job id.

## Key reference docs

- [docs/INFERENCE.md](docs/INFERENCE.md) — the inference bring-up stages and what each script does.
- [docs/CONFIG_SCHEMA.md](docs/CONFIG_SCHEMA.md) — every config field and its allowed values.
- [docs/VERIFICATION.md](docs/VERIFICATION.md) — verification reference format and the native verifier.
- [docs/OVERRIDES.md](docs/OVERRIDES.md) — `src/overrides/` policy.
- [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) — the clean venv.
- [COMMANDS.md](COMMANDS.md) — terse command index.
- [MIGRATION.md](MIGRATION.md) — phase tracker.

## Safety rules

- No giant rewrites; each change is small and reviewable.
- No imports at runtime from `../structured_cpu_run/`. The legacy artifacts dir is consumed only as a data path (`SHARDED_CKPT_PATH`).
- Do not modify `../FlashMLA`, `../FlashMLA_CPU`, `../DeepSeek-V3.2`, or `../structured_cpu_run`.
- Pipeline parallelism is out of scope (`PP_SIZE` must stay `1`).
