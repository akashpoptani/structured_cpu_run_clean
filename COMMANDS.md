# Commands

See [README.md](README.md) for the full flow. This file is the terse command index.

## Real native runs (current)

```bash
bash scripts/submit_experiment.sh TPCHECKREAL_NOLOAD   # construct only, ~few seconds wall
bash scripts/submit_experiment.sh TPCHECKREAL_NOGEN    # weight-load only, ~5-10 min wall
bash scripts/submit_experiment.sh TPCHECKREAL          # full token-exact decode, ~60 min wall
```

`submit_experiment.sh <TAG>` is the single user command. It validates the config, writes the resolved env snapshot, writes the sbatch, calls `sbatch`, captures the job id, and writes run metadata under `results_clean/runs/<TAG>/<JOB_ID>/`. Do not call `sbatch` directly.

After submit:

```bash
cat .last_job                       # latest submitted JOB_ID
cat .last_sbatch                    # latest generated sbatch path
cat .last_resolved_config           # latest resolved env path
squeue -j "$(cat .last_job)"
tail -F results_clean/logs/TPCHECKREAL_$(cat .last_job).out
cat results_clean/results/TPCHECKREAL/native_verify_results.json
```

## Legacy mock path (earlier bring-up stage, not the main launcher)

`TPCHECK` was the original mock/dry-run config. It still exists but submitting it now also goes through the real-distributed sbatch (`REAL_RUN=1` is the new baseline default). For the in-process mock verification runner used during early bring-up:

```bash
PYTHON=.venv/bin/python bash scripts/run_case.sh results_clean/resolved_configs/TPCHECK_resolved.env
```

Clean venv setup:
```bash
cd /home/akashpt/DeepSeekRun/structured_cpu_run_clean
module load python/3.12.1
bash scripts/setup_venv.sh --reset
```

`--reset` removes any existing `.venv` automatically and recreates it.
Without `--reset`, the script refuses to overwrite an existing `.venv`.

Clean venv validation:
```bash
.venv/bin/python -c "import torch; print('torch', torch.__version__)"
.venv/bin/python scripts/inference_import_smoke.py --resolved-config results_clean/resolved_configs/TPCHECK_resolved.env
.venv/bin/python scripts/model_preflight.py --resolved-config results_clean/resolved_configs/TPCHECK_resolved.env
PYTHON=.venv/bin/python bash scripts/run_case.sh results_clean/resolved_configs/TPCHECK_resolved.env
```

If you are not using an Lmod Python module, pass an explicit interpreter:
```bash
PYTHON_BIN=/path/to/python3.12 bash scripts/setup_venv.sh
PYTHON_BIN=/path/to/python3.12 bash scripts/setup_venv.sh --reset
```

If setup failed partway and you do not want `--reset`, remove the partial venv before retrying:
```bash
rm -rf .venv
```

Temporary known-good old venv:
```bash
/home/akashpt/DeepSeekRun/structured_cpu_run/without_vllm/.venv/bin/python scripts/inference_import_smoke.py --resolved-config results_clean/resolved_configs/TPCHECK_resolved.env
```

submit_experiment.sh calls parse_config.sh file:
```bash
bash scripts/parse_config.sh TPCHECK
bash scripts/parse_config.sh --format env TPCHECK
```

Placeholder case runner:
```bash
bash scripts/run_case.sh results_clean/resolved_configs/TPCHECK_resolved.env
```

Verification reference inspector:
```bash
python scripts/inspect_reference_cases.py --reference-root verification/references/
python scripts/inspect_reference_cases.py --reference-root verification/references/ --format json
```

Mock clean verification runner:
```bash
python3 scripts/run_verify.py --resolved-config results_clean/resolved_configs/TPCHECK_resolved.env
python3 scripts/run_verify.py --resolved-config results_clean/resolved_configs/TPCHECK_resolved.env --mock-mode golden
python3 scripts/run_verify.py --resolved-config results_clean/resolved_configs/TPCHECK_resolved.env --format json
```

Native CPU inference import smoke test:
```bash
.venv/bin/python scripts/inference_import_smoke.py --resolved-config results_clean/resolved_configs/TPCHECK_resolved.env
```

Model preflight (no model construction, no weight loading):
```bash
.venv/bin/python scripts/model_preflight.py --resolved-config results_clean/resolved_configs/TPCHECK_resolved.env
.venv/bin/python scripts/model_preflight.py --resolved-config results_clean/resolved_configs/TPCHECK_resolved.env --format json
```

`model_preflight.py` inspects the environment, paths, override imports, model
directory metadata, ModelArgs / Transformer signatures, and module globals. It
does not instantiate Transformer and does not load weights.

`model_preflight.py` also loads the native DeepSeek ModelArgs JSON pointed at by
`MODEL_ARGS_CONFIG_PATH` (default `../DeepSeek-V3.2/inference/config_671B_v3.2.json`)
and prints which ModelArgs fields it populated and which fell back to dataclass
defaults. The HF-style `<ACTIVE_MODEL_PATH>/config.json` is *not* consumed for
ModelArgs; preflight only notes its existence as checkpoint metadata.

## Staged TPCHECKREAL real-run sequence

No `sbatch` is submitted automatically — Akash submits each stage manually. TPCHECK stays as the mock/dry-run config; TPCHECKREAL is the real config.

### Stage 1 — Safe construct-only local check (no Slurm, no weights)
```bash
.venv/bin/python scripts/native_verify.py \
    --resolved-config results_clean/resolved_configs/TPCHECKREAL_resolved.env \
    --reference-group prompt1_bs1_lin10_lout15 \
    --no-load-weights
```
Run after `bash scripts/submit_experiment.sh TPCHECKREAL` regenerates the resolved config. Confirms model construction + ModelArgs source under the real config.

### Stage 2 — Slurm weight-load-only smoke (NATIVE_NO_GENERATE=1)
Edit `scripts/configs/TPCHECKREAL_*.env`, set `NATIVE_NO_GENERATE=1`, then:
```bash
bash scripts/submit_experiment.sh TPCHECKREAL          # regenerates resolved_config + sbatch
sbatch tmp/sbatch/TPCHECKREAL_*.sbatch                 # Akash submits
```
`run_native_distributed.sh` reads `NATIVE_NO_GENERATE=1` from the resolved env and passes `--no-generate` to `native_verify.py`. Validates rank-aware shard loading + optional pre-dequant. Reset to `0` for Stage 3.

### Stage 3 — Full decode (defaults)
With `NATIVE_NO_LOAD_WEIGHTS=0` and `NATIVE_NO_GENERATE=0`:
```bash
bash scripts/submit_experiment.sh TPCHECKREAL
sbatch tmp/sbatch/TPCHECKREAL_*.sbatch
```
Loads weights, tokenizes the prompt(s), runs greedy decode for `Lout` tokens, and compares against `expected_output_token_ids` for every case in `--reference-group prompt1_bs1_lin10_lout15`. Result JSON written to `results_clean/results/TPCHECKREAL/native_verify_results.json` (rank 0).

### Inside the generated sbatch
```bash
srun --nodes=$SBATCH_NODES --ntasks=$SBATCH_NODES --ntasks-per-node=1 \
    bash scripts/run_native_distributed.sh <resolved_config>
```
Which invokes:
```bash
python -m torch.distributed.run --nnodes=$SBATCH_NODES --nproc-per-node=1 \
    --node-rank=$SLURM_NODEID --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT \
    scripts/native_verify.py --resolved-config <...> --reference-group prompt1_bs1_lin10_lout15
```
`--case-id` is omitted by default so every case in the group is run sequentially against the constructed/loaded model. Set `NATIVE_CASE_ID=case_0001` (env var to the launcher) to restrict to one case.

TPCHECK remains the mock/dry-run config (`REAL_RUN=0`, generates the placeholder sbatch that calls `run_case.sh`). TPCHECKREAL (`REAL_RUN=1`) is the first real distributed verification config.

Model construction smoke (constructs Transformer from the native ModelArgs JSON; no weights, no forward, no generation):
```bash
.venv/bin/python scripts/model_construct_smoke.py --resolved-config results_clean/resolved_configs/TPCHECK_resolved.env --max-batch-size 1 --max-seq-len 32
```

`model_construct_smoke.py` reads `MODEL_ARGS_CONFIG_PATH` (the native
DeepSeek `config_671B_v3.2.json`), instantiates ModelArgs via
`ModelArgs(**native_config)`, and then applies these explicit runtime
overrides: `dtype` (from `WEIGHTS_PRECISION`), `max_batch_size` (from
`--max-batch-size`), and `max_seq_len` (from `--max-seq-len`). It does not
load weights, does not run forward, and does not call `torch.distributed`.
`max_seq_len` is a runtime allocation/generation limit and must come from the
experiment, not from the checkpoint's `max_position_embeddings`. Construction
may still allocate full-model parameter shapes regardless of reduced batch/seq
overrides.

Default `parse_config.sh` output is human-readable.

`parse_config.sh --format env` output is machine-readable and future scripts can source it.

`parse_config.sh` parses, validates, and prints the resolved config.

`submit_experiment.sh` parses config, writes a resolved config snapshot, and generates a dry-run sbatch under `tmp/sbatch`.

It does not submit jobs. The generated sbatch is placeholder only and does not run inference yet.

Generated sbatches now delegate to `scripts/run_case.sh`. `run_case.sh` is currently a placeholder and does not run inference yet.

`run_case.sh` has placeholder dispatch for `verify`, `bench`, `both`, and `generate`.

Only `direct_native` is allowed as a placeholder path right now. `server_client` is schema-visible but not implemented.

Verification reference data exists under `verification/references/`.

Initial reference group: `verification/references/prompt1_bs1_lin10_lout15/` for 1 prompt, batch size 1, Lin 10, and Lout 15.

No clean verification command exists yet.

`inspect_reference_cases.py` inspects and validates reference-case JSON files. It does not run model inference.

`run_verify.py` reads resolved config, loads reference cases, generates mock output tokens, compares generated token IDs to expected GPU token IDs, and writes a result JSON file.

Random mode is the default and usually fails because it generates deterministic random tokens. Golden mode uses expected tokens and should pass; it is only for plumbing tests.

`run_case.sh` verify now calls `run_verify.py` in random mock mode.

This still does not load model weights.

`inference_import_smoke.py` is the first native CPU inference bring-up check. It verifies clean override import ordering, does not instantiate the model, does not load weights, and does not generate tokens.

Generated artifact examples:

- `examples/generated/TPCHECK_resolved.env`
- `examples/generated/TPCHECK_verify_tp2_lin10_lout15_bs1_n2_c32_mem600G_tprof0_mprof0.sbatch`

Live generated files go to `tmp/sbatch/` and `results_clean/resolved_configs/` and are gitignored.

Future intended command style:

```bash
bash scripts/submit_experiment.sh <config_tag>
bash scripts/check_run.sh <job_id> "LATER"
bash scripts/append_to_results.sh <job_id> <config_tag> "LATER"
```
