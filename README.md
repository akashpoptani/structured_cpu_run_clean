# structured_cpu_run_clean

This repo is a supervised clean migration lane for CPU DeepSeek-V3.2 inference.

It is not production yet.

The original `structured_cpu_run` directory is the historical source of truth, but it is not a runtime dependency for this clean lane.

No giant rewrites are allowed.

## Current Status

Implemented:

- Config parser.
- Dry-run sbatch generation.
- `run_case.sh` placeholder.
- Verification reference data.
- Mock verification runner.
- Clean override import smoke test.
- Clean venv setup script with `--reset`.
- Shared `src/clean_inference/` config/import/file-inspection helpers.
- Native CPU inference preflight script.

Not implemented yet:

- Actual model construction.
- Weight loading.
- Real token generation.
- `torch.distributed` launch.

## Current Command Flow

Create the clean venv first:

```bash
module load python/3.12.1
bash scripts/setup_venv.sh --reset
```

Recommended validation flow:

```bash
bash scripts/submit_experiment.sh TPCHECK
.venv/bin/python -c "import torch; print('torch', torch.__version__)"
.venv/bin/python scripts/inference_import_smoke.py --resolved-config results_clean/resolved_configs/TPCHECK_resolved.env
.venv/bin/python scripts/model_preflight.py --resolved-config results_clean/resolved_configs/TPCHECK_resolved.env
PYTHON=.venv/bin/python bash scripts/run_case.sh results_clean/resolved_configs/TPCHECK_resolved.env
```

`model_preflight.py` is the current last preflight gate before model construction. It does not instantiate `Transformer` and does not load weights. See `docs/INFERENCE.md` for the full bring-up stage list.

## Directory Map

- `scripts/`: local parser, dry-run entrypoint, placeholders, and smoke tests.
- `scripts/configs/`: baseline and override env configs.
- `src/overrides/`: clean-owned CPU override modules.
- `verification/references/`: committed GPU reference case data.
- `examples/generated/`: committed examples of generated artifacts.
- `results_clean/`: runtime results and resolved config snapshots.
- `tmp/sbatch/`: generated dry-run sbatch files.
- `docs/`: design and migration documentation.

## Safety Rules

- No giant rewrites.
- Keep changes small and reviewed.
- Do not change `FlashMLA` or `FlashMLA_CPU` right now.
- Do not modify `DeepSeek-V3.2`.
- Pipeline parallelism is out of scope.
