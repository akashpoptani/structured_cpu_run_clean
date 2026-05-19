# Migration

Current phase: clean venv setup and README polish.

Completed:
- Skeleton repo created and pushed.
- Inventory documentation added.
- Baseline config and TPCHECK override added.
- Non-submitting `submit_experiment.sh` skeleton added.
- `parse_config.sh` human and env output added.
- Dry-run sbatch generation added.
- `run_case.sh` placeholder and mode dispatch added.
- Minimal verification reference data added with descriptive group name.
- Python verification reference inspector added.
- Mock verification runner added.
- Native CPU inference import smoke test with old known-good venv.

Next planned phase: create clean venv and rerun `inference_import_smoke.py` using `.venv/bin/python`.

Not yet started:
- Model construction.
- Weight loading.
- `torch.distributed` launch.
- Real token generation.

No code has been copied yet.
