# Migration

Current phase: native CPU inference import smoke test.

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

Next planned phase: load-only model construction smoke test.

Not yet started:
- Model construction.
- Weight loading.
- `torch.distributed` launch.
- Real token generation.

No code has been copied yet.
