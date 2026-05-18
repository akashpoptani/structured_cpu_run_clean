# Migration

Current phase: clean verification runner with mock inference.

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

Next planned phase: replace mock inference with minimal clean CPU inference entrypoint.

Not yet started:
- Actual model loading.
- `torch.distributed` launch.
- Real token generation.
- Code copy from old `verify_cpu.py`.

No code has been copied yet.
