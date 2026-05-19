# Migration

Current phase: native inference preflight layer.

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
- Clean venv setup script with `--reset`.
- Shared `src/clean_inference/` utilities (config parsing, override import setup, model-directory inspection).
- Import smoke refactor onto shared utilities.
- Model preflight script (`scripts/model_preflight.py`).

Next planned phase: model construction smoke test (first explicit `Transformer(ModelArgs(...))` instantiation with controlled minimal args and no weight loading).

Not yet started:
- Transformer construction.
- Weight loading.
- `torch.distributed` launch.
- Real token generation.

No DeepSeek model code has been copied yet. The clean override modules in `src/overrides/` are owned by the clean lane.
