# Migration

Current phase: model construction smoke.

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
- Native inference preflight layer (`scripts/model_preflight.py`).
- Clean venv validated end-to-end.
- Checkpoint config mapping (`src/clean_inference/model_config.py`): initial HF-style `config.json` alias-mapper, since superseded.
- Model construction smoke script (`scripts/model_construct_smoke.py`).
- Switched the ModelArgs source of truth to the **native** DeepSeek JSON at `MODEL_ARGS_CONFIG_PATH` (default `../DeepSeek-V3.2/inference/config_671B_v3.2.json`). The HF-style alias mapper was removed. `ModelArgs(**native_config)` matches the upstream pattern in `inference/generate.py`. `dtype`, `max_batch_size`, and `max_seq_len` are explicit runtime overrides — `max_seq_len` is no longer auto-derived from `max_position_embeddings`.

Next planned phase: clean weight-loading smoke (first explicit safetensor weight load into a constructed `Transformer`).

Not yet started:
- Safetensor weight loading into model.
- `torch.distributed` launch.
- Real token generation.

No DeepSeek model code has been copied yet. The clean override modules in `src/overrides/` are owned by the clean lane.
