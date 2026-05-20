# Migration

Current phase: first real native TP2 verification.

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
- Native inference preflight layer (`scripts/model_preflight.py`).
- Clean venv validated end-to-end.
- Model construction smoke script (`scripts/model_construct_smoke.py`).
- Native ModelArgs JSON path (`MODEL_ARGS_CONFIG_PATH`) replacing earlier HF alias mapper.
- TP2 distributed runtime: `src/clean_inference/native_runtime.py` (thread env, dist init ordering, ModelArgs build for case, Transformer construction).
- TP2 weight loading: `src/clean_inference/weight_loading.py` (rank-aware safetensor shard load via `safetensors.torch.load_model`; missing/unexpected keys reported; optional pre-dequant of FP8 to BF16).
- Tokenizer loading: `src/clean_inference/tokenization.py` (`PreTrainedTokenizerFast` from `tokenizer.json`, special-token kwargs from `tokenizer_config.json`, `add_special_tokens=False`).
- Greedy decode loop: `src/clean_inference/generation.py` (prefill + per-token decode, `logits.argmax(-1)`).
- Clean re-write of `dequant_weights.py` into `src/overrides/dequant_weights.py` (block-broadcast FP8 → BF16, no `repeat_interleave` grid).
- `scripts/native_verify.py` (full pipeline with `--no-load-weights` and `--no-generate` safety flags).
- `scripts/run_native_distributed.sh` (torchrun launcher mirroring the legacy srun shape).
- `REAL_RUN` gate in `_baseline.env` / `parse_config.sh` / `submit_experiment.sh`. `REAL_RUN=1` generates a real-distributed sbatch that calls `run_native_distributed.sh`; `REAL_RUN=0` keeps the dry-run placeholder.
- `SHARDED_CKPT_PATH` plumbed through (per-rank TP shard directory; empty in baseline, set in TPCHECKREAL).
- `TPCHECKREAL` override config: tp2, fp8, BS=1, Lin=10, Lout=15, 2 nodes × 32 cores, 750G mem, 2h walltime.

Next planned phase: actual TPCHECKREAL run inside a 2-node Slurm allocation, in stages — `--no-load-weights` first (already runs locally), then `--no-generate` weight-load smoke, then full decode against `prompt1_bs1_lin10_lout15/case_0001.json`.

Not yet started:
- DP2 / DP2 EP-on clean paths.
- Additional reference cases (BS=4, 4 prompts, logits/layer/attention/MoE outputs).
- Perf benchmark mode (`bench` / `both` in the clean runner).
- Profiling lane (mem + time).

No DeepSeek model code has been copied. `src/overrides/` contains clean re-writes (`kernel.py`, `fast_hadamard_transform.py`, `dequant_weights.py`); nothing under `../structured_cpu_run/without_vllm/overrides/` is imported at runtime. The legacy TP2 shards directory (`../structured_cpu_run/artifacts/deepseek-v3.2-mp2-active/`) is consumed only as a data path via `SHARDED_CKPT_PATH`.
