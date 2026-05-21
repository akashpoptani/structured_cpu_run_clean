# Native Inference Bring-up

Purpose: Track the ordered stages of native CPU inference bring-up in the clean lane.

The clean lane brings up native CPU DeepSeek-V3.2 inference in small, reviewable stages. Each stage must pass before the next is attempted.

## Stages

1. **Clean override import smoke** — `scripts/inference_import_smoke.py`.
   Validates `sys.path` ordering: `kernel` and `fast_hadamard_transform` resolve to `src/overrides/`, and `model` resolves to `<DEEPSEEK_REPO>/inference`. Does not instantiate `Transformer`, does not load weights, does not call `torch.distributed`.

2. **Model preflight** — `scripts/model_preflight.py`.
   Inspects the environment (Python, torch, transformers, safetensors versions), resolved paths, imported override module files, model-directory metadata (`config.json`, tokenizer files, safetensors count and total GiB, index files), DeepSeek model symbols (`Transformer`, `ModelArgs`, `Block`/`TransformerBlock`, `MLA`, `MoE`), `ModelArgs` field defaults, `Transformer.__init__` / `Transformer.forward` / `MLA.forward` / `MoE.forward` signatures, and module globals (`world_size`, `rank`, `local_rank`, `block_size`, `gemm_impl`, `attn_impl`).
   It does **not** instantiate `Transformer`, does **not** load weights, does **not** call `torch.distributed`, and does **not** run generation. Heavy safetensor payloads are never read — only file sizes and names.

3. **Model construction smoke** — `scripts/model_construct_smoke.py`.
   First explicit `Transformer(ModelArgs(...))` instantiation. Reads the native DeepSeek ModelArgs JSON at `MODEL_ARGS_CONFIG_PATH` (default `../DeepSeek-V3.2/inference/config_671B_v3.2.json`), instantiates via `ModelArgs(**native_config)`, and applies explicit runtime overrides: `dtype` (from `WEIGHTS_PRECISION`), `max_batch_size` (CLI), `max_seq_len` (CLI). Does not load weights, does not call `forward`, does not call `torch.distributed`. Reports parameter count, parameter dtype counts, and whether buffers exist. Construction may still allocate full-model parameter shapes regardless of reduced batch/seq overrides.

4. **Weight loading** — `scripts/native_verify.py --no-generate`.
   First real safetensor weight load into a constructed `Transformer`. Reads the rank-aware per-shard file `model{rank}-mp{world_size}.safetensors` under `SHARDED_CKPT_PATH` via `safetensors.torch.load_model` with `strict=False`. Missing/unexpected keys are reported, not silenced. Optional `DEQUANT_FP8_WEIGHTS=all` pre-dequantizes every FP8 Linear to BF16 in place once at load (TP2 token-exact baseline). Must run inside a 2-node Slurm allocation for TP2.

5. **Real token generation** — `scripts/native_verify.py` (no flag).
   First end-to-end greedy decode against a clean-lane reference case. Tokenizes the prompt (`add_special_tokens=False`), runs prefill + Lout single-token decode steps, argmax greedy, compares generated IDs to `expected_output_token_ids`. Rank 0 writes `results_clean/results/<TAG>/native_verify_results.json`. Targeted first config: TPCHECKREAL (prompt1, BS=1, Lin=10, Lout=15).

## Shared utilities

`src/clean_inference/` owns the helpers shared by the bring-up scripts:

- `config.py` — `parse_resolved_env`, `resolve_path`, `require_config_keys`. Parses the snapshot emitted by `scripts/parse_config.sh --format env`.
- `imports.py` — `setup_deepseek_imports`, `import_deepseek_modules`. Inserts `src/overrides/` before `<DEEPSEEK_REPO>/inference` on `sys.path` and validates module origins.
- `model_files.py` — `inspect_model_path`. Metadata-only model-directory inspection (no payload reads).
- `model_config.py` — `load_native_modelargs_config`, `modelargs_from_native_config`, `summarize_modelargs`. Reads the native DeepSeek ModelArgs JSON (path comes from `MODEL_ARGS_CONFIG_PATH`), validates every key against `ModelArgs` fields, and instantiates via `ModelArgs(**native_config)`. Applies explicit runtime overrides (`dtype`, `max_batch_size`, `max_seq_len`) after construction. Reports which fields came from the native config, which were overridden, and which fell back to dataclass defaults. No alias mapping; unknown keys fail loudly.

`scripts/inference_import_smoke.py`, `scripts/model_preflight.py`, and `scripts/model_construct_smoke.py` consume these helpers. `scripts/run_verify.py` shares the resolved-env parser.

Scripts add the clean root to `sys.path` so that `from src.clean_inference import ...` works when invoked from `scripts/`.

## Native ModelArgs JSON vs. HF `config.json`

DeepSeek's native `inference/model.py` consumes a JSON whose keys are exactly `ModelArgs` field names. The upstream pattern in `inference/generate.py` is:

```python
args = ModelArgs(**json.load(f))
model = Transformer(args)
```

The clean lane follows the same pattern. The native JSON lives at `MODEL_ARGS_CONFIG_PATH`, default `../DeepSeek-V3.2/inference/config_671B_v3.2.json` (note the dot before `2`, matching the upstream filename). For DeepSeek-V3.2 this declares `n_layers=61`, `dim=7168`, `n_heads=128`, `n_routed_experts=256`, `dtype="fp8"`, `scale_fmt="ue8m0"`, etc.

The HF-style `<ACTIVE_MODEL_PATH>/config.json` is **not** consumed by the native ModelArgs path. It is checkpoint/HF metadata only and may be used by future diagnostic tools for cross-checking, never as the source of truth for `ModelArgs`.

Three fields are intentionally *not* read from the native JSON and must come from runtime/experiment configuration:

- `dtype` — from `WEIGHTS_PRECISION` in the resolved env.
- `max_batch_size` — from CLI / experiment config.
- `max_seq_len` — from CLI / experiment config. **Not** auto-mapped from HF's `max_position_embeddings`; `max_seq_len` is a runtime KV/RoPE allocation limit, not a model capability ceiling.

`ModelArgs()` defaults remain a smaller demo configuration (`n_layers=27`, `dim=2048`, `n_heads=16`, `n_routed_experts=64`). The bring-up scripts must go through `modelargs_from_native_config` so bring-up never silently uses those defaults.

## TP2 distributed init ordering

For `SHARDING_MODE=tp2`, `torch.distributed.init_process_group("gloo")` runs **before** `Transformer(args)` is called. DeepSeek's `model.py` reads its module-global `world_size` inside `ColumnParallelLinear.__init__` and `RowParallelLinear.__init__` to decide per-rank shard shapes; the parallel layers will be the wrong size if init happens after construction. Per-rank TP shards (e.g. `model0-mp2.safetensors`, `model1-mp2.safetensors`) are produced by `../DeepSeek-V3.2/inference/convert.py` and contain tensors already sliced to the per-rank shape, so `safetensors.torch.load_model(model, shard_path, strict=False)` does a direct copy — no key remapping at load time.

For DP modes (`dp2`, `dp2_epon`) the init ordering is inverted: construction first (with the world_size=1 default so each rank builds the full replicated model), distributed init second. That branch is documented but not implemented in this first bring-up.

## Launcher

`submit_experiment.sh <TAG>` is the single user entry point. It validates the config, snapshots the resolved env, writes an sbatch whose body is:

```bash
srun --nodes=$SBATCH_NODES --ntasks=$SBATCH_NODES --ntasks-per-node=$SBATCH_TASKS_PER_NODE \
    bash scripts/run_native_distributed.sh <resolved_config>
```

then calls `sbatch` and records the job id. `REAL_RUN=1` is the baseline default; the placeholder dry-run path through `run_case.sh` is no longer reachable through the main launcher and remains only as earlier-bring-up debug code.

`scripts/run_native_distributed.sh` exports OMP env, computes `MASTER_ADDR`/`MASTER_PORT` from `SLURM_JOB_NODELIST`/`SLURM_JOB_ID`, and calls `torch.distributed.run` with one rank per node, which invokes `scripts/native_verify.py`.

The current real-run targets are three TPCHECKREAL variants:

| Tag | NATIVE_NO_LOAD_WEIGHTS | NATIVE_NO_GENERATE | Purpose |
|---|---|---|---|
| `TPCHECKREAL_NOLOAD` | 1 | 0 | construct only |
| `TPCHECKREAL_NOGEN` | 0 | 1 | weight load only |
| `TPCHECKREAL` | 0 | 0 | full token-exact decode (known-good) |

## What preflight does not cover

- Model construction.
- Weight materialization or sharding.
- KV cache allocation.
- `torch.distributed` rendezvous.
- Token generation.

Those are explicit future stages, each gated by its own small step.

Do not treat this document as complete yet.
