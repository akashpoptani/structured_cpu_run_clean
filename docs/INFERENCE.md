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

4. **Weight loading** — future.
   First real weight load from `ACTIVE_MODEL_PATH` (fp8 or bf16, derived from `WEIGHTS_PRECISION`).

5. **Real token generation** — future.
   First end-to-end greedy decode on a clean-lane reference case, comparing generated token IDs against `expected_output_token_ids`.

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

## What preflight does not cover

- Model construction.
- Weight materialization or sharding.
- KV cache allocation.
- `torch.distributed` rendezvous.
- Token generation.

Those are explicit future stages, each gated by its own small step.

Do not treat this document as complete yet.
