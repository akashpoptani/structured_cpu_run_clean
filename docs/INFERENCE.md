# Native Inference Bring-up

Purpose: Track the ordered stages of native CPU inference bring-up in the clean lane.

The clean lane brings up native CPU DeepSeek-V3.2 inference in small, reviewable stages. Each stage must pass before the next is attempted.

## Stages

1. **Clean override import smoke** — `scripts/inference_import_smoke.py`.
   Validates `sys.path` ordering: `kernel` and `fast_hadamard_transform` resolve to `src/overrides/`, and `model` resolves to `<DEEPSEEK_REPO>/inference`. Does not instantiate `Transformer`, does not load weights, does not call `torch.distributed`.

2. **Model preflight** — `scripts/model_preflight.py`.
   Inspects the environment (Python, torch, transformers, safetensors versions), resolved paths, imported override module files, model-directory metadata (`config.json`, tokenizer files, safetensors count and total GiB, index files), DeepSeek model symbols (`Transformer`, `ModelArgs`, `Block`/`TransformerBlock`, `MLA`, `MoE`), `ModelArgs` field defaults, `Transformer.__init__` / `Transformer.forward` / `MLA.forward` / `MoE.forward` signatures, and module globals (`world_size`, `rank`, `local_rank`, `block_size`, `gemm_impl`, `attn_impl`).
   It does **not** instantiate `Transformer`, does **not** load weights, does **not** call `torch.distributed`, and does **not** run generation. Heavy safetensor payloads are never read — only file sizes and names.

3. **Model construction smoke** — future.
   First explicit `Transformer(ModelArgs(...))` instantiation with controlled, minimal args and no weight loading. Aimed at catching parameter allocation, KV cache shape, and override compatibility issues separately from weight I/O.

4. **Weight loading** — future.
   First real weight load from `ACTIVE_MODEL_PATH` (fp8 or bf16, derived from `WEIGHTS_PRECISION`).

5. **Real token generation** — future.
   First end-to-end greedy decode on a clean-lane reference case, comparing generated token IDs against `expected_output_token_ids`.

## Shared utilities

`src/clean_inference/` owns the helpers shared by the bring-up scripts:

- `config.py` — `parse_resolved_env`, `resolve_path`, `require_config_keys`. Parses the snapshot emitted by `scripts/parse_config.sh --format env`.
- `imports.py` — `setup_deepseek_imports`, `import_deepseek_modules`. Inserts `src/overrides/` before `<DEEPSEEK_REPO>/inference` on `sys.path` and validates module origins.
- `model_files.py` — `inspect_model_path`. Metadata-only model-directory inspection (no payload reads).

Both `scripts/inference_import_smoke.py` and `scripts/model_preflight.py` consume these helpers. `scripts/run_verify.py` shares the resolved-env parser.

Scripts add the clean root to `sys.path` so that `from src.clean_inference import ...` works when invoked from `scripts/`.

## What preflight does not cover

- Model construction.
- Weight materialization or sharding.
- KV cache allocation.
- `torch.distributed` rendezvous.
- Token generation.

Those are explicit future stages, each gated by its own small step.

Do not treat this document as complete yet.
