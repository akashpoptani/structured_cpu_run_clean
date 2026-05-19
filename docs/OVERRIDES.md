# Overrides

Purpose: Document CPU override modules and their boundaries.

The clean-owned override directory is `src/overrides/`.

Copied so far:

- `kernel.py`
- `fast_hadamard_transform.py`

The clean override path must be inserted before the DeepSeek inference path in `sys.path`.

Expected import order:

1. `src/overrides/`
2. `<DEEPSEEK_REPO>/inference`

`scripts/inference_import_smoke.py` validates that:

- `kernel` is loaded from `src/overrides/`.
- `fast_hadamard_transform` is loaded from `src/overrides/`.
- `model` is loaded from `<DEEPSEEK_REPO>/inference`.

The smoke test does not instantiate `Transformer`, load weights, call `torch.distributed`, or generate tokens.

Do not treat this document as complete yet.
