# Verification

Purpose: Document how clean-lane correctness checks should work.

The clean verifier is not implemented yet.

Reference data lives under `verification/references/`.

The initial reference group is `prompt1_bs1_lin10_lout15`, which means 1 prompt, batch size 1, Lin 10, and Lout 15.

It has one initial case:

- `verification/references/prompt1_bs1_lin10_lout15/case_0001.json`

The current comparison target is generated token IDs.

Future verification goals:

- At least 4 prompts.
- Batch-size-4 verification.
- Logit comparisons.
- Layer output comparisons.
- Attention output comparisons.
- MoE output comparisons.

The current reference data is for a future clean verification runner and is not consumed by any runner yet.

Reference inspector command:

```bash
python scripts/inspect_reference_cases.py --reference-root verification/references/
```

Machine-readable inspector output:

```bash
python scripts/inspect_reference_cases.py --reference-root verification/references/ --format json
```

Required case JSON fields:

- `case_id`
- `tag`
- `description`
- `prompt_text`
- `lin_tokens`
- `lout_tokens`
- `batch_size`
- `sampling`
- `expected_output_token_ids`
- `source`

Required `sampling` fields:

- `method`
- `temperature`
- `min_tokens`
- `max_tokens`
- `ignore_eos`

The inspector currently validates schema, basic field types, positive token and batch counts, expected output token count for batch size 1, and min/max token consistency.

It does not validate model output, logits, layer outputs, attention outputs, or MoE outputs yet.

## Clean verification runner, mock mode

`scripts/run_verify.py` is the first end-to-end verification runner.

It reads the resolved config, reads `GPU_REFERENCE_PATH`, loads reference cases, generates mock output tokens, and compares generated token IDs to expected GPU token IDs.

It writes results to `results_clean/results/<TAG>/verify_results.json`.

Random mode is expected to fail because it generates deterministic random token IDs.

Golden mode is expected to pass because it uses the expected tokens as generated tokens. Golden mode only tests plumbing.

Actual model inference is not implemented yet.

Current commands:

```bash
python3 scripts/run_verify.py --resolved-config results_clean/resolved_configs/TPCHECK_resolved.env
python3 scripts/run_verify.py --resolved-config results_clean/resolved_configs/TPCHECK_resolved.env --mock-mode golden
python3 scripts/run_verify.py --resolved-config results_clean/resolved_configs/TPCHECK_resolved.env --format json
```

Do not treat this document as complete yet.
