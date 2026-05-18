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

Do not treat this document as complete yet.
