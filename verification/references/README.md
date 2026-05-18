# Verification References

`GPU_REFERENCE_PATH` points to this directory.

This directory will contain one or more verification case subdirectories.

Each case contains input prompt metadata and expected GPU output tokens.

The clean verifier is not implemented yet.

This data is for the future clean verification runner.

The current first reference group is `prompt1_bs1_lin10_lout15`.

That group means 1 prompt, batch size 1, Lin 10, and Lout 15.

The current first case is `prompt1_bs1_lin10_lout15/case_0001.json`.

Future reference groups are expected for batch-size-4 and long-context cases.
