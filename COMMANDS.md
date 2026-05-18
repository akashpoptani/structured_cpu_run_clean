# Commands

Current local commands:

```bash
bash scripts/submit_experiment.sh TPCHECK
```

submit_experiment.sh calls parse_config.sh file:
```bash
bash scripts/parse_config.sh TPCHECK
bash scripts/parse_config.sh --format env TPCHECK
```

Placeholder case runner:
```bash
bash scripts/run_case.sh results_clean/resolved_configs/TPCHECK_resolved.env
```

Verification reference inspector:
```bash
python scripts/inspect_reference_cases.py --reference-root verification/references/
python scripts/inspect_reference_cases.py --reference-root verification/references/ --format json
```

Default `parse_config.sh` output is human-readable.

`parse_config.sh --format env` output is machine-readable and future scripts can source it.

`parse_config.sh` parses, validates, and prints the resolved config.

`submit_experiment.sh` parses config, writes a resolved config snapshot, and generates a dry-run sbatch under `tmp/sbatch`.

It does not submit jobs. The generated sbatch is placeholder only and does not run inference yet.

Generated sbatches now delegate to `scripts/run_case.sh`. `run_case.sh` is currently a placeholder and does not run inference yet.

`run_case.sh` has placeholder dispatch for `verify`, `bench`, `both`, and `generate`.

Only `direct_native` is allowed as a placeholder path right now. `server_client` is schema-visible but not implemented.

Verification reference data exists under `verification/references/`.

Initial reference group: `verification/references/prompt1_bs1_lin10_lout15/` for 1 prompt, batch size 1, Lin 10, and Lout 15.

No clean verification command exists yet.

`inspect_reference_cases.py` inspects and validates reference-case JSON files. It does not run model inference.

`run_case.sh` verify now calls this inspector.

Generated artifact examples:

- `examples/generated/TPCHECK_resolved.env`
- `examples/generated/TPCHECK_verify_tp2_lin10_lout15_bs1_n2_c32_mem600G_tprof0_mprof0.sbatch`

Live generated files go to `tmp/sbatch/` and `results_clean/resolved_configs/` and are gitignored.

Future intended command style:

```bash
bash scripts/submit_experiment.sh <config_tag>
bash scripts/check_run.sh <job_id> "LATER"
bash scripts/append_to_results.sh <job_id> <config_tag> "LATER"
```
