# Commands

Current local commands:

```bash
bash scripts/submit_experiment.sh TPCHECK
```

submit_experiment.sh calls parse_config.sh file:
```bash
bash scripts/parse_config.sh TPCHECK
```

`parse_config.sh` parses, validates, and prints the resolved config.

`submit_experiment.sh` is currently a non-submitting skeleton entrypoint. It calls `parse_config.sh` but does not generate or submit jobs yet.

Future intended command style:

```bash
bash scripts/submit_experiment.sh <config_tag>
bash scripts/check_run.sh <job_id> "LATER"
bash scripts/append_to_results.sh <job_id> <config_tag> "LATER"
```
