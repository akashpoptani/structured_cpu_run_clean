# Environment

The clean lane needs its own Python virtual environment so inference bring-up can stop relying on the old `structured_cpu_run` environment.

The old venv is only a known-good reference:

```text
/home/akashpt/DeepSeekRun/structured_cpu_run/without_vllm/.venv/
```

The clean venv requires Python >= 3.12.

The default login-node `python3` may be too old for the pinned clean-lane dependencies.

## Clean setup sequence

```bash
module load python/3.12.1
bash scripts/setup_venv.sh --reset
```

`--reset` removes any existing `.venv` and recreates it in one step.

Without `--reset`, the script refuses to overwrite an existing `.venv` and prints the manual cleanup option:

```bash
rm -rf .venv
bash scripts/setup_venv.sh
```

If Lmod is not available, pass the interpreter explicitly:

```bash
PYTHON_BIN=/path/to/python3.12 bash scripts/setup_venv.sh --reset
```

## Validation commands

```bash
.venv/bin/python -c "import torch; print('torch', torch.__version__)"
.venv/bin/python scripts/inference_import_smoke.py --resolved-config results_clean/resolved_configs/TPCHECK_resolved.env
.venv/bin/python scripts/model_preflight.py --resolved-config results_clean/resolved_configs/TPCHECK_resolved.env
PYTHON=.venv/bin/python bash scripts/run_case.sh results_clean/resolved_configs/TPCHECK_resolved.env
```

The smoke test imports DeepSeek through `src/overrides/`. It does not instantiate the model or load weights.

`model_preflight.py` performs the broader preflight (env versions, model directory metadata, ModelArgs / signatures, module globals). It also does not instantiate the model and does not load weights.

The clean venv should eventually replace use of the old venv.
