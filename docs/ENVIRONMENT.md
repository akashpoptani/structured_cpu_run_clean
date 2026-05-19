# Environment

The clean lane needs its own Python virtual environment so inference bring-up can stop relying on the old `structured_cpu_run` environment.

The old venv is only a known-good reference:

```text
/home/akashpt/DeepSeekRun/structured_cpu_run/without_vllm/.venv/
```

The clean venv requires Python >= 3.12.

The default login-node `python3` may be too old for the pinned clean-lane dependencies.

Use Lmod to load Python 3.12 if available:

```bash
module load python/3.12.1
bash scripts/setup_venv.sh
```

Or explicitly pass the interpreter:

```bash
PYTHON_BIN=/path/to/python3.12 bash scripts/setup_venv.sh
```

If setup failed partway, remove the partial `.venv` before retrying:

```bash
rm -rf .venv
```

Validate the clean venv:

```bash
.venv/bin/python scripts/inference_import_smoke.py --resolved-config results_clean/resolved_configs/TPCHECK_resolved.env
```

The current smoke test imports DeepSeek through `src/overrides/`. It does not instantiate the model or load weights.

The clean venv should eventually replace use of the old venv.
