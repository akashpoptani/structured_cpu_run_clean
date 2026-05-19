#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: bash scripts/setup_venv.sh [--reset]" >&2
}

RESET=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --reset)
      RESET=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLEAN_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$CLEAN_ROOT/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "Selected Python:"
command -v "$PYTHON_BIN" || true
"$PYTHON_BIN" --version

if ! "$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 12):
    raise SystemExit(1)
PY
then
  echo "The selected Python is too old for the pinned clean-lane dependencies." >&2
  echo "Load a Python 3.12 module, e.g.:" >&2
  echo "  module load python/3.12.1" >&2
  echo "Or run:" >&2
  echo "  PYTHON_BIN=/path/to/python3.12 bash scripts/setup_venv.sh" >&2
  exit 1
fi

if [[ -d "$VENV_DIR" ]]; then
  if [[ "$RESET" -eq 1 ]]; then
    echo "Removing existing .venv at $VENV_DIR"
    rm -rf "$VENV_DIR"
  else
    echo "Existing .venv found. Remove it manually with:" >&2
    echo "  rm -rf .venv" >&2
    echo "Or rerun with:" >&2
    echo "  bash scripts/setup_venv.sh --reset" >&2
    exit 1
  fi
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r "$CLEAN_ROOT/requirements.txt"

echo "Clean venv Python:"
echo "$VENV_DIR/bin/python"
echo
echo "Validation commands:"
echo "$VENV_DIR/bin/python -c \"import torch; print(torch.__version__)\""
echo "$VENV_DIR/bin/python scripts/inference_import_smoke.py --resolved-config results_clean/resolved_configs/TPCHECK_resolved.env"
echo "$VENV_DIR/bin/python scripts/model_preflight.py --resolved-config results_clean/resolved_configs/TPCHECK_resolved.env"
