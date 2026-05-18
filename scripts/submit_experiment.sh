#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: bash scripts/submit_experiment.sh <TAG>" >&2
  echo "Example: bash scripts/submit_experiment.sh TPCHECK" >&2
}

if [[ $# -ne 1 || -z "${1:-}" ]]; then
  usage
  exit 2
fi

TAG="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLEAN_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "submit_experiment.sh: non-submitting skeleton"
echo "TAG: $TAG"
echo

bash "$CLEAN_ROOT/scripts/parse_config.sh" "$TAG"

echo
echo "Resolved env preview:"
bash "$CLEAN_ROOT/scripts/parse_config.sh" --format env "$TAG"

echo
echo "TODO future step: generate sbatch script from parsed config."
echo "TODO future step: submit sbatch."
echo "TODO future step: write .last_job."
echo "TODO future step: write resolved config snapshot into results_clean."
