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

RESOLVED_ENV="$(bash "$CLEAN_ROOT/scripts/parse_config.sh" --format env "$TAG")"

# parse_config.sh emits shell-safe KEY=VALUE assignments for a fixed key set.
eval "$RESOLVED_ENV"

SBATCH_DIR="$CLEAN_ROOT/tmp/sbatch"
RESOLVED_CONFIG_DIR="$CLEAN_ROOT/results_clean/resolved_configs"
mkdir -p "$SBATCH_DIR" "$RESOLVED_CONFIG_DIR"

RESOLVED_CONFIG_PATH="$RESOLVED_CONFIG_DIR/${TAG}_resolved.env"
printf '%s\n' "$RESOLVED_ENV" > "$RESOLVED_CONFIG_PATH"

SBATCH_BASENAME="${TAG}_${RUN_MODE}_${SHARDING_MODE}_lin${LIN_TOKENS}_lout${LOUT_TOKENS}_bs${BATCH_SIZE}_n${SBATCH_NODES}_c${SBATCH_CPUS_PER_TASK}_mem${SBATCH_MEM}_tprof${TIME_PROFILE}_mprof${MEM_PROFILE}.sbatch"
SBATCH_PATH="$SBATCH_DIR/$SBATCH_BASENAME"

cat > "$SBATCH_PATH" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=$RUN_LABEL
#SBATCH --nodes=$SBATCH_NODES
#SBATCH --ntasks-per-node=$SBATCH_TASKS_PER_NODE
#SBATCH --cpus-per-task=$SBATCH_CPUS_PER_TASK
#SBATCH --mem=$SBATCH_MEM
#SBATCH --time=$SBATCH_TIME
#SBATCH --partition=$SBATCH_PARTITION
#SBATCH --output=$OUTPUT_ROOT/logs/%x_%j.out
#SBATCH --error=$OUTPUT_ROOT/logs/%x_%j.err

set -euo pipefail

echo "Dry-run generated sbatch. Calling run_case.sh placeholder."
bash "$CLEAN_ROOT/scripts/run_case.sh" "$RESOLVED_CONFIG_PATH"
EOF

echo
echo "Generated dry-run sbatch:"
echo "$SBATCH_PATH"
echo
echo "Resolved config snapshot:"
echo "$RESOLVED_CONFIG_PATH"
echo
echo "To inspect:"
echo "sed -n '1,160p' $SBATCH_PATH"
echo
echo "Not submitted."
