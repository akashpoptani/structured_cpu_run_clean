#!/usr/bin/env bash
# One-line experiment launcher for the clean native CPU DeepSeek lane.
#
# Usage: bash scripts/submit_experiment.sh <TAG>
#
# Pipeline:
#   1. Run scripts/parse_config.sh to validate the resolved config.
#   2. Snapshot the resolved env to results_clean/resolved_configs/<TAG>_resolved.env.
#   3. Write an sbatch under tmp/sbatch/<TAG>_...sbatch whose body is
#      `srun ... bash scripts/run_native_distributed.sh <resolved_config>`.
#   4. sbatch it and capture the job id.
#   5. Write run metadata under results_clean/runs/<TAG>/<JOB_ID>/run_metadata.env
#      and update .last_job / .last_sbatch / .last_resolved_config.
#
# This script is the single entry point for real native runs. It always
# generates a real-distributed sbatch (no placeholder/mock branch).

set -euo pipefail

usage() {
  echo "Usage: bash scripts/submit_experiment.sh <TAG>" >&2
  echo "Example: bash scripts/submit_experiment.sh TPCHECKREAL" >&2
}

if [[ $# -ne 1 || -z "${1:-}" ]]; then
  usage
  exit 2
fi

TAG="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLEAN_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "submit_experiment.sh: TAG=$TAG"

# Human-readable parsed config for transparency in the user's terminal.
bash "$CLEAN_ROOT/scripts/parse_config.sh" "$TAG"

# Machine-readable env for the sbatch body to source via the snapshot.
RESOLVED_ENV="$(bash "$CLEAN_ROOT/scripts/parse_config.sh" --format env "$TAG")"
# Fixed key set; safe to eval.
eval "$RESOLVED_ENV"

SBATCH_DIR="$CLEAN_ROOT/tmp/sbatch"
RESOLVED_CONFIG_DIR="$CLEAN_ROOT/results_clean/resolved_configs"
LOG_DIR="$CLEAN_ROOT/results_clean/logs"
mkdir -p "$SBATCH_DIR" "$RESOLVED_CONFIG_DIR" "$LOG_DIR"

RESOLVED_CONFIG_PATH="$RESOLVED_CONFIG_DIR/${TAG}_resolved.env"
printf '%s\n' "$RESOLVED_ENV" > "$RESOLVED_CONFIG_PATH"

SBATCH_BASENAME="${TAG}_${RUN_MODE}_${SHARDING_MODE}_lin${LIN_TOKENS}_lout${LOUT_TOKENS}_bs${BATCH_SIZE}_n${SBATCH_NODES}_c${SBATCH_CPUS_PER_TASK}_mem${SBATCH_MEM}_tprof${TIME_PROFILE}_mprof${MEM_PROFILE}.sbatch"
SBATCH_PATH="$SBATCH_DIR/$SBATCH_BASENAME"

cat > "$SBATCH_PATH" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=$RUN_LABEL
#SBATCH --nodes=$SBATCH_NODES
#SBATCH --ntasks=$SBATCH_NODES
#SBATCH --ntasks-per-node=$SBATCH_TASKS_PER_NODE
#SBATCH --cpus-per-task=$SBATCH_CPUS_PER_TASK
#SBATCH --mem=$SBATCH_MEM
#SBATCH --time=$SBATCH_TIME
#SBATCH --partition=$SBATCH_PARTITION
#SBATCH --account=$SBATCH_ACCOUNT
#SBATCH --output=$LOG_DIR/%x_%j.out
#SBATCH --error=$LOG_DIR/%x_%j.err

set -euo pipefail

echo "[submit] launching native distributed verification via srun"
srun --nodes=$SBATCH_NODES --ntasks=$SBATCH_NODES --ntasks-per-node=$SBATCH_TASKS_PER_NODE \\
    bash "$CLEAN_ROOT/scripts/run_native_distributed.sh" "$RESOLVED_CONFIG_PATH"
EOF

# Latest-pointers (always update; consumers read these without scanning).
printf '%s\n' "$RESOLVED_CONFIG_PATH" > "$CLEAN_ROOT/.last_resolved_config"
printf '%s\n' "$SBATCH_PATH" > "$CLEAN_ROOT/.last_sbatch"

# Submit.
set +e
SBATCH_OUTPUT="$(sbatch "$SBATCH_PATH" 2>&1)"
SBATCH_RC=$?
set -e
echo "$SBATCH_OUTPUT"
if [[ $SBATCH_RC -ne 0 ]]; then
  echo "ERROR: sbatch failed (rc=$SBATCH_RC). See output above." >&2
  exit "$SBATCH_RC"
fi

JOB_ID="$(awk '/^Submitted batch job / {print $NF}' <<<"$SBATCH_OUTPUT" | tail -n1)"
if [[ -z "${JOB_ID:-}" ]]; then
  echo "ERROR: could not parse job id from sbatch output." >&2
  exit 1
fi
printf '%s\n' "$JOB_ID" > "$CLEAN_ROOT/.last_job"

# Run metadata snapshot per (TAG, JOB_ID).
RUN_DIR="$CLEAN_ROOT/results_clean/runs/$TAG/$JOB_ID"
mkdir -p "$RUN_DIR"
RUN_METADATA="$RUN_DIR/run_metadata.env"
SUBMIT_TIME="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
cat > "$RUN_METADATA" <<EOF
TAG=$TAG
JOB_ID=$JOB_ID
SBATCH_PATH=$SBATCH_PATH
RESOLVED_CONFIG_PATH=$RESOLVED_CONFIG_PATH
SUBMIT_TIME=$SUBMIT_TIME
RUN_LABEL=$RUN_LABEL
SBATCH_PARTITION=$SBATCH_PARTITION
SBATCH_ACCOUNT=$SBATCH_ACCOUNT
SBATCH_NODES=$SBATCH_NODES
SBATCH_CPUS_PER_TASK=$SBATCH_CPUS_PER_TASK
SBATCH_MEM=$SBATCH_MEM
NATIVE_NO_LOAD_WEIGHTS=$NATIVE_NO_LOAD_WEIGHTS
NATIVE_NO_GENERATE=$NATIVE_NO_GENERATE
DEQUANT_FP8_WEIGHTS=$DEQUANT_FP8_WEIGHTS
EOF

case "$RUN_MODE" in
  verify)   RESULT_LINES="  $CLEAN_ROOT/$OUTPUT_ROOT/results/$TAG/native_verify_results.json" ;;
  generate) RESULT_LINES="  $CLEAN_ROOT/$OUTPUT_ROOT/results/$TAG/native_generate_results.json" ;;
  bench)    RESULT_LINES="  $CLEAN_ROOT/$OUTPUT_ROOT/results/$TAG/native_bench_results.json" ;;
  both)     RESULT_LINES="  $CLEAN_ROOT/$OUTPUT_ROOT/results/$TAG/native_verify_results.json
  $CLEAN_ROOT/$OUTPUT_ROOT/results/$TAG/native_bench_results.json
  $CLEAN_ROOT/$OUTPUT_ROOT/results/$TAG/native_both_results.json" ;;
  *)        RESULT_LINES="  $CLEAN_ROOT/$OUTPUT_ROOT/results/$TAG/native_${RUN_MODE}_results.json" ;;
esac

cat <<EOF

Submitted batch job $JOB_ID
Resolved config: $RESOLVED_CONFIG_PATH
Sbatch: $SBATCH_PATH
Run metadata: $RUN_METADATA
RUN_MODE: $RUN_MODE
Logs:
  $LOG_DIR/${RUN_LABEL}_${JOB_ID}.out
  $LOG_DIR/${RUN_LABEL}_${JOB_ID}.err
Result JSON(s):
$RESULT_LINES
EOF
