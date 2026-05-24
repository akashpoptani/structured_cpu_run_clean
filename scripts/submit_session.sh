#!/usr/bin/env bash
# One-line session launcher.
#   bash scripts/submit_session.sh <SESSION_TAG>
#
# Reads scripts/session_configs/<SESSION_TAG>.env, validates that every child
# tag listed in SESSION_CHILDREN parses cleanly, writes a per-child resolved
# env snapshot, generates ONE sbatch that calls scripts/run_session_distributed.sh,
# and submits it. The session sbatch loads the model once and iterates child
# modes sequentially via scripts/native_session.py.

set -euo pipefail

usage() {
  echo "Usage: bash scripts/submit_session.sh <SESSION_TAG>" >&2
  echo "Example: bash scripts/submit_session.sh TPSESSION" >&2
}

if [[ $# -ne 1 || -z "${1:-}" ]]; then
  usage
  exit 2
fi

SESSION_TAG_ARG="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLEAN_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SESSION_ENV="$CLEAN_ROOT/scripts/session_configs/${SESSION_TAG_ARG}.env"

if [[ ! -f "$SESSION_ENV" ]]; then
  echo "ERROR: session config not found: $SESSION_ENV" >&2
  exit 1
fi

# Load session config (defines SESSION_CHILDREN + sbatch fields + cache config).
# shellcheck disable=SC1090
source "$SESSION_ENV"

echo "submit_session.sh: SESSION_TAG=$SESSION_TAG  CHILDREN=$SESSION_CHILDREN"
echo "  partition=$SBATCH_PARTITION account=$SBATCH_ACCOUNT nodes=$SBATCH_NODES cpus=$SBATCH_CPUS_PER_TASK mem=$SBATCH_MEM time=$SBATCH_TIME"
echo "  DEQUANT_CACHE_MODE=$DEQUANT_CACHE_MODE DEQUANT_CACHE_PATH=$DEQUANT_CACHE_PATH"

# Resolve and snapshot each child config, checking compatibility invariants.
RESOLVED_CONFIG_DIR="$CLEAN_ROOT/results_clean/resolved_configs"
SBATCH_DIR="$CLEAN_ROOT/tmp/sbatch"
LOG_DIR="$CLEAN_ROOT/results_clean/logs"
SESSION_RUN_DIR="$CLEAN_ROOT/results_clean/runs/$SESSION_TAG"
mkdir -p "$RESOLVED_CONFIG_DIR" "$SBATCH_DIR" "$LOG_DIR" "$SESSION_RUN_DIR"

# Track the per-child resolved-config paths (one per child) so the session
# runner can read them in order.
CHILD_PATHS=()
declare -A first_seen
first_pivot_value=""
for child_tag in $SESSION_CHILDREN; do
  echo "--- session child: $child_tag ---"
  child_env=$(bash "$CLEAN_ROOT/scripts/parse_config.sh" --format env "$child_tag")
  child_resolved_path="$RESOLVED_CONFIG_DIR/${child_tag}_resolved.env"
  printf '%s\n' "$child_env" > "$child_resolved_path"

  # Quick invariant check across children. Pull a few fields with shell eval.
  (
    eval "$child_env"
    PIVOT="${SHARDING_MODE}|${TP_SIZE}|${DP_SIZE}|${EP_SIZE}|${PP_SIZE}|${WEIGHTS_PRECISION}|${SHARDED_CKPT_PATH}|${MODEL_ARGS_CONFIG_PATH}|${DEQUANT_FP8_WEIGHTS}"
    printf '%s' "$PIVOT"
  ) > "$SESSION_RUN_DIR/${child_tag}.pivot"
  pivot_value=$(cat "$SESSION_RUN_DIR/${child_tag}.pivot")
  if [[ -z "$first_pivot_value" ]]; then
    first_pivot_value="$pivot_value"
    first_pivot_child="$child_tag"
  elif [[ "$pivot_value" != "$first_pivot_value" ]]; then
    echo "ERROR: session child '$child_tag' has incompatible config vs first child '$first_pivot_child':" >&2
    echo "  $first_pivot_child: $first_pivot_value" >&2
    echo "  $child_tag: $pivot_value" >&2
    exit 1
  fi

  CHILD_PATHS+=("$child_resolved_path")
done

CHILDREN_LIST="$(IFS=,; echo "${CHILD_PATHS[*]}")"

SBATCH_BASENAME="${SESSION_TAG}_session_n${SBATCH_NODES}_c${SBATCH_CPUS_PER_TASK}_mem${SBATCH_MEM}.sbatch"
SBATCH_PATH="$SBATCH_DIR/$SBATCH_BASENAME"

cat > "$SBATCH_PATH" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=$SESSION_TAG
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

export SESSION_TAG=$SESSION_TAG
export SESSION_CHILDREN_PATHS="$CHILDREN_LIST"
export DEQUANT_CACHE_MODE=$DEQUANT_CACHE_MODE
export DEQUANT_CACHE_PATH=$DEQUANT_CACHE_PATH
export OMP_NUM_THREADS=$OMP_NUM_THREADS
export OMP_PROC_BIND=$OMP_PROC_BIND
export OMP_PLACES=$OMP_PLACES

echo "[submit-session] launching session via srun"
srun --nodes=$SBATCH_NODES --ntasks=$SBATCH_NODES --ntasks-per-node=$SBATCH_TASKS_PER_NODE \\
    bash "$CLEAN_ROOT/scripts/run_session_distributed.sh"
EOF

printf '%s\n' "$SBATCH_PATH" > "$CLEAN_ROOT/.last_sbatch"

set +e
SBATCH_OUTPUT="$(sbatch "$SBATCH_PATH" 2>&1)"
SBATCH_RC=$?
set -e
echo "$SBATCH_OUTPUT"
if [[ $SBATCH_RC -ne 0 ]]; then
  echo "ERROR: sbatch failed (rc=$SBATCH_RC)" >&2
  exit "$SBATCH_RC"
fi

JOB_ID="$(awk '/^Submitted batch job / {print $NF}' <<<"$SBATCH_OUTPUT" | tail -n1)"
if [[ -z "${JOB_ID:-}" ]]; then
  echo "ERROR: could not parse job id from sbatch output." >&2
  exit 1
fi
printf '%s\n' "$JOB_ID" > "$CLEAN_ROOT/.last_job"

JOB_RUN_DIR="$SESSION_RUN_DIR/$JOB_ID"
mkdir -p "$JOB_RUN_DIR"
SUBMIT_TIME="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
cat > "$JOB_RUN_DIR/session_metadata.env" <<EOF
SESSION_TAG=$SESSION_TAG
JOB_ID=$JOB_ID
SBATCH_PATH=$SBATCH_PATH
SUBMIT_TIME=$SUBMIT_TIME
SESSION_CHILDREN=$SESSION_CHILDREN
SESSION_CHILDREN_PATHS=$CHILDREN_LIST
SBATCH_PARTITION=$SBATCH_PARTITION
SBATCH_ACCOUNT=$SBATCH_ACCOUNT
SBATCH_NODES=$SBATCH_NODES
SBATCH_CPUS_PER_TASK=$SBATCH_CPUS_PER_TASK
SBATCH_MEM=$SBATCH_MEM
SBATCH_TIME=$SBATCH_TIME
DEQUANT_CACHE_MODE=$DEQUANT_CACHE_MODE
DEQUANT_CACHE_PATH=$DEQUANT_CACHE_PATH
EOF

cat <<EOF

Submitted session batch job $JOB_ID
Session tag: $SESSION_TAG
Children: $SESSION_CHILDREN
Sbatch: $SBATCH_PATH
Session metadata: $JOB_RUN_DIR/session_metadata.env
Logs:
  $LOG_DIR/${SESSION_TAG}_${JOB_ID}.out
  $LOG_DIR/${SESSION_TAG}_${JOB_ID}.err
Result JSON(s) (one per child + session summary):
  results_clean/results/<child>/native_<mode>_results.json
  results_clean/results/$SESSION_TAG/session_results.json
EOF
