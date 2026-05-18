#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: bash scripts/run_case.sh <resolved_config_path>" >&2
}

if [[ $# -ne 1 || -z "${1:-}" ]]; then
  usage
  exit 2
fi

RESOLVED_CONFIG_PATH="$1"

if [[ ! -f "$RESOLVED_CONFIG_PATH" ]]; then
  echo "ERROR: resolved config file not found: $RESOLVED_CONFIG_PATH" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$RESOLVED_CONFIG_PATH"

cat <<EOF
run_case.sh placeholder
-----------------------
TAG: $TAG
EXPERIMENT_ID: $EXPERIMENT_ID
RUN_MODE: $RUN_MODE
RUN_LABEL: $RUN_LABEL
SHARDING_MODE: $SHARDING_MODE
DP_SIZE / TP_SIZE / EP_SIZE / PP_SIZE: $DP_SIZE / $TP_SIZE / $EP_SIZE / $PP_SIZE
LIN_TOKENS / LOUT_TOKENS / BATCH_SIZE: $LIN_TOKENS / $LOUT_TOKENS / $BATCH_SIZE
WEIGHTS_PRECISION: $WEIGHTS_PRECISION
KV_CACHE_DTYPE: $KV_CACHE_DTYPE
ACTIVE_MODEL_PATH: $ACTIVE_MODEL_PATH
SBATCH_NODES: $SBATCH_NODES
SBATCH_CPUS_PER_TASK: $SBATCH_CPUS_PER_TASK
OMP_NUM_THREADS: $OMP_NUM_THREADS
MEM_PROFILE: $MEM_PROFILE
TIME_PROFILE: $TIME_PROFILE
INFERENCE_ARCHITECTURE: $INFERENCE_ARCHITECTURE
SESSION_MODE: $SESSION_MODE

TODO: load modules
TODO: export OMP_NUM_THREADS / OMP_PROC_BIND / OMP_PLACES
TODO: choose run mode: verify | bench | both | generate
TODO: launch torch.distributed.run for direct_native mode
TODO: call clean verification/benchmark runner
EOF
