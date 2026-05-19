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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLEAN_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "ERROR: neither '$PYTHON_BIN' nor python3 was found" >&2
    exit 1
  fi
fi

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
EOF

echo
case "$RUN_MODE" in
  verify)
    echo "RUN_MODE=verify"
    "$PYTHON_BIN" "$CLEAN_ROOT/scripts/run_verify.py" \
      --resolved-config "$RESOLVED_CONFIG_PATH" \
      --mock-mode random \
      --format human
    echo "TODO: replace mock verification with actual clean CPU inference"
    echo "TODO: next inference bring-up command is scripts/inference_import_smoke.py"
    ;;
  bench)
    echo "RUN_MODE=bench"
    echo "TODO: run clean benchmark runner"
    ;;
  both)
    echo "RUN_MODE=both"
    echo "TODO: run small verification first"
    echo "TODO: then run benchmark"
    ;;
  generate)
    echo "RUN_MODE=generate"
    echo "TODO: run clean generation runner"
    ;;
  *)
    echo "ERROR: unknown RUN_MODE: $RUN_MODE" >&2
    exit 1
    ;;
esac

echo
case "$INFERENCE_ARCHITECTURE" in
  direct_native)
    echo "INFERENCE_ARCHITECTURE=direct_native"
    echo "TODO: next safe command is scripts/model_preflight.py"
    echo "TODO: use direct PyTorch model.forward / torch.distributed.run path"
    ;;
  server_client)
    echo "INFERENCE_ARCHITECTURE=server_client"
    echo "TODO: server/client mode is future-facing and not implemented yet"
    exit 1
    ;;
  *)
    echo "ERROR: unknown INFERENCE_ARCHITECTURE: $INFERENCE_ARCHITECTURE" >&2
    exit 1
    ;;
esac

echo
echo "TODO: load modules"
echo "TODO: export OMP_NUM_THREADS / OMP_PROC_BIND / OMP_PLACES"
