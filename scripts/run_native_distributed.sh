#!/usr/bin/env bash
# Launch native_verify.py via torch.distributed.run inside a Slurm allocation.
#
# Mirrors the legacy ../structured_cpu_run/verification/run_milestone_bench.sbatch
# launch shape:
#   srun --nodes=N --ntasks=N --ntasks-per-node=1   (provided by sbatch)
#     python -m torch.distributed.run
#         --nnodes=N --nproc-per-node=1
#         --node-rank=$SLURM_NODEID
#         --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT
#         scripts/native_verify.py --resolved-config ... --reference-group ... --case-id ...
#
# This script:
#   1. Sources the resolved env snapshot.
#   2. Exports OMP_NUM_THREADS / OMP_PROC_BIND / OMP_PLACES.
#   3. Computes MASTER_ADDR / MASTER_PORT from SLURM env (if present).
#   4. Runs torch.distributed.run for the current node.
#
# It does NOT call sbatch. Must be invoked inside a Slurm allocation (via
# `srun bash scripts/run_native_distributed.sh <resolved_config>`).

set -euo pipefail

usage() {
  echo "Usage: bash scripts/run_native_distributed.sh <resolved_config_path> [--no-load-weights | --no-generate]" >&2
  echo "" >&2
  echo "  Reference group/case default to env vars:" >&2
  echo "    NATIVE_REFERENCE_GROUP (default: prompt1_bs1_lin10_lout15)" >&2
  echo "    NATIVE_CASE_ID         (default: case_0001)" >&2
  echo "" >&2
  echo "  Must be invoked inside a Slurm allocation (srun) or with WORLD_SIZE etc." >&2
  echo "  preset in the environment when running single-rank locally." >&2
}

if [[ $# -lt 1 || -z "${1:-}" ]]; then
  usage
  exit 2
fi

RESOLVED_CONFIG_PATH="$1"; shift || true
EXTRA_FLAGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-load-weights|--no-generate)
      EXTRA_FLAGS+=("$1")
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

if [[ ! -f "$RESOLVED_CONFIG_PATH" ]]; then
  echo "ERROR: resolved config not found: $RESOLVED_CONFIG_PATH" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLEAN_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON:-$CLEAN_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "ERROR: no suitable python interpreter found (tried $PYTHON_BIN and python3)" >&2
    exit 1
  fi
fi

# shellcheck disable=SC1090
source "$RESOLVED_CONFIG_PATH"

REFERENCE_GROUP="${NATIVE_REFERENCE_GROUP:-prompt1_bs1_lin10_lout15}"
CASE_ID="${NATIVE_CASE_ID:-case_0001}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export SHARDING_MODE="${SHARDING_MODE:-tp2}"
export WEIGHTS_PRECISION="${WEIGHTS_PRECISION:-fp8}"
export DEQUANT_FP8_WEIGHTS="${DEQUANT_FP8_WEIGHTS:-all}"

echo "[run-native] hostname: $(hostname)"
echo "[run-native] resolved_config: $RESOLVED_CONFIG_PATH"
echo "[run-native] TAG=$TAG"
echo "[run-native] SHARDING_MODE=$SHARDING_MODE TP_SIZE=$TP_SIZE DP_SIZE=$DP_SIZE EP_SIZE=$EP_SIZE PP_SIZE=$PP_SIZE"
echo "[run-native] WEIGHTS_PRECISION=$WEIGHTS_PRECISION DEQUANT_FP8_WEIGHTS=$DEQUANT_FP8_WEIGHTS"
echo "[run-native] OMP_NUM_THREADS=$OMP_NUM_THREADS OMP_PROC_BIND=$OMP_PROC_BIND OMP_PLACES=$OMP_PLACES"
echo "[run-native] SLURM_JOB_ID=${SLURM_JOB_ID:-} SLURM_NODEID=${SLURM_NODEID:-} SLURM_PROCID=${SLURM_PROCID:-}"
echo "[run-native] SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "[run-native] reference: group=$REFERENCE_GROUP case=$CASE_ID"
echo "[run-native] python: $PYTHON_BIN"

NNODES="${SBATCH_NODES:-1}"
NPROC_PER_NODE=1

if [[ -n "${SLURM_JOB_NODELIST:-}" && -z "${MASTER_ADDR:-}" ]]; then
  if command -v scontrol >/dev/null 2>&1 && command -v getent >/dev/null 2>&1; then
    MASTER_NODE="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
    MASTER_ADDR="$(getent ahostsv4 "$MASTER_NODE" | head -n1 | awk '{print $1}')"
  fi
fi
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

if [[ -z "${MASTER_PORT:-}" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    export MASTER_PORT="$((29500 + SLURM_JOB_ID % 200))"
  else
    export MASTER_PORT="29500"
  fi
fi

NODE_RANK="${SLURM_NODEID:-0}"

echo "[run-native] launch: nnodes=$NNODES nproc_per_node=$NPROC_PER_NODE node_rank=$NODE_RANK master=$MASTER_ADDR:$MASTER_PORT"
echo "[run-native] extra flags: ${EXTRA_FLAGS[*]:-<none>}"

set -x
exec "$PYTHON_BIN" -m torch.distributed.run \
    --nnodes="$NNODES" \
    --nproc-per-node="$NPROC_PER_NODE" \
    --node-rank="$NODE_RANK" \
    --master-addr="$MASTER_ADDR" \
    --master-port="$MASTER_PORT" \
    "$CLEAN_ROOT/scripts/native_verify.py" \
    --resolved-config "$RESOLVED_CONFIG_PATH" \
    --reference-group "$REFERENCE_GROUP" \
    --case-id "$CASE_ID" \
    "${EXTRA_FLAGS[@]}"
