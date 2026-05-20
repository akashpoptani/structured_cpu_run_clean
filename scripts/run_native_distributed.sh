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
  echo "Usage: bash scripts/run_native_distributed.sh <resolved_config_path>" >&2
  echo "" >&2
  echo "  Reference group defaults to env var NATIVE_REFERENCE_GROUP" >&2
  echo "  (default: prompt1_bs1_lin10_lout15). All cases in the group are" >&2
  echo "  run unless NATIVE_CASE_ID is set." >&2
  echo "" >&2
  echo "  The flags NATIVE_NO_LOAD_WEIGHTS and NATIVE_NO_GENERATE come from" >&2
  echo "  the resolved config and translate into --no-load-weights and" >&2
  echo "  --no-generate respectively." >&2
  echo "" >&2
  echo "  Must be invoked inside a Slurm allocation (srun) or with WORLD_SIZE etc." >&2
  echo "  preset in the environment when running single-rank locally." >&2
}

if [[ $# -ne 1 || -z "${1:-}" ]]; then
  usage
  exit 2
fi

RESOLVED_CONFIG_PATH="$1"

if [[ ! -f "$RESOLVED_CONFIG_PATH" ]]; then
  echo "ERROR: resolved config not found: $RESOLVED_CONFIG_PATH" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLEAN_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Python selection (no silent fall-through to system python3 for real runs):
#   1. if $PYTHON is set, use it directly.
#   2. else if $CLEAN_ROOT/.venv/bin/activate exists, source it and use venv python.
#   3. else if $CLEAN_ROOT/.venv/bin/python exists, use it directly.
#   4. else fail with explicit setup instructions.
if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif [[ -f "$CLEAN_ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$CLEAN_ROOT/.venv/bin/activate"
  PYTHON_BIN="$CLEAN_ROOT/.venv/bin/python"
elif [[ -x "$CLEAN_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$CLEAN_ROOT/.venv/bin/python"
else
  echo "ERROR: no clean venv found at $CLEAN_ROOT/.venv" >&2
  echo "Build it first:" >&2
  echo "  module load python/3.12.1" >&2
  echo "  bash scripts/setup_venv.sh --reset" >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: selected python is not executable: $PYTHON_BIN" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$RESOLVED_CONFIG_PATH"

REFERENCE_GROUP="${NATIVE_REFERENCE_GROUP:-prompt1_bs1_lin10_lout15}"
CASE_ID="${NATIVE_CASE_ID:-}"

# Translate resolved-config flags into native_verify CLI flags.
EXTRA_FLAGS=()
if [[ "${NATIVE_NO_LOAD_WEIGHTS:-0}" == "1" && "${NATIVE_NO_GENERATE:-0}" == "1" ]]; then
  echo "ERROR: NATIVE_NO_LOAD_WEIGHTS=1 already implies no generation; both flags set is ambiguous." >&2
  exit 2
elif [[ "${NATIVE_NO_LOAD_WEIGHTS:-0}" == "1" ]]; then
  EXTRA_FLAGS+=(--no-load-weights)
elif [[ "${NATIVE_NO_GENERATE:-0}" == "1" ]]; then
  EXTRA_FLAGS+=(--no-generate)
fi

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
echo "[run-native] reference: group=$REFERENCE_GROUP case=${CASE_ID:-<all>}"
echo "[run-native] python: $PYTHON_BIN"
"$PYTHON_BIN" --version 2>&1 | sed 's/^/[run-native] python version: /'

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
echo "[run-native] NATIVE_NO_LOAD_WEIGHTS=${NATIVE_NO_LOAD_WEIGHTS:-0} NATIVE_NO_GENERATE=${NATIVE_NO_GENERATE:-0}"
echo "[run-native] extra flags: ${EXTRA_FLAGS[*]:-<none>}"

CASE_FLAG=()
if [[ -n "$CASE_ID" ]]; then
  CASE_FLAG=(--case-id "$CASE_ID")
fi

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
    "${CASE_FLAG[@]}" \
    "${EXTRA_FLAGS[@]}"
