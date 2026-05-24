#!/usr/bin/env bash
# Per-node launcher for a session. Invoked by the session sbatch via srun.
# Reads SESSION_TAG / SESSION_CHILDREN_PATHS / DEQUANT_CACHE_{MODE,PATH} from
# the environment, finds the clean venv, computes MASTER_ADDR/MASTER_PORT,
# and runs torch.distributed.run scripts/native_session.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLEAN_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

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
  echo "  module load python/3.12.1 && bash scripts/setup_venv.sh --reset" >&2
  exit 1
fi

: "${SESSION_TAG:?SESSION_TAG must be set by the session sbatch}"
: "${SESSION_CHILDREN_PATHS:?SESSION_CHILDREN_PATHS must be set by the session sbatch}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export DEQUANT_CACHE_MODE="${DEQUANT_CACHE_MODE:-off}"
export DEQUANT_CACHE_PATH="${DEQUANT_CACHE_PATH:-}"

echo "[run-session] hostname: $(hostname)"
echo "[run-session] SESSION_TAG=$SESSION_TAG"
echo "[run-session] SESSION_CHILDREN_PATHS=$SESSION_CHILDREN_PATHS"
echo "[run-session] DEQUANT_CACHE_MODE=$DEQUANT_CACHE_MODE DEQUANT_CACHE_PATH=$DEQUANT_CACHE_PATH"
echo "[run-session] OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "[run-session] SLURM_JOB_ID=${SLURM_JOB_ID:-} SLURM_NODEID=${SLURM_NODEID:-}"
echo "[run-session] python: $PYTHON_BIN"
"$PYTHON_BIN" --version 2>&1 | sed 's/^/[run-session] python version: /'

NNODES="${SBATCH_NODES:-${SLURM_NNODES:-1}}"

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

echo "[run-session] launch: nnodes=$NNODES node_rank=$NODE_RANK master=$MASTER_ADDR:$MASTER_PORT"

# Children paths are comma-separated; pass as repeated --child arguments.
IFS=',' read -r -a CHILD_PATHS <<<"$SESSION_CHILDREN_PATHS"
CHILD_ARGS=()
for p in "${CHILD_PATHS[@]}"; do
  CHILD_ARGS+=(--child "$p")
done

set -x
exec "$PYTHON_BIN" -m torch.distributed.run \
    --nnodes="$NNODES" \
    --nproc-per-node=1 \
    --node-rank="$NODE_RANK" \
    --master-addr="$MASTER_ADDR" \
    --master-port="$MASTER_PORT" \
    "$CLEAN_ROOT/scripts/native_session.py" \
    --session-tag "$SESSION_TAG" \
    --dequant-cache-mode "$DEQUANT_CACHE_MODE" \
    --dequant-cache-path "$DEQUANT_CACHE_PATH" \
    "${CHILD_ARGS[@]}"
