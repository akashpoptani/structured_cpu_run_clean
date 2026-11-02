#!/usr/bin/env bash
# Launch native_run.py via torch.distributed.run inside a Slurm allocation.
#
# Mirrors the legacy ../structured_cpu_run/verification/run_milestone_bench.sbatch
# launch shape:
#   srun --nodes=N --ntasks=N --ntasks-per-node=1   (provided by sbatch)
#     python -m torch.distributed.run
#         --nnodes=N --nproc-per-node=1
#         --node-rank=$SLURM_NODEID
#         --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT
#         scripts/native_run.py --resolved-config ... --reference-group ... --case-id ...
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
# Kernel toggles/tunables, read by src/overrides/flash_attn_cpu.py.
export FLASHMLA_DECODE="${FLASHMLA_DECODE:-1}"
export FLASHMLA_BLOCK_SIZE="${FLASHMLA_BLOCK_SIZE:-0}"
export FLASHMLA_THREADS="${FLASHMLA_THREADS:-0}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
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

# Route gloo over InfiniBand, not the 1 GbE management interface.
# getent resolves the hostname to eth0 (1000 Mb/s); ib0 is 200000 Mb/s and was
# sitting idle. Measured consequence: all_reduce of [32768,7168] fp32 (0.94 GB)
# took 8.36 s = 0.1 GB/s, which is 1 Gb/s wire speed. Prefill at Lin=120k fires
# 476 such collectives (61 out_proj + 58 MoE layers x 4 chunks) -- about 66 min
# of a 129 min prefill spent on a misconfigured network path.
# OPT-IN via USE_IB=1. An ib0 that is UP but not ROUTABLE between the allocated
# nodes makes gloo hang at rendezvous rather than fall back -- a standalone test
# of exactly this hung for 8+ minutes on a 90-second workload. Defaulting it on
# would turn every run into a potential hang, so it stays behind a flag until a
# routability check passes.
# If the batch script already resolved ib0 for every rank (the normal path),
# GLOO_SOCKET_IFNAME and MASTER_ADDR are both set and this block is skipped --
# no nested srun per rank, and no way for ranks to disagree.
if [[ "${USE_IB:-0}" == "1" && -z "${GLOO_SOCKET_IFNAME:-}" ]] && ip -o -br addr show ib0 2>/dev/null | grep -q UP; then
    export GLOO_SOCKET_IFNAME=ib0
    # MASTER_ADDR must be the master's ib0 ADDRESS. Resolving a "${host}.ib"
    # hostname does not work here -- that name does not exist, the lookup
    # silently returns nothing, MASTER_ADDR stays on eth0, and gloo then tries
    # to reach an eth0 address through the ib0 interface and HANGS at rendezvous
    # (observed: 8+ min on an 80 s workload). Ask the master node directly.
    if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
        _ibhost="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
        # '|| true' is load-bearing. This is a NESTED srun inside an srun
        # step; under contention it can fail or be killed, and `set -e` then
        # aborts the whole rank on the assignment -- before the fallback
        # below can run. Job 27933115 died exactly this way: rank 0 resolved
        # the address, rank 1's nested srun exited 192, rank 1 never reached
        # init_process_group, and rank 0 sat in rendezvous until
        # "Timed out after 901 seconds waiting for clients. 1/2 clients joined".
        # The fallback was always correct; it was just unreachable.
        _ibaddr="$(srun --nodes=1 --ntasks=1 -w "$_ibhost" --overlap \
                     ip -o -4 addr show ib0 2>/dev/null \
                     | awk '{print $4}' | cut -d/ -f1 | head -n1 || true)" || true
        if [[ "$_ibaddr" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
            export MASTER_ADDR="$_ibaddr"
        else
            echo "[run-native] WARNING: could not resolve ib0 address for $_ibhost;" \
                 "falling back to eth0 to avoid a rendezvous hang"
            unset GLOO_SOCKET_IFNAME
        fi
    fi
fi
echo "[run-native] GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-<default>} MASTER_ADDR=$MASTER_ADDR"

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
    "$CLEAN_ROOT/scripts/native_run.py" \
    --resolved-config "$RESOLVED_CONFIG_PATH" \
    --reference-group "$REFERENCE_GROUP" \
    "${CASE_FLAG[@]}" \
    "${EXTRA_FLAGS[@]}"
