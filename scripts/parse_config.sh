#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: bash scripts/parse_config.sh [--format human|env] <TAG>" >&2
  echo "       bash scripts/parse_config.sh <TAG> [--format human|env]" >&2
  echo "Example: bash scripts/parse_config.sh TPCHECK" >&2
  echo "Example: bash scripts/parse_config.sh --format env TPCHECK" >&2
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_non_empty() {
  local name="$1"
  local value="${!name:-}"

  if [[ -z "$value" ]]; then
    die "required field is empty or unset: $name"
  fi
}

require_one_of() {
  local name="$1"
  local value="$2"
  shift 2

  local allowed
  for allowed in "$@"; do
    if [[ "$value" == "$allowed" ]]; then
      return 0
    fi
  done

  die "$name must be one of: $*; got: $value"
}

print_env_var() {
  local name="$1"
  local value="${!name:-}"

  printf '%s=%q\n' "$name" "$value"
}

FORMAT="human"
TAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --format)
      if [[ $# -lt 2 || -z "${2:-}" ]]; then
        usage
        exit 2
      fi
      FORMAT="$2"
      shift 2
      ;;
    -*)
      usage
      exit 2
      ;;
    *)
      if [[ -n "$TAG" ]]; then
        usage
        exit 2
      fi
      TAG="$1"
      shift
      ;;
  esac
done

if [[ -z "$TAG" ]]; then
  usage
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLEAN_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_DIR="$CLEAN_ROOT/scripts/configs"
BASELINE_CONFIG="$CONFIG_DIR/_baseline.env"

require_one_of FORMAT "$FORMAT" human env

[[ -f "$BASELINE_CONFIG" ]] || die "baseline config not found: $BASELINE_CONFIG"

# A tag match requires the filename to be <TAG>_<runmode>_... so e.g.
# TPCHECKREAL does not silently pick up TPCHECKREAL_NOLOAD / TPCHECKREAL_NOGEN.
shopt -s nullglob
override_matches=()
for candidate in "$CONFIG_DIR/${TAG}"_*.env; do
  rest="${candidate##*/${TAG}_}"
  case "$rest" in
    verify_*.env|bench_*.env|both_*.env|generate_*.env)
      override_matches+=("$candidate")
      ;;
  esac
done
shopt -u nullglob

if [[ "${#override_matches[@]}" -eq 0 ]]; then
  die "no override config found matching: $CONFIG_DIR/${TAG}_*.env"
fi

if [[ "${#override_matches[@]}" -gt 1 ]]; then
  echo "ERROR: multiple override configs found for tag '$TAG':" >&2
  printf '  %s\n' "${override_matches[@]}" >&2
  exit 1
fi

OVERRIDE_CONFIG="${override_matches[0]}"

# shellcheck disable=SC1090
source "$BASELINE_CONFIG"
# shellcheck disable=SC1090
source "$OVERRIDE_CONFIG"

case "${WEIGHTS_PRECISION:-}" in
  fp8)
    ACTIVE_MODEL_PATH="${MODEL_PATH_FP8:-}"
    ;;
  bf16)
    ACTIVE_MODEL_PATH="${MODEL_PATH_BF16:-}"
    ;;
  *)
    die "WEIGHTS_PRECISION must be one of: fp8 bf16; got: ${WEIGHTS_PRECISION:-<unset>}"
    ;;
esac

required_fields=(
  EXPERIMENT_ID
  RUN_MODE
  RUN_LABEL
  REAL_RUN
  NATIVE_NO_LOAD_WEIGHTS
  NATIVE_NO_GENERATE
  DEEPSEEK_REPO
  MODEL_PATH_FP8
  MODEL_PATH_BF16
  ACTIVE_MODEL_PATH
  GPU_REFERENCE_PATH
  OUTPUT_ROOT
  MODEL_ARGS_CONFIG_PATH
  DEQUANT_CACHE_MODE
  SBATCH_NODES
  SBATCH_TASKS_PER_NODE
  SBATCH_CPUS_PER_TASK
  SBATCH_MEM
  SBATCH_TIME
  DP_SIZE
  TP_SIZE
  EP_SIZE
  PP_SIZE
  SHARDING_MODE
  LIN_TOKENS
  LOUT_TOKENS
  BATCH_SIZE
  SESSION_MODE
  WEIGHTS_PRECISION
  KV_CACHE_DTYPE
  OMP_NUM_THREADS
  SKIP_INDEXER_BROADCAST
  PROMPT_SOURCE
  MEM_PROFILE
  TIME_PROFILE
  PROFILE_GRANULARITY
)

for field in "${required_fields[@]}"; do
  require_non_empty "$field"
done

require_one_of RUN_MODE "$RUN_MODE" verify bench both generate
require_one_of SHARDING_MODE "$SHARDING_MODE" tp2 dp2_epon dp2_true
require_one_of WEIGHTS_PRECISION "$WEIGHTS_PRECISION" fp8 bf16
require_one_of KV_CACHE_DTYPE "$KV_CACHE_DTYPE" fp8 bf16
require_one_of PROFILE_GRANULARITY "$PROFILE_GRANULARITY" off coarse detailed
require_one_of SESSION_MODE "$SESSION_MODE" 0 1
require_one_of SKIP_INDEXER_BROADCAST "$SKIP_INDEXER_BROADCAST" 0 1
require_one_of PROMPT_SOURCE "$PROMPT_SOURCE" synthetic infinitebench
require_one_of MEM_PROFILE "$MEM_PROFILE" 0 1
require_one_of TIME_PROFILE "$TIME_PROFILE" 0 1
require_one_of REAL_RUN "$REAL_RUN" 0 1
require_one_of DEQUANT_CACHE_MODE "$DEQUANT_CACHE_MODE" off write read read_or_write
require_one_of NATIVE_NO_LOAD_WEIGHTS "$NATIVE_NO_LOAD_WEIGHTS" 0 1
require_one_of NATIVE_NO_GENERATE "$NATIVE_NO_GENERATE" 0 1

if [[ "$NATIVE_NO_LOAD_WEIGHTS" == "1" && "$NATIVE_NO_GENERATE" == "1" ]]; then
  die "NATIVE_NO_LOAD_WEIGHTS=1 already implies no generation; do not also set NATIVE_NO_GENERATE=1"
fi

if [[ "$PP_SIZE" != "1" ]]; then
  die "PP_SIZE must remain 1 because pipeline parallelism is out of scope; got: $PP_SIZE"
fi

# EP_SIZE is descriptive: the expert split is driven by world_size, never by this
# field. Upstream MoE.__init__ shards experts by rank unconditionally, so experts
# are expert-parallel in BOTH modes -- tp2 included. Only the dp2_epon case is
# checked here, because install_ep_moe is the only code path that consumes EP_SIZE.
#
# NOTE: the tp2 configs declaring EP_SIZE=1 are mislabelled -- they really run
# 128 experts/rank. Left alone for now; correcting them touches ~20 files.
# BF16_CKPT_INDEX silently overrides the FP8 load path and forces ModelArgs.dtype
# to bf16, so a config declaring WEIGHTS_PRECISION="fp8" alongside it describes a
# run that never happens. Reject the contradiction rather than resolve it quietly:
# DPHET/DPHOM read as FP8 runs while actually running BF16 end to end.
if [[ -n "$BF16_CKPT_INDEX" ]]; then
  if [[ "$WEIGHTS_PRECISION" != "bf16" ]]; then
    die "BF16_CKPT_INDEX is set, so weights are BF16; WEIGHTS_PRECISION must be \"bf16\", got: $WEIGHTS_PRECISION"
  fi
  if [[ "$DEQUANT_FP8_WEIGHTS" != "none" ]]; then
    die "BF16_CKPT_INDEX is set, so there is no FP8 weight to dequantize; DEQUANT_FP8_WEIGHTS must be \"none\", got: $DEQUANT_FP8_WEIGHTS"
  fi
  if [[ "$DEQUANT_CACHE_MODE" != "off" ]]; then
    die "BF16_CKPT_INDEX takes precedence over DEQUANT_CACHE_MODE; set DEQUANT_CACHE_MODE=\"off\", got: $DEQUANT_CACHE_MODE"
  fi
fi

if [[ "$PROMPT_SOURCE" == "infinitebench" ]]; then
  if [[ -z "$PROMPT_CORPUS_PATH" ]]; then
    die "PROMPT_SOURCE=infinitebench requires PROMPT_CORPUS_PATH"
  fi
  if [[ ! -d "$PROMPT_CORPUS_PATH" ]]; then
    die "PROMPT_CORPUS_PATH does not exist: $PROMPT_CORPUS_PATH (run scripts/fetch_infinitebench.sh from a login node)"
  fi
fi

if [[ "$SHARDING_MODE" == "dp2_epon" || "$SHARDING_MODE" == "dp2_true" ]]; then
  if [[ "$EP_SIZE" != "$DP_SIZE" ]]; then
    die "SHARDING_MODE=$SHARDING_MODE requires EP_SIZE == DP_SIZE (world_size); got EP_SIZE=$EP_SIZE DP_SIZE=$DP_SIZE"
  fi
fi

# dp2_true shards the batch across ranks, and reduce_scatter splits equally, so a
# batch that does not divide evenly would misalign the slices. B=1 is allowed and
# falls back to replicated attention (install_true_dp_moe logs this loudly) --
# the alternative would be overriding upstream Transformer.forward for the
# zero-rows-on-a-rank case, i.e. copying upstream numerics into this repo.
if [[ "$SHARDING_MODE" == "dp2_true" && "$BATCH_SIZE" != "1" ]]; then
  if (( BATCH_SIZE % DP_SIZE != 0 )); then
    die "SHARDING_MODE=dp2_true requires BATCH_SIZE divisible by DP_SIZE (or BATCH_SIZE=1); got BATCH_SIZE=$BATCH_SIZE DP_SIZE=$DP_SIZE"
  fi
fi

if [[ "$FORMAT" == "env" ]]; then
  env_fields=(
    TAG
    OVERRIDE_CONFIG
    EXPERIMENT_ID
    RUN_MODE
    RUN_LABEL
    REAL_RUN
    NATIVE_NO_LOAD_WEIGHTS
    NATIVE_NO_GENERATE
    PROJECT_ROOT
    CLEAN_ROOT
    DEEPSEEK_REPO
    MODEL_PATH_FP8
    MODEL_PATH_BF16
    ACTIVE_MODEL_PATH
    TOKENIZER_PATH
    GPU_REFERENCE_PATH
    OUTPUT_ROOT
    MODEL_ARGS_CONFIG_PATH
    SHARDED_CKPT_PATH
    DEQUANT_FP8_WEIGHTS
    DEQUANT_CACHE_MODE
    DEQUANT_CACHE_PATH
    BF16_CKPT_INDEX
    SBATCH_NODES
    SBATCH_TASKS_PER_NODE
    SBATCH_CPUS_PER_TASK
    SBATCH_MEM
    SBATCH_TIME
    SBATCH_PARTITION
    SBATCH_ACCOUNT
    DP_SIZE
    TP_SIZE
    EP_SIZE
    PP_SIZE
    SHARDING_MODE
    LIN_TOKENS
    LOUT_TOKENS
    BATCH_SIZE
    SESSION_MODE
    WEIGHTS_PRECISION
    KV_CACHE_DTYPE
    OMP_NUM_THREADS
    OMP_PROC_BIND
    OMP_PLACES
    SKIP_INDEXER_BROADCAST
    PROMPT_SOURCE
    PROMPT_CORPUS_PATH
    PROMPT_SEED
    PROMPT_TASK
    DEQUANT_FP8_WEIGHTS
    MEM_PROFILE
    TIME_PROFILE
    PROFILE_GRANULARITY
  )

  for field in "${env_fields[@]}"; do
    print_env_var "$field"
  done
  exit 0
fi

cat <<EOF
Resolved clean CPU config
-------------------------
TAG: $TAG
Override file: $OVERRIDE_CONFIG
EXPERIMENT_ID: $EXPERIMENT_ID
RUN_MODE: $RUN_MODE
RUN_LABEL: $RUN_LABEL
REAL_RUN: $REAL_RUN
NATIVE_NO_LOAD_WEIGHTS: $NATIVE_NO_LOAD_WEIGHTS
NATIVE_NO_GENERATE: $NATIVE_NO_GENERATE
SHARDING_MODE: $SHARDING_MODE
DP_SIZE / TP_SIZE / EP_SIZE / PP_SIZE: $DP_SIZE / $TP_SIZE / $EP_SIZE / $PP_SIZE
LIN_TOKENS / LOUT_TOKENS / BATCH_SIZE: $LIN_TOKENS / $LOUT_TOKENS / $BATCH_SIZE
WEIGHTS_PRECISION: $WEIGHTS_PRECISION
KV_CACHE_DTYPE: $KV_CACHE_DTYPE
ACTIVE_MODEL_PATH: $ACTIVE_MODEL_PATH
MODEL_ARGS_CONFIG_PATH: $MODEL_ARGS_CONFIG_PATH
SHARDED_CKPT_PATH: $SHARDED_CKPT_PATH
DEQUANT_FP8_WEIGHTS: $DEQUANT_FP8_WEIGHTS
DEQUANT_CACHE_MODE: $DEQUANT_CACHE_MODE
DEQUANT_CACHE_PATH: $DEQUANT_CACHE_PATH
SBATCH_NODES: $SBATCH_NODES
SBATCH_CPUS_PER_TASK: $SBATCH_CPUS_PER_TASK
SBATCH_MEM: $SBATCH_MEM
SBATCH_TIME: $SBATCH_TIME
OMP_NUM_THREADS: $OMP_NUM_THREADS
MEM_PROFILE: $MEM_PROFILE
TIME_PROFILE: $TIME_PROFILE
PROFILE_GRANULARITY: $PROFILE_GRANULARITY
SESSION_MODE: $SESSION_MODE
SKIP_INDEXER_BROADCAST: $SKIP_INDEXER_BROADCAST
PROMPT_SOURCE: $PROMPT_SOURCE
PROMPT_CORPUS_PATH: $PROMPT_CORPUS_PATH
GPU_REFERENCE_PATH: $GPU_REFERENCE_PATH
OUTPUT_ROOT: $OUTPUT_ROOT
EOF
