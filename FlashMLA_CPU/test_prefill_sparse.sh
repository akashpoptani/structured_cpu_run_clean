#!/usr/bin/env bash
# Standalone correctness test for the sparse prefill kernel (bench_prefill_sparse_amx.cpp).
# Needs an AMX node, e.g.:
#   srun -p ramanvr -A ramanvr -N1 -n1 -c32 --mem=64G -t 00:20:00 bash FlashMLA_CPU/test_prefill_sparse.sh
#
# Three sets of real top-k indices, 64 heads (one TP2 rank); every one must print PASS:
#   rows0     queries 0-255: their top-k includes FUTURE positions the causal filter must drop
#   boundary  queries 1984-2111: crossing 2048, where top-k stops selecting every past token
#   default   queries 2048-2303: plain sparse case
# Timing printed here is setup-dominated (small Sq); it is not a performance number.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
PY="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
TOPK_NPY="${TOPK_NPY:-/home/akashpt/DeepSeekRun/structured_cpu_run_clean/results_clean/results/IDXDUMP/topk/topk_layer000.npy}"
OUT="${OUT:-$ROOT/tmp/prefill_sparse_test}"
module load gcc/13.2.0 binutils/2.45 2>/dev/null || true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}" BENCH_ITERS="${BENCH_ITERS:-3}"
mkdir -p "$OUT"
echo "=== host $(hostname)  topk=$TOPK_NPY  out=$OUT"
g++ -O3 -fopenmp -march=sapphirerapids "$HERE/bench_prefill_sparse_amx.cpp" -o "$OUT/bench" || exit 1
for spec in "rows0:0:256" "boundary:1984:128" "default::256"; do
    IFS=: read -r name q0 sq <<<"$spec"
    qarg=(); [[ -n "$q0" ]] && qarg=(--q-start "$q0")
    echo; echo "=== $name"
    "$PY" "$HERE/prefill_sparse_gen_bins.py" --topk-npy "$TOPK_NPY" "${qarg[@]}" --sq "$sq" --heads 64 \
        --out "$OUT/bins_$name" | grep -E "^B=|future"
    SPARSE_DIR="$OUT/bins_$name" "$OUT/bench" 2>&1 | grep -E "Max Diff|FATAL"
done
