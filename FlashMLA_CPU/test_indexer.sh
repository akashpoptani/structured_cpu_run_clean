#!/usr/bin/env bash
# Standalone correctness + speed test for the AMX indexer kernel (bench_indexer_amx.cpp).
# Needs an AMX node, e.g.:
#   srun -p ramanvr -A ramanvr -N1 -n1 -c96 --mem=64G -t 00:20:00 bash FlashMLA_CPU/test_indexer.sh
#
# Random inputs (fp8-representable for the fp8 path, full bf16 for the bf16 path),
# 64 heads x 128 dims (one TP2 rank's indexer).
# Every shape must print PASS (64 sampled rows vs a double-precision reference).
# Shapes are M = queries in a prefill chunk, N = keys = end_pos.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-$(dirname "$HERE")/tmp/indexer_test}"
module load gcc/13.2.0 binutils/2.45 2>/dev/null || true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-88}" OMP_PROC_BIND=close OMP_PLACES=cores
mkdir -p "$OUT"
echo "=== host $(hostname)  threads=$OMP_NUM_THREADS"
g++ -O3 -fopenmp -march=sapphirerapids "$HERE/bench_indexer_amx.cpp" -o "$OUT/bench" || exit 1
for kv in 0 1; do              # IDX_KV_BF16: 0 = fp8 path (x k_s), 1 = bf16 path (no k_s)
for s in "16 64" "2048 4096" "2048 32768" "2048 120000"; do
    set -- $s
    echo; echo "=== kv=$([ $kv = 1 ] && echo bf16 || echo fp8) M=$1 N=$2"
    IDX_KV_BF16=$kv IDX_M=$1 IDX_N=$2 "$OUT/bench" 2>&1 | grep -E "Max Diff|time|FATAL"
done
done
