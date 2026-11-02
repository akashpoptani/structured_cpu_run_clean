#!/usr/bin/env bash
# Build and run every decode benchmark here, on one AMX node, with the thread
# environment pinned explicitly and WARMUP ITERATIONS DISCARDED.
#
# Both of those are load-bearing. Measured on the author's own prebuilt binary
# (job 27924213) -- unchanged code, same node, same data:
#
#   OMP_NUM_THREADS=32 -> 3.70 ms        OMP_NUM_THREADS=64 -> 7.76 ms
#   oneDNN iteration 1 -> Q*K^T 1.398 ms
#   oneDNN iteration 2 -> Q*K^T 0.103 ms   (13.6x faster -- JIT on first call)
#
# A run that leaves OMP_NUM_THREADS unset and quotes iteration 1 lands ~24x off.
# That is precisely what the first version of this script did.
#
# Toolchain is also not optional: g++ 8.5 rejects -mamx-*, binutils 2.30 does not
# know tileloadd/tdpbf16ps. gcc/13.2.0 + binutils/2.45 fix both.
# -march=sapphirerapids covers -mavx512bf16, which the README's explicit flag
# list omits (without it _mm512_cvtneps_pbh fails to inline).
#
# Must run on a COMPUTE node -- the login node reports no amx_bf16/amx_tile.
#
# Dense has ONE kernel now: oneDNN. The custom dense AMX kernel was dropped --
# oneDNN beat it at every thread count measured (5.89 vs 13.6 ms at 72, 5.87 vs
# 7.53 at 96) because its brg_matmul:amx does its own blocking, while the custom
# path staged K into VNNI by hand. Sparse keeps its custom AMX kernel, which
# wins there (0.504 vs 0.842 ms at 32 threads) because the gather + FP8 dequant
# + RoPE fuse into one pass and oneDNN cannot express that.
set -uo pipefail                 # NOT -e: one failing build must not stop the rest

cd "$(dirname "${BASH_SOURCE[0]}")"
THREAD_LIST="${THREAD_LIST:-32 48}"

echo "=== host $(hostname) ==="
echo "AMX: $(grep -oE 'amx_bf16|amx_tile|avx512_bf16' /proc/cpuinfo | sort -u | tr '\n' ' ')"
echo "cores available: $(nproc)"

module load gcc/13.2.0 binutils/2.45 2>/dev/null
echo "g++: $(g++ --version | head -1)"
echo "as : $(as --version | head -1)"

DNNL_ROOT="${DNNL_ROOT:-/home/akashpt/DeepSeekRun/FlashMLA_CPU/oneDNN/install}"
DNNL_INC="$DNNL_ROOT/include"; DNNL_LIB="$DNNL_ROOT/lib64"
[[ -f "$DNNL_LIB/libdnnl.so" ]] && echo "oneDNN: $DNNL_ROOT" \
                                || echo "oneDNN: MISSING at $DNNL_ROOT"
export LD_LIBRARY_PATH="$DNNL_LIB:${LD_LIBRARY_PATH:-}"

# Plan item 2 measured OMP_PROC_BIND=close at 5.5x slower on decode; false is the
# project default. Set it rather than inheriting whatever the shell happened to have.
export OMP_PROC_BIND=false
export OMP_PLACES=cores
echo "OMP_PROC_BIND=$OMP_PROC_BIND OMP_PLACES=$OMP_PLACES   thread sweep: $THREAD_LIST"
echo

FLAGS="-O3 -fopenmp -march=sapphirerapids"
declare -A BUILT
echo "=== BUILD ==="
for t in bench_sparse_amx; do
    g++ $FLAGS "$t.cpp" -o "$t" -lstdc++ 2>"build_$t.log" \
        && { echo "  OK      $t"; BUILT[$t]=1; } \
        || { echo "  FAILED  $t"; tail -3 "build_$t.log" | sed 's/^/          /'; }
done
for t in bench_sparse_onednn bench_dense_onednn; do
    g++ $FLAGS -I"$DNNL_INC" "$t.cpp" -o "$t" \
        -L"$DNNL_LIB" -Wl,-rpath,"$DNNL_LIB" -ldnnl -lstdc++ 2>"build_$t.log" \
        && { echo "  OK      $t"; BUILT[$t]=1; } \
        || { echo "  FAILED  $t"; tail -2 "build_$t.log" | sed 's/^/          /'; }
done

# Each kernel prints its per-stage breakdown once per profiled pass. The FIRST
# pass is cold (oneDNN JIT, first-touch pages), so report the LAST one.
last_breakdown() {
    awk '/K-Packing|oneDNN Q|^1\.|^1a\./ {blk=""}
         {blk = blk $0 "\n"}
         /Total Map Phase|Total Time|Total Loop Time/ {last=blk}
         END {printf "%s", last}' "$1" \
    | grep -E "^ *[0-9]+[a-z]?\. |Total |Barrier |Phase 2" | head -10
}

for T in $THREAD_LIST; do
  echo
  echo "################ OMP_NUM_THREADS=$T ################"
  for t in bench_sparse_amx bench_sparse_onednn bench_dense_onednn; do
      [[ -n "${BUILT[$t]:-}" ]] || { echo "--- $t: not built ---"; continue; }
      echo "--- $t ---"
      out="run_${t}_${T}.log"
      OMP_NUM_THREADS=$T FORCE_EFFECTIVE_THREADS=$T timeout 900 "./$t" >"$out" 2>&1
      last_breakdown "$out" | sed 's/^/   /'
      grep -oE "Max Diff[^$]*" "$out" | tail -1 | sed 's/^/   /'
      grep -oE "Total End-to-End Time *\| *[0-9.]+" "$out" | tail -1 \
          | grep -oE "[0-9.]+$" | sed 's/^/   END-TO-END mean of 50 iters: /;s/$/ ms/'
      echo
  done
done
echo "=== DONE ==="
