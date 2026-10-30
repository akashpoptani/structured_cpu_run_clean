#!/usr/bin/env bash
# Build the model's AMX kernels as shared libraries, loaded from Python with ctypes.
#   amx_prefill_attention.so  <- bench_prefill_sparse_amx.cpp, loaded by src/overrides/amx_prefill_attention.py
#   amx_prefill_indexer.so    <- bench_indexer_amx.cpp,        loaded by src/overrides/amx_prefill_indexer.py
# One library per source: each defines its own AMX helpers (init_amx, pack), so
# linking them into one object would clash.
#
# -DFLASHMLA_SHARED_LIB drops the benchmark main() and keeps the C entry point.
# -static-libstdc++ -static-libgcc is load-bearing: the model job does NOT
# `module load gcc`, so a dynamically linked .so dies with
#   OSError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.32' not found
# Run on a node with AMX (any ramanvr node); -march=sapphirerapids needs gcc 13.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
module load gcc/13.2.0 binutils/2.45 2>/dev/null || true
FLAGS="-O3 -fopenmp -march=sapphirerapids -fPIC -shared -DFLASHMLA_SHARED_LIB -static-libstdc++ -static-libgcc"
g++ $FLAGS bench_prefill_sparse_amx.cpp -o amx_prefill_attention.so
g++ $FLAGS bench_indexer_amx.cpp        -o amx_prefill_indexer.so
for f in amx_prefill_attention.so amx_prefill_indexer.so; do
    printf "  %-24s %s  glibcxx_refs=%s\n" "$f" "$(nm -D $f | grep -o 'flashmla_[a-z_]*' | head -1)" \
        "$(objdump -p $f | grep -ci glibcxx || true)"
done
