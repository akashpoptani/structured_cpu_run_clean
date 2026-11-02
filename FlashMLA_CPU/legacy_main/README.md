# legacy_main

Main's earlier kernel sources, kept because main's existing paths still load
libraries built from them (their C interfaces differ from the prefill-branch
kernels that now use these file names one level up):

- `bench_prefill_sparse_amx.cpp` -> `libflashmla_prefill.so`, prefill kernel of
  `src/overrides/flash_attn_cpu.py` (`USE_AMX_ATTENTION=1`)
- `bench_indexer_amx.cpp` -> `libflashmla_indexer.so`, the AMX path of
  `kernel.fp8_index` (`USE_AMX_INDEXER`, decode and non-`USE_AMX_PREFILL_INDEXER` prefill)
- `prefill_sparse_gen_bins.py` -> test bins for the old prefill kernel

Built by `../build_libs.sh`.
