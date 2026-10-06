# PROFILER.md — design for the clean-lane profiler

Status: **design only, not implemented.** Supersedes the approach in `profiling/LEGACY_PROFILER.md`.

## 1. Non-negotiables

| # | Rule | Why (legacy failure it fixes) |
|---|---|---|
| 1 | One file: `src/clean_inference/profiler.py` | Legacy had 2 unrelated profilers, 2 schemas, 4 files |
| 2 | Library, not an entrypoint. Enabled by `TIME_PROFILE` / `MEM_PROFILE` / `PROFILE_GRANULARITY` | Legacy required running `generate_profiled.py` instead of the real script → profiled a *different* code path than the validated one |
| 3 | Never re-implement a `forward()` | Legacy profiler #1 forked MLA/MoE/Block math to insert timers; the profiled and production paths then drifted |
| 4 | Inclusive **and** exclusive (self) time | Legacy was inclusive-only; its printed "component sum" double-counted `step_total ⊇ forward ⊇ ...` |
| 5 | Calibrate and report timer overhead | Legacy had ~1,400 wrap sites/forward with zero overhead accounting |
| 6 | Sample memory | Legacy measured **none**, yet OOM was the actual failure mode |
| 7 | Zero cost when off | Must be safe to leave the tags in production code |

## 2. Two instrumentation mechanisms

**(a) Module hooks — automatic, no code edits.**
`register_forward_pre_hook` / `register_forward_hook` on `Block`, `MLA`, `Indexer`, `MoE`, `MLP`, `Expert`, `Linear`.
Gives the coarse tree for free and cannot fork the math (rule 3).

**(b) `LLM_INSTRUMENT` — manual, for sub-module regions.**
```python
from clean_inference.profiler import LLM_INSTRUMENT
with LLM_INSTRUMENT("mla.scores"):
    scores = torch.einsum(...)
```
Use only where a hook cannot reach: inside `src/overrides/kernel.py` (`fp8_index`, `act_quant`),
`dequant_weights.py`, `weight_loading.py`, `generation.py`. **Never inside upstream `model.py`.**

Zero-cost-when-off: `LLM_INSTRUMENT` returns a shared no-op context object when disabled — one
module-global bool check, no allocation.

## 3. Key schema

`{phase}.{region}` where phase ∈ `load | prefill | decode`. Layer regions prefixed `L00..L60`.
Per key: `incl_s, excl_s, count, mean_s, min_s, max_s`.
Exclusive time via a thread-local stack: on exit, `incl[self] += d` and `child[parent] += d`;
then `excl = incl - child`.

## 4. Granularity (`PROFILE_GRANULARITY`)

| Level | Hooks installed | Approx. wrap sites / forward |
|---|---|---|
| `off` | none | 0 |
| `coarse` | Block, MLA, MoE + phase boundaries | ~180 |
| `detailed` | + Indexer, Expert, Linear, all `LLM_INSTRUMENT` | ~1,400 |

Default `coarse`. `detailed` only for a targeted hunt — it perturbs absolute times.

## 5. Memory

`/proc/self/status` → `VmRSS` (current) + `VmHWM` (peak, kernel-maintained, free).
Sampled at: load done, prefill done, every N decode steps, run end. Never in a hot loop.
Cross-check against `sacct -j <id> --format=MaxRSS` (cgroup high-water, per step).

## 6. Output

`results_clean/profiles/<TAG>/<JOB_ID>/profile.rank{N}.json` — one file per rank, written
temp+rename (atomic), never a shared path. Schema `{metadata, timer_overhead_ns, timings}`.
Rank 0 additionally writes `summary.json` referencing the per-rank files.

## 7. Open questions

1. Do we need per-thread timings, or is per-rank enough? (Per-thread costs a lot more.)
2. Should `detailed` auto-subtract calibrated timer overhead, or only report it?
3. Decode-step checkpointing interval — legacy used 64; is that enough to survive a walltime kill?
