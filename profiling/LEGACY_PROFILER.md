# Legacy Profiler Inspection

Read-only inspection of the profiling code in `../structured_cpu_run/`, written
on the `profile-llm-cpu` branch to inform the design of a clean-lane profiler.
No legacy files were modified. No profiler is implemented here yet — this is
documentation only.

## 1. Executive summary

The legacy lane has **two independent, unrelated profilers** plus a JSON
recovery helper:

1. **Fine-grained hierarchical profiler** — `without_vllm/profiling/profiler.py`
   (the `Profiler` class) driven by `without_vllm/profiling/generate_profiled.py`.
   It monkey-patches the `forward()` of every model component (Transformer,
   Block, MLA, Indexer, MoE/MLP, plus sub-operations inside each) with
   `with profiler.timed(name): ...` context managers. Timings are keyed
   `{phase}.{name}` where phase ∈ {load, prefill, decode}, and per-layer keys
   are prefixed `L00`, `L01`, …. It aggregates total/count/mean/min/max per key
   and writes a per-rank JSON report, with periodic checkpoint dumps during
   decode so partial data survives a job timeout. This is the deep, "where does
   the time go inside one forward pass" tool.

2. **Coarse per-class profiler** — `optimisations/profile_cpu.py`. It wraps
   `forward()` on every *instance* of a fixed set of classes (MLA, MoE, Indexer,
   MLP, Block, Linear, ColumnParallelLinear, RowParallelLinear, RMSNorm, Expert)
   and accumulates `{calls, total_s, max_s}` per class. After a short generation
   it prints a single sorted table ("which class eats the most wall time") to
   stdout on rank 0. This is the "what's the next bottleneck class to attack"
   tool. It does a warmup forward before installing timers.

Both are **wall-clock only** (`time.perf_counter()`), **time-only** (no memory
instrumentation of any kind), and both rely on monkey-patching rather than any
config-driven switch. Neither uses the `TIME_PROFILE` / `MEM_PROFILE` /
`PROFILE_GRANULARITY` fields — those exist only in the clean lane's config
schema and are **not** read by any legacy code.

## 2. Files inspected

| File | Why it matters |
|---|---|
| `without_vllm/profiling/profiler.py` | The `Profiler` class: hierarchical timer, summary stats, per-rank JSON writer, human report. Core of profiler #1. |
| `without_vllm/profiling/generate_profiled.py` | The driver that instruments the model (`instrument_model` / `_instrument_block`) and runs a profiled generation loop with phase switching + checkpoint saves. |
| `optimisations/profile_cpu.py` | Profiler #2: per-class forward-wrapping timer + sorted stdout table. Reuses the dp2_epon runtime path (late dist init, expert pruning, custom MoE forward). |
| `without_vllm/profiling/recover_profile_json.py` | Post-hoc tool to salvage JSON from race-corrupted shared-path profile files (historical bug, since fixed by rank-suffixing). |
| `without_vllm/profiling/run_profile.sbatch` | Slurm launcher for profiler #1: 2 nodes × 1 task, torchrun, env-driven (PROMPT_FILE, MAX_NEW_TOKENS, OUTPUT_JSON, …). |
| `without_vllm/profiling/submit_all.sh` | Generates prompts and submits a matrix of profiling cases (LIN × LOUT × BATCH × MEM × CPUS × PREFILL_CHUNK). Source of the only memory data (MaxRSS comments). |
| `without_vllm/profiling/make_batch_prompt.py` | Builds a prompt file with `BATCH` copies of an exactly-`LIN`-token prompt (chat-template-overhead-aware). Feeds profiler #1. |
| `optimisations/run_profile_cpu.sbatch` | Slurm launcher for profiler #2 (per-class). |
| `verification/verify_cpu.py`, `verification/bench_cpu.py` | Checked for profiling hooks: they use ad-hoc `perf_counter` prints (PREFILL/DECODE timing) but contain **no** structured profiler and no `TIME_PROFILE`/`MEM_PROFILE`. |
| `verification/run_milestone_bench.sbatch` | The bench/verify launcher; emits TTFT/TPOT via stdout parsing, not via either profiler. |

## 3. How profiling is enabled

There is **no config flag** that turns profiling on. Profiling is enabled by
*running a different entrypoint script*:

- **Profiler #1**: run `generate_profiled.py` instead of `generate_cpu.py`.
  Controlled entirely by CLI args:
  `--repo-dir --ckpt-path --config --prompt/--prompt-file --max-new-tokens
  --max-batch-size --max-seq-len --temperature --top-p --threads --seed
  --output-json --checkpoint-interval --prefill-chunk-size`.
  Launched by `run_profile.sbatch`, which is driven by env vars exported from
  `submit_all.sh` (`PROMPT_FILE`, `MAX_NEW_TOKENS`, `MAX_BATCH_SIZE`,
  `MAX_SEQ_LEN`, `OUTPUT_JSON`, plus optional `THREADS`, `TEMPERATURE`,
  `TOP_P`, `CHECKPOINT_INTERVAL`, `PREFILL_CHUNK_SIZE`, `MASTER_PORT`).
  Distributed-ness comes from `WORLD_SIZE`/`RANK` (torchrun) as usual.
- **Profiler #2**: run `optimisations/profile_cpu.py`. CLI: `--repo-dir
  --ckpt-path --config --lin --lout --threads`. Reads `SHARDING_MODE`,
  `WORLD_SIZE`, `RANK`, `CKPT_BF16_PATH`/`CKPT_BF16_INDEX`,
  `DEQUANT_FP8_WEIGHTS` (via `get_scope_from_env`), `FAST_LINEAR` from the
  environment — i.e. the same env contract as `verify_cpu.py`.

No `RUN_CONFIG` JSON drives either profiler (that mechanism belongs to
`verify_cpu.py`/`run_config.py`, which the profilers do not call). No
mode-specific profiling behavior except that profiler #2 re-applies the
dp2_epon expert-pruning + custom MoE forward before timing.

## 4. What time profiling measures

### Profiler #1 (`generate_profiled.py` + `profiler.py`)

Every region is a `with profiler.timed(name)` block; the stored key is
`{phase}.{name}`. Phase is set via `profiler.set_phase(...)` to `load`,
`prefill`, or `decode`. All timings are **rank-local** (each rank records its
own); the JSON is written per-rank. Output field per key:
`{total_s, count, mean_s, min_s, max_s}` under `timings`.

Coarse / loop-level (keyed `{phase}.NAME`):

| Region | Where | Granularity |
|---|---|---|
| `model_construct` | `main`, load phase | one-shot |
| `model_load` | `main`, load phase | one-shot |
| `step_total` | `generate_profiled` loop | per generation step (outermost) |
| `forward` | inside `step_total` | per step (whole model forward) |
| `sampling` | inside `step_total` | per step |
| `embedding` | `transformer_forward` | per forward |
| `final_norm` | `transformer_forward` | per forward |
| `lm_head` | `transformer_forward` | per forward |
| `logits_allgather` | `transformer_forward` (ws>1) | per forward |

Per-layer / per-op (keyed `{phase}.L{ID}.…`, recorded for all 61 layers):

- Block: `L{ID}.attn_norm`, `L{ID}.attn`, `L{ID}.ffn_norm`, `L{ID}.ffn`.
- MLA: `L{ID}.mla.q_proj`, `.kv_proj`, `.kv_quant`, `.wkv_dequant`, `.scores`,
  `.indexer`, `.mask_softmax`, `.value`, `.wo`.
- Indexer: `L{ID}.idx.q_proj`, `.k_proj`, `.hadamard`, `.quant_cache`,
  `.weights_proj`, `.fp8_index`, `.topk`, `.broadcast`.
- MoE: `L{ID}.moe.gate`, `.experts`, `.shared`, `.allreduce` (ws>1).
- Dense MLP: `L{ID}.mlp.w1_w3`, `.silu`, `.w2`.

So per token in decode there are ~61 × (4 block + 9 mla + 8 idx + 3-4 ffn) ≈
**1,400+ timed regions per forward**, each a context-manager enter/exit.

### Profiler #2 (`profile_cpu.py`)

Per-*class* aggregation, not per-region. Wraps `forward` on each instance of
the 10 target classes; records `{n, total, max}` per class name. Single table
to stdout sorted by total time, with `%total`, `avg ms`, `max ms`, and a
`ms/tok` figure. Rank-0-only print. No JSON, no phase split. Warmup forward is
run untimed before timers are installed.

## 5. What memory profiling measures

**Nothing.** Neither profiler instruments memory. There is no RSS, VMS,
cgroup, `resource.getrusage`, `tracemalloc`, `psutil`, or `/proc/self/status`
reading anywhere in the profiling or optimisations code.

The only memory data in the legacy lane is **out-of-band**: comments in
`submit_all.sh` record Slurm `MaxRSS` observations from earlier OOM failures
(e.g. "2k b=64 with --mem=500G OOM'd at MaxRSS=524GB during init", "indexer
k_cache alone is ~122GB per rank at b=64 seq=4096"). That is `sacct`-derived
after the fact, not in-process sampling. Memory is a complete gap.

## 6. Output files and schema

### Profiler #1
- `Profiler.save_json(path, metadata)` writes the report. If `metadata` has a
  `rank` key and the path has a suffix, the path is rewritten
  `foo.json` → `foo.rank{rank}.json` (per-rank, anti-race).
- Final path comes from `--output-json` (set per case by `submit_all.sh`,
  written under `/scratch/.../DeepSeekRun_runtime/profiling_results/`).
- Schema:
  ```json
  {
    "metadata": { "lin_target", "lout_target", "batch_size", "max_batch_size",
                  "max_seq_len", "world_size", "rank", "threads", "temperature",
                  "top_p", "seed", "hostname", "slurm_job_id",
                  "generation_elapsed_s", "total_tokens_generated",
                  "decode_steps", "tokens_per_second" },
    "timings": { "<phase>.<name>": { "total_s", "count", "mean_s",
                                     "min_s", "max_s" }, ... }
  }
  ```
- Checkpoint dumps every `--checkpoint-interval` decode steps reuse the same
  path (overwrite) with `metadata.decode_steps_so_far` added.
- No cross-rank aggregation step — separate rank files; analysis is manual.

### Profiler #2
- No file output. Human-readable table to stdout only (captured in the Slurm
  `.out` log).

### Recovery
- `recover_profile_json.py` turns a race-corrupted `profile_*.json`
  (concatenated rank objects) into `*.recovered_rank{N}.json` via
  `JSONDecoder.raw_decode` looping.

## 7. Rank / distributed behavior

- **Profiler #1**: every rank records its own timings and writes its own
  `foo.rank{rank}.json`. Non-zero ranks have `print` silenced
  (`print = lambda *_, **__: None`). No barriers around timed regions, no
  rank-0 aggregation. The rank-suffix in `save_json` exists specifically
  because an earlier version had both ranks writing the same path on a shared
  filesystem → interleaved truncate/append → invalid JSON (the reason
  `recover_profile_json.py` exists). Fixed by suffixing, but there is still no
  barrier, so two ranks can write concurrently to *different* paths fine.
- **Profiler #2**: timers run on all ranks but only rank 0 prints; other ranks
  produce no output. dp2_epon path does `dist.all_reduce` inside the timed MoE
  forward, so the `MoE` class total includes collective wait time (rank-skew
  sensitive).

## 8. Nested timer / overhead risks

Mapping to the `fA`/`fB`/`fC` example (fC = big region, fB loops inside fC, fA
timed many times inside fB, fA overhead lands inside fC):

**Profiler #1 is deeply nested and inclusive-only.** The literal nesting is:

```
step_total                              (fC, outer)
└─ forward                              (contains a full model forward)
   └─ (per layer) L{ID}.attn / L{ID}.ffn   (fB, ×61 layers)
      └─ L{ID}.mla.scores / .moe.experts / …  (fA, timed many times)
```

- **Inclusive, not exclusive.** Every parent key's `total_s` already contains
  all of its children's time *plus* the children's timer overhead. There is no
  exclusive (self) time computed anywhere.
- **No timer-overhead calibration.** `_TimedBlock` does two `perf_counter()`
  calls + a dict `setdefault`/`append` per region. With ~1,400 regions per
  forward and (e.g.) hundreds of decode steps, that's hundreds of thousands of
  context-manager enter/exits whose cost is silently folded into the enclosing
  `L{ID}.*`, `forward`, and `step_total` numbers. The deepest, most-frequently
  entered keys (e.g. `L{ID}.moe.experts`, the per-op MLA blocks) carry the most
  accumulated overhead.
- **Perturbation.** Because every sub-op is wrapped, the absolute times are
  inflated vs. an uninstrumented run; only the *relative* ranking is
  trustworthy, and even that is biased toward regions entered most often.
- **The printed "component-sum" is double-counted.** `print_report` computes a
  phase total as `sum(total_s for keys whose post-phase name has no '.')`. That
  set includes `step_total`, `forward`, `sampling`, `embedding`, `final_norm`,
  `lm_head`, `logits_allgather` — but `step_total ⊇ forward ⊇ embedding …`, so
  the sum massively over-counts. It is labeled "≈" but is not a real breakdown.

**Profiler #2 is shallower but still nested.** Block wraps MLA/MoE which wrap
Linear/Expert/RMSNorm, and all of those are wrapped classes. So the `Block`
total includes `MLA`+`MoE` totals which include `Linear` totals — again
inclusive, again double-counted in the table's `%total` column (the `grand`
total sums all classes including parents). Its mitigations: a warmup forward
before timing, and per-class (not per-region) granularity → far fewer wrap
sites than profiler #1, so lower overhead. But the table cannot be read as an
exclusive breakdown.

## 9. Known problems / limitations

- **No memory profiling at all.** Biggest gap. Memory was the actual failure
  mode (repeated OOMs); the profilers can't see it.
- **Inclusive-only timing, no self/exclusive time.** Both profilers
  double-count parent vs child. The headline "phase total" in profiler #1 is
  inflated.
- **No timer-overhead calibration / subtraction.** ~1,400 wrapped regions/
  forward in profiler #1; overhead is invisible and lands in parents.
- **Output volume.** Profiler #1 emits 61 layers × ~25 keys × 3 phases of
  stats per rank — large JSON, no aggregation, manual analysis.
- **Rank race history.** Shared-path writes corrupted JSON; fixed by
  rank-suffix but only defensively (no barrier, no atomic write/rename).
- **Two unrelated profilers** with different output formats (JSON vs stdout
  table), different region models (per-op vs per-class), different launchers,
  and different runtime paths (generate_cpu-derived vs verify_cpu/dp2_epon-
  derived). No shared core.
- **Monkey-patch fragility.** Both re-implement large chunks of the model
  `forward` (profiler #1 literally re-writes MLA/Indexer/MoE forward to insert
  timers) — they will silently drift from upstream `model.py` if it changes,
  and can introduce correctness differences vs the real path.
- **Hardcoded paths** in launchers (`/home/akashpt/...`, `/scratch/...`,
  `artifacts/deepseek-v3.2-mp2-active`), `MASTER_PORT` fixed at 29500 (collision
  risk), `--time=24:00:00`.
- **Unclear phase boundary.** `is_prefill = prev_pos == 0 and cur_pos > 1`
  classifies the first multi-token step as prefill; semantics get subtle with
  `prefill_chunk_size` (per-chunk counts mix into "prefill" keys).
- **No correctness guard.** Profiler #1 uses temperature/top-p sampling
  (non-greedy) and the chat template, so it is not token-exact-comparable to
  the verify reference; it's a perf tool only.

## 10. Good ideas to reuse

- **Phase awareness** (load / prefill / decode) as a first-class dimension of
  every timing key.
- **`{total, count, mean, min, max}` per key** is a sensible summary shape.
- **Per-rank output files** with a rank suffix (avoids the shared-path race).
- **Checkpoint dumps** during long decode so a timeout still yields partial
  data.
- **Rich run metadata** in the JSON (lin/lout/batch/threads/world_size/host/
  job id/tok-per-s) — makes results self-describing.
- **Warmup-before-timing** (profiler #2) to exclude first-call/JIT/allocator
  effects.
- **Context-manager timer ergonomics** (`with profiler.timed(name):`).
- **Coarse "per-class, sorted by total" table** (profiler #2) is a great
  first-look "what dominates" view — cheap and readable.

## 11. Things to redesign (do not copy directly)

- The **inclusive-only, double-counted** time model → add exclusive/self time.
- **~1,400 wrap sites per forward** with no overhead accounting → make
  granularity configurable and calibrate timer cost.
- **Re-implementing model forward to insert timers** → prefer non-invasive
  hooks or a single coarse boundary set that doesn't fork the math.
- **The "component-sum ≈" phase total** → replace with a real
  inclusive+exclusive breakdown.
- **Two separate profilers / two output formats** → one core, one schema.
- **Hardcoded launcher paths and fixed master port** → derive from config like
  the clean lane already does (`submit_experiment.sh`, `run_native_distributed.sh`).
- **No memory** → add memory sampling.
- **Sampling (non-greedy) in the profiled loop** → keep the profiler on the
  same greedy path as `native_run.py` so perf numbers correspond to the
  validated run.

## 12. Proposed clean-profiler requirements (NOT implemented yet)

- **Low, measured overhead.** Calibrate per-timer cost once at startup and
  report it; allow disabling fine timers entirely.
- **Coarse-first design.** Default granularity = phase + a handful of coarse
  regions (embedding / attn / ffn / norm+head / sampling). Fine per-op timing
  is opt-in.
- **Inclusive AND exclusive time.** Compute self-time per region so a parent's
  number isn't conflated with its children.
- **Timer-overhead reporting/subtraction** for nested regions.
- **Rank-local profile files**, atomic write (temp + rename), rank-suffixed;
  no shared-path writes; optional barrier before the final dump.
- **A profile summary** (single rank-0 rollup that references the per-rank
  files) rather than forcing manual diffing.
- **Memory snapshots** — at minimum RSS at phase boundaries (load done,
  prefill done, per-N decode steps), ideally peak RSS; cheap `/proc/self/status`
  read or `resource.getrusage`.
- **Configurable granularity** via the existing schema fields
  (`TIME_PROFILE`, `MEM_PROFILE`, `PROFILE_GRANULARITY=off|coarse|detailed`),
  so profiling is a config switch, not a separate entrypoint.
- **No correctness changes.** Same greedy decode, same dtype, same cache path
  as `native_run.py`; profiling must not alter generated tokens.
- **Reuse the clean run path.** Integrate into `native_run.py` /
  `run_native_distributed.sh` rather than a forked `generate_profiled.py`.

## 13. Open questions for Akash

1. **Integrate vs. separate entrypoint?** Should profiling be a
   `PROFILE_GRANULARITY` switch inside `native_run.py` (preferred per the schema
   that already exists), or a parallel `native_profile.py`?
2. **Granularity default.** Is coarse (phase + ~5 regions) the right default,
   with `detailed` (per-layer/per-op) opt-in? Or do we always want per-layer?
3. **Memory scope.** Is per-phase RSS + peak RSS enough, or do we need
   per-component memory attribution (much harder, likely needs allocator hooks)?
4. **Hook strategy.** Acceptable to use `nn.Module.register_forward_hook`
   (non-invasive, but only wraps whole-module forward, not sub-ops), or do we
   accept re-implemented forwards for sub-op timing in `detailed` mode only?
5. **Overhead budget.** What overhead is tolerable for the default coarse mode
   (e.g. <2% wall)? That bounds how fine we can go without calibration.
6. **Output location.** Keep profile JSON under
   `results_clean/results/<TAG>/` next to the run results, or a dedicated
   `results_clean/profiles/<TAG>/`?
7. **Aggregation.** Do we want an automatic rank-0 rollup/summary file, and
   should it live in the runner or be a separate post-processing script?
8. **Token-exactness.** Confirm profiling must stay on the greedy,
   cache-backed path (so numbers map to TPCHECKREAL-style runs) rather than the
   legacy sampling path.
