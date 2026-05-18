# Current Inventory

Purpose: Track what exists in the current `../structured_cpu_run` project before any clean migration code is copied.

This inventory is documentation only. It describes the current working CPU inference lane and the surrounding files that should be handled carefully during the clean migration.

## 1. Current Active CPU Inference Path

The current source of truth for CPU verification and benchmark execution is:

- `../structured_cpu_run/verification/run_milestone_bench.sbatch`
- `../structured_cpu_run/verification/verify_cpu.py`
- `../structured_cpu_run/verification/bench_cpu.py`
- `../structured_cpu_run/without_vllm/overrides/`
- `../DeepSeek-V3.2/inference/model.py`

The clean lane should treat these files as inventory sources only for now. No working code has been copied into `structured_cpu_run_clean` yet.

## 2. Current Execution Flow

Current milestone verification and benchmark flow:

```text
sbatch
  -> srun
  -> torch.distributed.run
  -> verify_cpu.py or bench_cpu.py
  -> sys.path override insertion
  -> Transformer construction
  -> weight loading
  -> optional EP pruning / monkey patches / dequant / FAST_LINEAR
  -> prefill
  -> decode
  -> verify tokens or print timing
```

Important details:

- `run_milestone_bench.sbatch` owns the cluster launch shape, mode selection, thread settings, checkpoint paths, BF16 index paths, and verify/bench phase selection.
- `verify_cpu.py` is the token-exact correctness path against a stored GPU reference.
- `bench_cpu.py` mirrors most of the setup logic but uses synthetic prompts and prints timing instead of checking correctness.
- The CPU path inserts `../structured_cpu_run/without_vllm/overrides` before the upstream DeepSeek inference code on `sys.path`.
- `../DeepSeek-V3.2/inference/model.py` remains the model implementation being patched and driven.

## 3. Important Current Files

- `../structured_cpu_run/verification/run_milestone_bench.sbatch`: current canonical multi-mode milestone launcher for `tp2`, `dp2`, and `dp2_epon` verify/bench runs.
- `../structured_cpu_run/verification/verify_cpu.py`: CPU correctness runner; loads the model, applies CPU overrides, runs greedy prefill/decode, and compares generated token IDs to the GPU reference.
- `../structured_cpu_run/verification/bench_cpu.py`: CPU timing runner; shares much of the verify setup but reports TTFT, TPOT, total time, and throughput.
- `../structured_cpu_run/verification/gpu_reference_27160861.json`: current GPU reference prompt and generated token IDs.
- `../structured_cpu_run/verification/capture_gpu_reference.py`: helper used to capture GPU reference output.
- `../structured_cpu_run/configs/*.json`: existing config presets, but they do not fully drive the launch behavior yet.
- `../structured_cpu_run/configs/README.md`: current config notes for the existing presets.
- `../structured_cpu_run/without_vllm/overrides/kernel.py`: CPU fallback for upstream GPU kernel imports.
- `../structured_cpu_run/without_vllm/overrides/dequant_weights.py`: runtime FP8 weight dequantization support.
- `../structured_cpu_run/without_vllm/overrides/run_config.py`: env/config bridge used by current scripts, with incomplete launch authority.
- `../structured_cpu_run/without_vllm/overrides/generate_cpu.py`: older CPU generation entry point; useful reference, but not the current milestone path.
- `../structured_cpu_run/without_vllm/overrides/fast_linear.py`: optional FAST_LINEAR monkey patch for decode GEMV.
- `../structured_cpu_run/without_vllm/overrides/fused_expert.py`: optional fused expert monkey patch.
- `../structured_cpu_run/without_vllm/overrides/batched_moe.py`: experimental batched MoE patch, currently not a first migration target.
- `../structured_cpu_run/without_vllm/overrides/skip_indexer_broadcast.py`: optional Indexer broadcast skip patch.
- `../structured_cpu_run/without_vllm/overrides/skip_kv_fp8_sim.py`: optional KV FP8 simulation skip patch; should remain noncanonical unless explicitly validated.
- `../structured_cpu_run/without_vllm/overrides/kv_cache_dtype.py`: currently a stub/fallback path, not a completed feature.
- `../structured_cpu_run/without_vllm/convert_checkpoint_streaming.py`: checkpoint conversion from HF format to demo shard format.
- `../structured_cpu_run/without_vllm/convert_to_bf16.py`: offline FP8-to-BF16 conversion helper.
- `../structured_cpu_run/without_vllm/convert_checkpoint.sbatch`, `convert_checkpoint_mp1.sbatch`, `convert_to_bf16.sbatch`: existing conversion launchers.
- `../structured_cpu_run/optimisations/profile_cpu.py`: lightweight current profiling helper for the milestone path.
- `../structured_cpu_run/optimisations/run_profile_cpu.sbatch`: launcher for the lightweight profiling helper.
- `../structured_cpu_run/optimisations/amx_probe.py`: AMX/oneDNN probe, not part of the clean migration first copy.
- `../structured_cpu_run/without_vllm/profiling/`: older profiling lane with separate prompt generation and profiling mechanics.
- `../structured_cpu_run/OPTIMISATIONS.md`: current optimization chronology and benchmark notes.
- `../structured_cpu_run/README.md`: useful overview, but parts are stale or internally inconsistent.
- `../DeepSeek-V3.2/inference/model.py`: upstream model structure used by the CPU path, including Transformer, MLA, MoE, Indexer, and parallel linear behavior.

## 4. Legacy, Stale, Or Experimental Files Not To Migrate First

Do not migrate these first:

- `../structured_cpu_run/with_vllm/`: older vLLM CPU attempt; not part of the current milestone path.
- `../structured_cpu_run/without_vllm/run_2node.sbatch`: older baseline launcher.
- `../structured_cpu_run/without_vllm/manual_run_2node.sh`: older manual run path.
- `../structured_cpu_run/without_vllm/submit_validation_run.sh`: older validation helper.
- `../structured_cpu_run/without_vllm/submit_matrix_case.sh`: older matrix helper.
- `../structured_cpu_run/verification/run_verify_cpu.sbatch`: older TP2-specific verifier.
- `../structured_cpu_run/verification/run_verify_cpu_dp2.sbatch`: older DP2-specific verifier.
- `../structured_cpu_run/verification/run_verify_cpu_dp2_epon.sbatch`: older DP2 EP-on-specific verifier.
- `../structured_cpu_run/verification/run_gpu_reference*.sbatch`: GPU reference capture launchers; useful reference, not first clean CPU migration target.
- `../structured_cpu_run/without_vllm/profiling/`: older profiling lane that should be documented before any migration.
- `../structured_cpu_run/optimisations/amx_probe.py`: optimization probe, not framework foundation.
- `../structured_cpu_run/without_vllm/overrides/amx_kernels/`: AMX extension code, not first migration target.
- `../structured_cpu_run/without_vllm/overrides/batched_moe.py`: experimental performance path.
- `../structured_cpu_run/without_vllm/overrides/kv_cache_dtype.py`: incomplete/stub behavior.
- `../structured_cpu_run/without_vllm/overrides/skip_kv_fp8_sim.py`: optional optimization that should not become default without explicit verification.

## 5. What Is Currently Verified

Current historical verification coverage:

- One GPU reference prompt is stored in `../structured_cpu_run/verification/gpu_reference_27160861.json`.
- Correctness is checked by exact generated token ID match.
- `tp2` has historically matched the GPU reference.
- `dp2` has historically matched the GPU reference.
- `dp2_epon` has historically matched the GPU reference.

This verification proves the known prompt path, not broad model equivalence.

## 6. What Is Not Verified Yet

Not yet verified in the clean framework:

- 4 prompts.
- Batch size 4.
- Logits.
- Layer outputs.
- Attention outputs.
- MoE outputs.

These should become explicit future verification targets before the clean lane is treated as stronger than the current milestone script.

## 7. Current Framework Gaps

Known gaps in the current `../structured_cpu_run` framework shape:

- Static sbatch scripts.
- Duplicated model setup and patching logic across verify, bench, and generation paths.
- Configs do not fully drive launch behavior.
- No generated sbatch flow.
- No clean session mode.
- Profiling is split across multiple paths.
- Memory profiling is not standardized.
- Request mode is not explicit.

These gaps are why `structured_cpu_run_clean` should first establish a clearer shape before code is copied.

## 8. What Should Be Copied First Later

When Akash approves a future code-copy step, likely first copy candidates are:

- The minimum verified setup flow from `../structured_cpu_run/verification/verify_cpu.py`.
- The matching benchmark setup flow from `../structured_cpu_run/verification/bench_cpu.py`.
- The CPU override modules required for correctness:
  - `kernel.py`
  - `dequant_weights.py`
  - `run_config.py`
  - `fast_hadamard_transform.py`
- The GPU reference JSON needed for the first clean verification check.
- A small subset of config data after the clean config schema is designed.

Do not copy these during the inventory phase.

## 9. What Should Not Be Touched Right Now

Do not touch these during the current documentation phase:

- `../FlashMLA`
- `../FlashMLA_CPU`
- `../DeepSeek-V3.2`
- `../structured_cpu_run`
- Pipeline parallelism work.
- AMX optimization work.

Do not treat this document as complete yet.
