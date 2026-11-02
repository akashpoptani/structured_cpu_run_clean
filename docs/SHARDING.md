# Sharding modes: what actually runs

`SHARDING_MODE` accepts `tp2`, `dp2_epon`, `dp2_true` (`parse_config.sh:176`).
Only **tp2** has a clear path to large batch at long context. The DP modes are
constrained by *when* expert pruning happens relative to weight loading.

## tp2

Weights are sharded on disk (`model<rank>-mp2.safetensors`), so each rank loads
~600 GB and never materialises the whole model. This is the only mode with a
bf16 dequant cache built
(`/scratch/.../models/deepseek-v3.2-tp2-bf16-dequant-all`, 674.7 GB/rank,
`world_size: 2`), which is why its weight load is ~865 s against the DP modes'
on-the-fly dequant.

## dp2_epon / dp2_true

Both load the **unsharded** `deepseek-v3.2-mp1` checkpoint and build every
parallel layer at full size; `install_ep_moe` / `install_true_dp_moe` prune
non-local experts **after** the load. Measured consequence (job 27933420):

```
[native-run] total parameters (numel sum): 671,877,944,064
error: Detected 1 oom_kill event in StepId=27933420.1
```

Per rank that is ~704 GB bf16 (shared + half the routed experts) **plus** KV.
At Lin=120k the KV is 9.4 GB/sequence, so:

| BS | KV/rank | total/rank | fits 820 G? |
|---:|--------:|-----------:|:------------|
| 4  |   38 GB |    ~742 GB | yes |
| 8  |   75 GB |    ~779 GB | marginal |
| 16 |  150 GB |    ~854 GB | **no — OOM killed** |

So DP reaches BS=4 at 120k where tp2 reaches 16. **DP's theoretical 2x KV
advantage (each rank owning BS/2 sequences) is not realised**, for two separate
reasons: the model is built with `max_batch_size = BATCH_SIZE` on every rank
rather than BS/2, and the unpruned load dominates memory anyway. Both are fixable;
neither is fixed today.

### Traps

**The tp2 dequant cache is unreadable under DP.** It carries no
`sharding_mode` metadata, so the loader treats it as legacy and refuses it:
`legacy cache without sharding_mode metadata can only be read under
SHARDING_MODE=tp2`. No DP cache exists. Set `DEQUANT_CACHE_MODE="off"`,
`DEQUANT_CACHE_PATH=""`, `WEIGHTS_PRECISION="bf16"`,
`DEQUANT_FP8_WEIGHTS="none"`, `SHARDED_CKPT_PATH=.../deepseek-v3.2-mp1`.

**`dp2_true` was absent from the weight loader** until 2026-10-19:
`resolve_shard_path` listed only `("dp2", "dp2_epon")`, so every `dp2_true`
run died with `unsupported SHARDING_MODE for weight loading: 'dp2_true'`.
It loads identically to `dp2_epon`; the modes differ only in which MoE forward
is bound after the load.

**`dp2_true` at BS=1 deadlocks.** `install_true_dp_moe` binds its forward from
`BATCH_SIZE` while `_decode_case_group_sharded` shards on `len(cases) > 1`.
At BS=1 those disagree and the ranks hang. `BS_SWEEP` skips BS=1 under
`dp2_true` for this reason.

## Launching (all modes)

The master's ib0 address is resolved **once** in the generated sbatch and
exported to every rank. It used to be resolved per-rank inside the srun step
via a nested `srun`; under contention one rank's lookup fails, and because the
assignment runs under `set -euo pipefail` that rank dies before reaching its
own fallback. Jobs 27933115/27933116/27933407 all died this way, and one run
was caught with rank 0 on `MASTER_ADDR=10.164.33.160` (eth0) and rank 1 on
`10.164.44.20` (ib0) -- a guaranteed rendezvous hang with no error message.
The signature to watch for is two `GLOO_SOCKET_IFNAME=` log lines that disagree.

## Measured 2026-10-19 (and why the DP numbers are not comparable)

| mode | BS | ctx | TPOT | tok/s |
|---|---:|---:|---:|---:|
| tp2 (bf16 weights) | 1 | 120k | 2,158 ms | 0.463 |
| dp2_epon (fp8 weights) | 1 | 120k | 78,227 ms | 0.013 |
| dp2_true (fp8 weights) | 2 | 120k | 79,339 ms | 0.025 |

**Do not read this as a sharding comparison.** Two separate confounds:

1. **Weight precision, not sharding.** DP only fits at fp8
   (`DEQUANT_FP8_WEIGHTS="none"`), so every GEMM dequantises at call time,
   every layer, every step. tp2 reads a pre-dequantised bf16 cache. That is
   the ~36x gap. DP at bf16 would be comparable but needs ~1.34 TB/rank.

2. **The tiled path does not shard the batch.** `ctx_tile.timed_decode` calls
   `transformer.forward()` with the full batch, bypassing
   `_decode_case_group_sharded`. Under `dp2_true` that is simply wrong: job
   27933614 crashed at BS=4 with "expanded size of the tensor (2) must match
   the existing size (4)", and at BS=2 the shapes lined up by luck and
   produced a number that was NOT measuring true DP.

A first reading of the BS=2 result -- same TPOT as dp2_epon BS=1, so "batch
sharding works" -- was wrong. Flat TPOT across batch is fully explained by
fp8 dequant being per-step and batch-independent: the same weights are
dequantised whether one sequence is decoded or two. No sharding effect is
needed to explain it, and per (2) none was present.

To get a real DP vs TP comparison: prune experts at construction (so bf16
fits), and shard the batch inside `timed_decode` for `dp2_true`.
