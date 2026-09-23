# DP_IMPLEMENTATION_PLAN.md — DP2_EPON in the clean lane

Status: **IMPLEMENTED and token-exact VERIFIED 2026-09-22** (job 27750972).
Reference: `../structured_cpu_run/without_vllm/overrides/generate_cpu.py:225-365`.

## Result

| Check | Outcome |
|---|---|
| Construct-only (27750971) | PASS — 671,877,944,064 params = FULL size, proving `world_size==1` at construction |
| dist init ordering | PASS — deferred, then `init_process_group('gloo') AFTER construction`, `dist ready: rank=0/2` |
| Weight load | `loaded=23428 missing=22517 unexpected=0 shape_mismatch=0` |
| Missing-key accounting | **exact**: 58 MoE layers x 128 non-local experts x 3 matrices = 22,272, + 245 KV/buffer keys = 22,517 |
| EP install | 58 MoE layers, 7,424 non-local Expert modules pruned, range [0, 128) of 256 |
| **Token-exact verify** | **PASS** vs `case_0001.json`, first mismatch index = -1 |
| Peak RSS | 750.0 GiB / rank; wall 17 min (928 s load + 47 s verify) |

`shared_experts` placement (delta B') was correct first time — had it been wrong, the tokens would
have diverged silently rather than crashed.

## 0. Scope

**DP2_EPON, BF16 only.** `dp2` (EP-off) is **out of scope**: ~1.34 TB/rank in BF16 is infeasible, and
FP8-only `dp2` would need the `dense` dequant scope that was removed from the clean lane.
FP8 `dp2_epon` is also out of scope — no FP8 expert-filtering converter exists (`convert_to_bf16.py`
always dequantizes), so it would need a new script plus 3-6 h of I/O per variant, to run 18x slower.

| Mode | Attention / dense | Experts | Comm per MoE layer |
|---|---|---|---|
| `tp2` (working) | sharded | sharded | all_reduce in MoE **and** every RowParallelLinear (`wo`, `w2`) |
| `dp2_epon` (this plan) | **replicated** | 128 of 256 per rank | one all_reduce in MoE only |

## 1. Four structural inversions vs TP2 (each is a correctness trap)

1. **dist init moves after construction.** TP2 needs `world_size=2` *before* `Transformer(args)` so the parallel layers bake in shard shapes. DP needs `world_size=1` at construction so layers are built **full size**.
2. **`model.world_size` stays 1 for the whole run.** Upstream's `if world_size > 1` guards never fire: MoE's `all_reduce` is skipped (must be re-added) and the logits `all_gather` is skipped (**correct** — each DP rank has full vocab).
3. **`shared_experts` moves after the all_reduce** (see §4).
4. **`experts_start_idx`/`experts_end_idx` are wrong.** Upstream sets them in `__init__` from `rank * (n_routed_experts // world_size)`, but with `world_size==1` that is **0-255 on both ranks**. We must set our own from the real rank/world size.

## 2. Weights — generation commands

```bash
A=/scratch/.../DeepSeekRun_runtime/artifacts
# stage 1 (already done): HF FP8 -> unsharded native FP8, rename only, no slicing
.venv/bin/python scripts/convert_checkpoint.py \
    --hf-ckpt-path /scratch/.../models/deepseek-v3.2 \
    --save-path ${A}/deepseek-v3.2-mp1 --n-experts 256 --model-parallel 1
# stage 2 (already done): FP8 -> BF16 with OFFLINE expert pruning, per rank
for RANK in 0 1; do
  .venv/bin/python scripts/convert_to_bf16.py \
      --src ${A}/deepseek-v3.2-mp1/model0-mp1.safetensors \
      --dst ${A}/deepseek-v3.2-bf16-dp2_epon-rank${RANK}/model0-mp1.safetensors \
      --expert-start $((RANK*128)) --expert-end $(((RANK+1)*128))
done
```

| Artifact | Size | Status |
|---|---|---|
| `deepseek-v3.2-mp1/model0-mp1.safetensors` | 674.1 GB / 627.8 GiB, **FP8** | exists |
| `deepseek-v3.2-bf16-dp2_epon-rank{0,1}/` | 689.9 GB / **642.5 GiB** each, BF16 + `index.json` | exists |

**Offline vs runtime pruning.** *Offline* = `convert_to_bf16.py --expert-start/--expert-end` never writes
non-local experts, so they are never allocated. *Runtime* = load the full model then
`experts[i] = None` + `gc.collect()`, which needs the whole model resident first (~1.34 TB in BF16 —
impossible). **Offline is the only viable route**, and those artifacts already exist.

## 3. Weight loading — already done

`BF16_CKPT_INDEX` (added 2026-09-21, validated by the `TP2BF16` run) does exactly what this needs:
resolves `{RANK}`, forces `ModelArgs.dtype="bf16"` before construction, streams shards via the
existing reader, skips dequant-cache metadata validation. **No new loader work.**

```
BF16_CKPT_INDEX=".../deepseek-v3.2-bf16-dp2_epon-rank{RANK}/model.safetensors.index.json"
DEQUANT_CACHE_MODE="off"
```
Expect ~62 `dtype mismatch` warnings (`head.weight` + 61 × `indexer.weights_proj.weight`): upstream
holds these as FP32 params while the checkpoint stores BF16. The loader casts on copy — benign.

## 4. `src/overrides/ep_moe.py` — exact override

**CURRENT** (upstream `model.py:790-800`):
```python
    for i in range(self.experts_start_idx, self.experts_end_idx):   # (A) 0-255 on BOTH ranks in DP
        if counts[i] == 0: continue
        idx, top = torch.where(indices == i)
        y[idx] += self.experts[i](x[idx]) * weights[idx, top, None]
    y += self.shared_experts(x)                                     # (B) BEFORE reduce
    if world_size > 1:                                              # (C) never true in DP
        dist.all_reduce(y)
```

**PROPOSED** (bound with `types.MethodType(_ep_moe_forward, layer.ffn)`):
```python
    for i in range(self.ep_start_idx, self.ep_end_idx):             # (A') our EP range
        if counts[i] == 0: continue
        idx, top = torch.where(indices == i)
        y[idx] += self.experts[i](x_flat[idx]) * weights[idx, top, None]
    if _reduce_dtype is torch.bfloat16:                             # (C') UNCONDITIONAL
        y_comm = y.to(torch.bfloat16); dist.all_reduce(y_comm); y = y_comm.to(torch.float32)
    else:
        dist.all_reduce(y)
    y += self.shared_experts(x_flat)                                # (B') AFTER reduce
```

Why each delta:
- **(A')** upstream's indices are 0-255 on both ranks because `world_size==1` at construction → every rank would run every expert. We set `ep_start_idx = rank * 128`, `ep_end_idx = ep_start_idx + 128`.
- **(C')** the `world_size > 1` guard never fires, so without this each rank keeps only its own experts' partial sum. `MOE_REDUCE_DTYPE=bf16` halves the payload; legacy measured all_reduce at **16.6% of decode**.
- **(B')** **the trap.** In TP, `shared_experts` is sharded (`reduce_output=False`) so a partial value must be added *before* the reduce. In DP every rank holds the **full** copy — adding before would **double-count it**.

Install pass, run once after dist init:
```python
layer.ffn.ep_start_idx = rank * (n_routed_experts // world_size)
layer.ffn.ep_end_idx   = layer.ffn.ep_start_idx + (n_routed_experts // world_size)
for i in range(n_routed_experts):                    # REQUIRED even with pre-pruned weights:
    if not (ep_start <= i < ep_end):                 # the modules exist (constructed) but were
        layer.ffn.experts[i] = None                  # never loaded -> uninitialized memory
layer.ffn.forward = types.MethodType(_ep_moe_forward, layer.ffn)
```

## 5. dist-init refactor (agreed)

Replace `initialize_distributed_if_needed()` — whose "caller must init AFTER construction" log
describes an obligation the code cannot enforce, so a `dp2_epon` run today would silently behave as
single-rank — with:

```python
def initialize_distributed_before_construction(config, dist_env, log_fn) -> bool:  # TP modes
def initialize_distributed_after_construction(config, dist_env, log_fn)  -> bool:  # DP modes
def assert_distributed_ready(dist_env) -> None:  # fail loudly if world_size>1 and no process group
```

The assertion is what makes two functions safer than one: "forgot to call it" becomes a hard error.
Call `assert_distributed_ready()` in `native_run.py` immediately after the post-construction hook.

## 6. Remaining code changes

1. `native_runtime.py` — the three functions above.
2. `native_run.py` — call the after-construction hook + assertion post-`construct_transformer()`; call the ep_moe install pass.
3. **New** `src/overrides/ep_moe.py` — §4.
4. `parse_config.sh` — `SHARDING_MODE` is **not** enum-validated today (a typo silently falls through). Add `require_one_of SHARDING_MODE tp2 dp2_epon`.

## 7. Validation order

1. `--no-load-weights`: assert `wq_b.weight.shape[0]` is **full** size (proves `world_size==1` at construction).
2. `--no-generate`: load the pre-pruned BF16 shard; `missing_keys` should be KV buffers + non-local experts only.
3. **Token-exact verify** against `verification/references/prompt1_bs1_lin10_lout15/case_0001.json` — must match TP2 exactly. A mismatch is almost certainly (B'): double-counted `shared_experts`.
4. Bench vs TP2 at Lin=2048/Lout=2048, 32 threads, `OMP_PROC_BIND=false`.

## 8. Risks

- The pre-pruned BF16 EP artifacts were built by the legacy lane and have **never been validated here**. Step 7.3 is the gate.
- DP2_EPON trades **replicated attention compute** (2x wasted FLOPs) for **less communication** (attention fully local vs TP2's all_reduce in every `wo`/`w2`). Net direction unknown on our hardware — measure, do not assume.
- Legacy ran `dp2_epon` at **48 threads**; our TP2 optimum is 32. Re-sweep; the comm/compute balance differs.
