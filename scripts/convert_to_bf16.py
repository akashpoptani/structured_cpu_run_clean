#!/usr/bin/env python3
"""Offline FP8 -> BF16 dequantization of an mp1/mp2 shard. NOT used at runtime.

Copied verbatim (logic unchanged) from
`../structured_cpu_run/without_vllm/convert_to_bf16.py`. Stage 2 of the weight
pipeline; stage 1 is `scripts/convert_checkpoint.py`:

    models/deepseek-v3.2 (HF FP8, 163 shards)
      -> convert_checkpoint.py --model-parallel {1,2}     [rename + TP slice + expert filter]
           -> artifacts/deepseek-v3.2-mp{1,2}/model{rank}-mp{N}.safetensors   (still FP8)
             -> convert_to_bf16.py                        [block-aware dequant, scales dropped]
                  -> <dst_dir>/model-NNNNN-of-MMMMM.safetensors + model.safetensors.index.json

Reads FP8 weights plus their companion (ceil(out/128), ceil(in/128)) FP32
`.scale` tensors, applies the block-broadcast dequant, and writes BF16. The
scales are DROPPED from the output: once w_bf16 = w_fp8 * scale, the scale has
been folded into the value and BF16's 8 exponent bits need no block rescaling.
All non-FP8 tensors (norms, biases, FP32 Linears, embeddings) pass through
unchanged.

Memory strategy: buffers tensors until `--shard-limit-gb` (default 400) then
flushes a shard and `gc.collect()`s, so peak RSS stays near one shard rather
than the full payload. Shards are written under a temporary
`model-NNNNN-of-XXXXX` name and renamed once the total count is known; a
single-shard run is renamed to the exact `--dst` filename instead.

`--expert-start N --expert-end M` keeps only routed experts in [N, M) — this is
the OFFLINE expert pruning that makes BF16 dp2_epon feasible. Non-expert
tensors and `shared_experts.*` are always kept. Without it, dp2_epon would have
to load the full ~1.34 TB BF16 model and prune in memory.

Relationship to the runtime path: `src/overrides/dequant_weights.py` performs
the same block-broadcast math in-process (DEQUANT_FP8_WEIGHTS=all), and
`src/clean_inference/weight_loading.py` can persist that result as the dequant
cache. This script does it once, offline, so no job pays the ~50 min dequant.
Verified equivalent: the key set produced here is identical (23,428 keys) to
the runtime dequant cache's.

Upstream `fp8_cast_bf16.py` (DeepSeek-V3 repo) is the GPU analogue; it operates
on the original HF layout and needs CUDA, so it cannot run here.
"""
"""Phase 3: offline FP8 -> BF16 conversion of an mp1 or mp2 safetensors shard.

Relationship to upstream `fp8_cast_bf16.py` (DeepSeek-V3 repo,
saved here as fp8_cast_bf16_upstream.py for reference):

  Upstream operates on the ORIGINAL HF multi-shard layout with
  per-shard `weight_scale_inv` tensors and uses CUDA's tilelang
  weight_dequant kernel. Our pipeline runs `convert_checkpoint_streaming.py`
  FIRST to produce a single-file mp1/mp2 shard with renamed `scale`
  tensors — that's where the HF -> mp1 conversion happens.

  This script is the OTHER half — dequant the resulting mp1/mp2
  shard's FP8 weights to BF16. It runs CPU-only with the same block-
  aware dequant math (model.py:493-494) that upstream's kernel does.
  Mathematically equivalent to fp8_cast_bf16.py applied to the HF
  source, then re-sharded via convert_checkpoint_streaming.py with
  the FP8 path replaced by identity.

  We validate by running a token-exact verify after each conversion
  output — if PASS, our converter matches upstream's numerics.

Reads a shard with FP8 weights + companion (out/128, in/128) FP32
scales, applies block-aware dequant, and writes BF16 output. The
scales are DROPPED (no longer needed once weights are BF16). All
non-FP8 tensors are written through as-is.

Why this exists: the runtime FP8 -> BF16 dequant pass in
overrides/dequant_weights.py takes ~50 minutes per launcher invocation.
This script does the conversion ONCE offline; subsequent runs load
BF16 directly and skip the dequant pass.

Memory: streams tensor-by-tensor, accumulates an in-memory dict that
gets flushed to disk in shards of bounded size. Peak memory =
SHARD_LIMIT_GB + one FP32 intermediate. Default SHARD_LIMIT_GB=400 so
that a 1.3 TB total output is split across ~4 shards, each within a
950 GB allocation. Output includes a safetensors index.json that
the launcher uses to find each tensor.

Optional per-rank expert filtering (for dp2_epon): use
--expert-start N --expert-end M to KEEP only routed experts in that
range. All non-expert tensors and shared_experts are always kept.

Usage:
    # Full mp1 conversion (writes multi-shard BF16 + index):
    python convert_to_bf16.py \\
        --src /scratch/.../deepseek-v3.2-mp1/model0-mp1.safetensors \\
        --dst /scratch/.../deepseek-v3.2-mp1-bf16/model0-mp1.safetensors

    # Per-rank dp2_epon shard (half experts only):
    python convert_to_bf16.py \\
        --src .../deepseek-v3.2-mp1/model0-mp1.safetensors \\
        --dst .../deepseek-v3.2-mp1-bf16-dp2_epon-rank0/model0-mp1.safetensors \\
        --expert-start 0 --expert-end 128
"""
import argparse
import gc
import json
import os
import re
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file


def dequant_block_aware(weight_fp8: torch.Tensor, scale_fp32: torch.Tensor,
                        block_size: int = 128) -> torch.Tensor:
    out, ind = weight_fp8.shape
    s_out, s_in = scale_fp32.shape
    pad_out = s_out * block_size - out
    pad_in = s_in * block_size - ind
    w = F.pad(weight_fp8, (0, pad_in, 0, pad_out)) if (pad_out or pad_in) else weight_fp8
    bf = ((w.float()
           .view(s_out, block_size, s_in, block_size)
           * scale_fp32.view(s_out, 1, s_in, 1))
          .to(torch.bfloat16)
          .reshape(s_out * block_size, s_in * block_size))
    return bf[:out, :ind].contiguous() if (pad_out or pad_in) else bf


def expert_index_from_key(key: str):
    """Extract routed-expert index from a key like
    'layers.5.ffn.experts.42.w1.weight'. Returns None if not an expert
    weight, or if it's a shared_experts.* path."""
    if ".experts." not in key or ".shared_experts." in key:
        return None
    m = re.search(r"\.experts\.(\d+)\.", key)
    return int(m.group(1)) if m else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True,
                   help="Final shard name; multi-shard renames to model-XXXXX-of-YYYYY.safetensors")
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--expert-start", type=int, default=None,
                   help="Keep only routed experts in [start, end). All other tensors always kept.")
    p.add_argument("--expert-end", type=int, default=None)
    p.add_argument("--shard-limit-gb", type=int, default=400,
                   help="Max bytes per shard file (in GB). Smaller = more shards, less peak mem.")
    args = p.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    if not src.exists():
        raise SystemExit(f"Source missing: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    SHARD_LIMIT = args.shard_limit_gb * (1 << 30)

    filt_active = args.expert_start is not None and args.expert_end is not None
    print(f"[bf16-convert] src: {src}", flush=True)
    print(f"[bf16-convert] dst: {dst}", flush=True)
    print(f"[bf16-convert] expert filter: "
          f"{f'[{args.expert_start}, {args.expert_end})' if filt_active else 'NONE (keep all)'}",
          flush=True)
    print(f"[bf16-convert] shard limit: {args.shard_limit_gb} GB", flush=True)
    t_start = time.perf_counter()

    with safe_open(str(src), framework="pt", device="cpu") as h:
        all_keys = list(h.keys())
    scale_keys = set(k for k in all_keys if k.endswith(".scale"))
    print(f"[bf16-convert] {len(all_keys)} tensors total, {len(scale_keys)} scales", flush=True)

    # First pass to filter keys we'll write.
    keys_to_write = []
    skipped_expert = 0
    for k in all_keys:
        if k in scale_keys:
            continue  # scales handled with their weight, dropped from output
        if filt_active:
            eidx = expert_index_from_key(k)
            if eidx is not None and not (args.expert_start <= eidx < args.expert_end):
                skipped_expert += 1
                continue
        keys_to_write.append(k)
    print(f"[bf16-convert] will write {len(keys_to_write)} tensors "
          f"(skipped {skipped_expert} expert weights outside [{args.expert_start}, {args.expert_end}))",
          flush=True)

    # Process + buffer + flush shards as we go.
    shard_buf = {}
    shard_bytes = 0
    shard_files = []  # list of (filename, list_of_keys)
    weight_map = {}   # key -> filename
    fp8_count = 0
    other_count = 0
    t_dequant = 0.0

    def flush(final=False):
        nonlocal shard_buf, shard_bytes
        if not shard_buf:
            return
        shard_idx = len(shard_files)
        # Temp name; we rename after we know the total shard count
        fname = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
        out_path = dst.parent / fname
        t_w0 = time.perf_counter()
        save_file(shard_buf, str(out_path))
        keys_in_shard = list(shard_buf.keys())
        shard_files.append((fname, keys_in_shard))
        for k in keys_in_shard:
            weight_map[k] = fname
        print(f"[bf16-convert] flushed shard {shard_idx} -> {fname}, "
              f"{len(keys_in_shard)} tensors, {shard_bytes / (1 << 30):.1f} GB, "
              f"write {time.perf_counter() - t_w0:.0f}s", flush=True)
        shard_buf = {}
        shard_bytes = 0
        gc.collect()

    with safe_open(str(src), framework="pt", device="cpu") as h:
        for k in keys_to_write:
            tensor = h.get_tensor(k)
            if k.endswith(".weight") and tensor.dtype == torch.float8_e4m3fn:
                scale_k = k.replace(".weight", ".scale")
                if scale_k in scale_keys:
                    scale = h.get_tensor(scale_k)
                    t0 = time.perf_counter()
                    out_t = dequant_block_aware(tensor, scale, args.block_size)
                    t_dequant += time.perf_counter() - t0
                else:
                    print(f"[bf16-convert] WARN: FP8 weight {k} has no scale; "
                          f"casting blind", flush=True)
                    out_t = tensor.float().to(torch.bfloat16)
                fp8_count += 1
            else:
                # Already non-FP8 (norm, bias, FP32 Linear, buffer, etc.)
                out_t = tensor
                other_count += 1

            t_bytes = out_t.numel() * out_t.element_size()
            if shard_bytes + t_bytes > SHARD_LIMIT and shard_buf:
                flush()
            shard_buf[k] = out_t
            shard_bytes += t_bytes

            done = fp8_count + other_count
            if done % 1000 == 0:
                elapsed = time.perf_counter() - t_start
                print(f"[bf16-convert] {done}/{len(keys_to_write)} tensors "
                      f"({elapsed:.0f}s elapsed, dequant pure {t_dequant:.0f}s)",
                      flush=True)

    flush(final=True)

    # If only one shard was written, rename to the user's desired single name.
    if len(shard_files) == 1:
        old_path = dst.parent / shard_files[0][0]
        new_path = dst
        os.rename(str(old_path), str(new_path))
        # Update weight_map
        for k in weight_map:
            weight_map[k] = dst.name
        shard_files = [(dst.name, shard_files[0][1])]
        print(f"[bf16-convert] single shard renamed -> {dst.name}", flush=True)
    else:
        # Multi-shard: rename "model-N-of-XXXXX.safetensors" -> "...of-{total}..."
        total = len(shard_files)
        new_shard_files = []
        new_weight_map = {}
        for idx, (fname, keys) in enumerate(shard_files):
            new_fname = f"model-{idx:05d}-of-{total:05d}.safetensors"
            os.rename(str(dst.parent / fname), str(dst.parent / new_fname))
            for k in keys:
                new_weight_map[k] = new_fname
            new_shard_files.append((new_fname, keys))
        shard_files = new_shard_files
        weight_map = new_weight_map
        print(f"[bf16-convert] {total} shards renamed to model-NNNNN-of-{total:05d} pattern",
              flush=True)

    # Write index.json (HF-style)
    total_size = sum(os.path.getsize(dst.parent / fn) for fn, _ in shard_files)
    index_path = dst.parent / "model.safetensors.index.json"
    with open(index_path, "w") as f:
        json.dump({"metadata": {"total_size": total_size},
                   "weight_map": weight_map},
                  f, indent=2)
    print(f"[bf16-convert] wrote {index_path}", flush=True)

    # Copy aux files (tokenizer + configs)
    src_dir = src.parent
    dst_dir = dst.parent
    for aux in ("config.json", "tokenizer.json", "tokenizer_config.json",
                "generation_config.json"):
        sp = src_dir / aux
        dp = dst_dir / aux
        if sp.exists() and not dp.exists():
            shutil.copy2(sp, dp)
            print(f"[bf16-convert] copied {aux}", flush=True)

    print(f"[bf16-convert] DONE. {fp8_count} FP8->BF16, {other_count} other. "
          f"Total: {time.perf_counter() - t_start:.0f}s ({t_dequant:.0f}s pure dequant). "
          f"Wrote {len(shard_files)} shard(s), {total_size / (1 << 30):.1f} GB total.",
          flush=True)


if __name__ == "__main__":
    main()
