#!/usr/bin/env python3
"""Offline HF -> native TP-sharded checkpoint conversion. NOT used at runtime.

Copied verbatim (logic unchanged) from
`../structured_cpu_run/without_vllm/convert_checkpoint_streaming.py`, which is
itself a fork of upstream `../DeepSeek-V3.2/inference/convert.py`. This file is
the provenance of `SHARDED_CKPT_PATH` — run it once, then every experiment
consumes its output as a data path.

What it does to each tensor (three steps; dtype is NEVER touched — FP8 in,
FP8 out):

  1. Rename HF -> native. `model.layers.0.self_attn.q_a_proj.weight` becomes
     `layers.0.attn.wq_a.weight` via self_attn->attn, mlp->ffn,
     weight_scale_inv->scale, e_score_correction_bias->bias, plus MAPPING.
  2. Slice by the dim recorded in MAPPING:
       dim 0    -> ColumnParallel, sliced on out_features (wq, wq_b, wkv_b,
                   w1, w3, embed, head)
       dim 1    -> RowParallel, sliced on in_features (wo, w2)
       dim None -> replicated in every rank file (norms, gate, wkv_a, and the
                   indexer's wq_b / wk / k_norm / weights_proj)
  3. Filter experts by rank: n_local_experts = n_experts // mp, so rank r keeps
     experts [r*n_local, (r+1)*n_local). Also drops layer 61 (the MTP head,
     unused for inference).

Why "streaming": upstream builds all `mp` rank dicts at once
(`state_dicts = [{} for _ in range(mp)]`), holding every shard in RAM
simultaneously. This version builds and writes one rank at a time, re-reading
the HF checkpoint per rank — more I/O, roughly half the peak RSS. That is what
makes mp=2 fit in ~900G.

The load-time invariant this establishes: output keys match
`Transformer.state_dict()` exactly, and the sharding lives in tensor SHAPES,
not names. Both rank files therefore have identical key sets with different
shapes, which is why `src/clean_inference/weight_loading.py` can call
`safetensors.torch.load_model(model, shard_path, strict=False)` with no key
remapping — and why `dist.init_process_group` must run BEFORE
`Transformer(args)` so `world_size` is baked into the parallel layers.

Note: the indexer's weights carry dim None, i.e. they are REPLICATED, not
sharded. All `index_n_heads` (64) live on every rank. See README.
"""
import gc
import os
from argparse import ArgumentParser
from glob import glob

import torch
from safetensors.torch import safe_open, save_file
from tqdm import tqdm


MAPPING = {
    "embed_tokens": ("embed", 0),
    "input_layernorm": ("attn_norm", None),
    "post_attention_layernorm": ("ffn_norm", None),
    "q_proj": ("wq", 0),
    "q_a_proj": ("wq_a", None),
    "q_a_layernorm": ("q_norm", None),
    "q_b_proj": ("wq_b", 0),
    "kv_a_proj_with_mqa": ("wkv_a", None),
    "kv_a_layernorm": ("kv_norm", None),
    "kv_b_proj": ("wkv_b", 0),
    "o_proj": ("wo", 1),
    "gate": ("gate", None),
    "gate_proj": ("w1", 0),
    "down_proj": ("w2", 1),
    "up_proj": ("w3", 0),
    "norm": ("norm", None),
    "lm_head": ("head", 0),
    "scale": ("scale", None),
    "wq_b": ("wq_b", None),
    "wk": ("wk", None),
    "k_norm": ("k_norm", None),
    "weights_proj": ("weights_proj", None),
}


def remap_name(name: str):
    if name.startswith("model."):
        name = name[len("model."):]
    name = name.replace("self_attn", "attn")
    name = name.replace("mlp", "ffn")
    name = name.replace("weight_scale_inv", "scale")
    name = name.replace("e_score_correction_bias", "bias")
    key = name.split(".")[-2]
    if key not in MAPPING:
        raise KeyError(f"Key {key} not found in mapping for tensor {name}")
    new_key, dim = MAPPING[key]
    return name.replace(key, new_key), dim


def build_rank_state_dict(hf_ckpt_path: str, n_experts: int, mp: int, rank: int):
    shard_paths = sorted(glob(os.path.join(hf_ckpt_path, "*.safetensors")))
    if not shard_paths:
        raise FileNotFoundError(f"No safetensors found in {hf_ckpt_path}")

    n_local_experts = n_experts // mp
    state_dict = {}

    for file_path in tqdm(shard_paths, desc=f"rank{rank}_read", unit="shard"):
        with safe_open(file_path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if "model.layers.61" in name:
                    continue

                param = handle.get_tensor(name)
                remapped_name, dim = remap_name(name)
                new_param = param

                if "experts" in remapped_name and "shared_experts" not in remapped_name:
                    idx = int(remapped_name.split(".")[-3])
                    if idx < rank * n_local_experts or idx >= (rank + 1) * n_local_experts:
                        continue
                elif dim is not None:
                    if param.size(dim) % mp != 0:
                        raise ValueError(f"Dimension {dim} for {remapped_name} is not divisible by {mp}")
                    shard_size = param.size(dim) // mp
                    new_param = param.narrow(dim, rank * shard_size, shard_size).contiguous()

                state_dict[remapped_name] = new_param

    return state_dict


def main(hf_ckpt_path: str, save_path: str, n_experts: int, mp: int):
    torch.set_num_threads(8)
    os.makedirs(save_path, exist_ok=True)

    for rank in range(mp):
        state_dict = build_rank_state_dict(hf_ckpt_path, n_experts, mp, rank)
        out_path = os.path.join(save_path, f"model{rank}-mp{mp}.safetensors")
        save_file(state_dict, out_path)
        del state_dict
        gc.collect()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hf-ckpt-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--n-experts", type=int, required=True)
    parser.add_argument("--model-parallel", type=int, required=True)
    args = parser.parse_args()
    if args.n_experts % args.model_parallel != 0:
        raise SystemExit("Number of experts must be divisible by model parallelism")
    main(args.hf_ckpt_path, args.save_path, args.n_experts, args.model_parallel)
