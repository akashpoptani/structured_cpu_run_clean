"""Generate the TP/DP x batch-size run matrix for long-context TPOT.

Memory model (measured, not assumed):
  weights  1.2 TB total bf16 on disk -> ~600 GB/rank  (du of the dequant cache)
  node     1003 GB usable            -> ~403 GB/rank for KV
  KV/seq   61 layers x (kv_lora_rank 512 + pe 64) bf16 + indexer caches
           = 78,324 B/token -> 10.27 GB at seq=131072

  TP2: kv_cache is NOT sharded (the latent is shared across heads), so every
       rank holds the whole batch -> max BS ~= 39
  DP2: each rank owns BS/2 sequences                 -> max BS ~= 78

That asymmetry is the point of the sweep: at long context the KV cache, not the
weights, is what runs you out of memory, so DP doubles the batch ceiling.

TPOT is per-step, so LOUT is small by design: over 8k output tokens the context
grows only 120k->128k (6.7%), and a full Lout=8192 run is ~4.3 h of decode per
cell -- days across the matrix.
"""
import argparse, os

BYTES_PER_TOKEN = 78_324          # per sequence, all 61 layers
WEIGHTS_GB_PER_RANK = 600
NODE_GB = 1003

def kv_gb(bs, seq, sharded):
    per_rank = bs / 2 if sharded else bs
    return per_rank * seq * BYTES_PER_TOKEN / 1e9

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lin", type=int, default=120000)
    ap.add_argument("--lout", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--bs", default="1,4,16,32,64")
    ap.add_argument("--hours", type=int, default=8)
    ap.add_argument("--emit", action="store_true", help="write the .env files")
    a = ap.parse_args()

    seq = a.lin + a.lout
    here = os.path.dirname(os.path.abspath(__file__))
    tmpl = open(os.path.join(here, "configs",
        "IDXDUMP_generate_tp2_lin4096_lout2_bs1_n2_c32_mem700g_tprof0_mprof0.env")).read()

    print(f"seq={seq}  KV={seq*BYTES_PER_TOKEN/1e9:.2f} GB/sequence  "
          f"budget={NODE_GB-WEIGHTS_GB_PER_RANK} GB/rank\n")
    print(f"{'mode':10s} {'BS':>4s} {'KV/rank GB':>11s} {'total/rank':>11s} {'fits':>6s}")
    made = []
    for mode, shard in (("tp2", False), ("dp2_epon", True)):
        for bs in [int(x) for x in a.bs.split(",")]:
            k = kv_gb(bs, seq, shard)
            tot = k + WEIGHTS_GB_PER_RANK
            fits = tot < NODE_GB
            print(f"{mode:10s} {bs:4d} {k:11.1f} {tot:11.1f} {'yes' if fits else 'NO':>6s}")
            if not fits or not a.emit:
                continue
            tag = f"M{mode.split('_')[0].upper()}B{bs}"
            cfg = tmpl
            for k_, v_ in (("EXPERIMENT_ID", tag), ("RUN_LABEL", tag)):
                cfg = "\n".join(f'{k_}="{v_}"' if l.startswith(k_+"=") else l
                                for l in cfg.split("\n"))
            repl = {
                "SHARDING_MODE": f'"{mode}"',
                "TP_SIZE": "2" if mode == "tp2" else "1",
                "DP_SIZE": "1" if mode == "tp2" else "2",
                "EP_SIZE": "1" if mode == "tp2" else "2",
                "LIN_TOKENS": str(a.lin), "LOUT_TOKENS": str(a.lout),
                "BATCH_SIZE": str(bs), "SBATCH_MEM": '"1000G"',
                "SBATCH_TIME": f'"{a.hours:02d}:00:00"',
                "INDEXER_DUMP_DIR": '""', "INDEXER_DUMP_LAYERS": '""',
            }
            out_lines = []
            for l in cfg.split("\n"):
                kk = l.split("=")[0]
                out_lines.append(f"{kk}={repl[kk]}" if kk in repl else l)
            cfg = "\n".join(out_lines)
            cfg += f"\n\nPREFILL_CHUNK={a.chunk}\n"
            fn = (f"{tag}_generate_{mode}_lin{a.lin}_lout{a.lout}_bs{bs}"
                  f"_n2_c32_mem1000g_tprof0_mprof0.env")
            open(os.path.join(here, "configs", fn), "w").write(cfg)
            made.append(fn)
    if made:
        print("\nwrote:"); [print("  " + m) for m in made]

if __name__ == "__main__":
    main()
