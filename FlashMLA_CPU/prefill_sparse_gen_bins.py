"""Reference data for the SPARSE prefill kernel (absorbed / MQA form).

Shapes follow the absorbed path that src/overrides/mla_absorbed.py installs,
NOT the per-head MHA form:

    Q      [B][Sq][H][576]   absorbed q_nope(512) ++ q_pe(64)
    KV     [B][Skv][576]     SHARED latent, V = first 512 dims
    topk   [B][Sq][K]        per-query indices into Skv
    golden [B][Sq][H][512]

Why this and not per-head K/V: by MLA absorption the two are mathematically
identical, but the absorbed form keeps the latent SHARED across all heads, so a
query's gathered tokens serve every one of its heads, and no 10.7 GB/layer of
per-head K/V is built at 128k.

Indices are taken from the REAL indexer dump (job 27925600) rather than
synthesized, so the kernel sees genuine index patterns, including the rows
below 2048 whose top-k is padded with future positions.

Usage: ../.venv/bin/python prefill_sparse_gen_bins.py --topk-npy <dump.npy> [--sq 256]
"""
import argparse, os, struct
import numpy as np
import torch

DLAT, DV = 576, 512


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk-npy", required=True, help="topk_layer*.npy from indexer_dump")
    ap.add_argument("--sq", type=int, default=256, help="query positions to keep")
    ap.add_argument("--q-start", type=int, default=None,
                    help="first query row to take (default: first genuinely sparse row)")
    ap.add_argument("--heads", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--no-golden", action="store_true")
    ap.add_argument("--out", default="prefill_sparse_bins")
    a = ap.parse_args()

    tk = np.load(a.topk_npy)
    if tk.ndim == 3:
        tk = tk[0]
    full_sq, K = tk.shape
    # Default skips rows below K (dense, uninteresting for PERFORMANCE). For
    # CORRECTNESS use --q-start 0: those are exactly the rows whose topk is
    # padded with future positions, i.e. the ones the causal filter exists for.
    q0 = a.q_start if a.q_start is not None else K
    if q0 + a.sq > full_sq:
        raise SystemExit(f"q_start+sq ({q0+a.sq}) > rows in dump ({full_sq})")
    # rows are absolute positions q0..q0+sq-1, so the cache must reach the last one
    Skv = max(int(tk.max()) + 1, q0 + a.sq)
    idx = tk[q0:q0 + a.sq].astype(np.int32)
    B, Sq, H = 1, a.sq, a.heads
    scale = a.scale if a.scale is not None else DLAT ** -0.5

    sel = (idx < Skv).all()
    assert sel, "indices out of range"
    n_future = int((idx > (q0 + np.arange(Sq))[:, None]).sum())
    print(f"B={B} Sq={Sq} Skv={Skv} H={H} K={K} q_start={q0} scale={scale:.8f}")
    print(f"  real indices from {os.path.basename(a.topk_npy)}; "
          f"{n_future} future indices that the causal filter must drop")

    g = torch.Generator().manual_seed(a.seed)
    q  = (torch.randn(B, Sq,  H, DLAT, generator=g) * 0.05).to(torch.bfloat16)
    kv = (torch.randn(B, Skv,    DLAT, generator=g) * 0.05).to(torch.bfloat16)

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), a.out)
    os.makedirs(outdir, exist_ok=True)
    q.view(torch.int16).numpy().tofile(f"{outdir}/q.bin")
    kv.view(torch.int16).numpy().tofile(f"{outdir}/kv.bin")
    idx.tofile(f"{outdir}/topk.bin")
    with open(f"{outdir}/config.bin", "wb") as f:
        # trailing start_pos = q0: the kernel's causal filter keys on it
        f.write(struct.pack("iiiiiifi", B, H, Sq, Skv, DLAT, K, scale, q0))

    if not a.no_golden:
        kvf = kv[0].float()                                    # [Skv, 576]
        golden = torch.empty(B, Sq, H, DV, dtype=torch.float32)
        ti = torch.from_numpy(idx).long()                      # [Sq, K]
        for s in range(Sq):
            keep = ti[s][ti[s] <= q0 + s]                      # causal, as upstream's cmask
            sel_kv = kvf[keep]                                 # [<=K, 576]
            sc = (q[0, s].float() @ sel_kv.T) * scale          # [H, K]
            golden[0, s] = torch.softmax(sc, dim=-1) @ sel_kv[:, :DV]
            if s % 64 == 0:
                print(f"  reference {s}/{Sq}")
        golden.numpy().tofile(f"{outdir}/out_golden.bin")
        assert torch.isfinite(golden).all(), "golden non-finite"
        vmin, vmax = kvf[:, :DV].min().item(), kvf[:, :DV].max().item()
        assert vmin - 1e-3 <= golden.min().item() and golden.max().item() <= vmax + 1e-3, \
            "golden outside V hull -- output must be a convex combination of selected V rows"
        print(f"  validated: finite, within V range [{vmin:.4f}, {vmax:.4f}]")

    for n in ("config", "q", "kv", "topk") + (() if a.no_golden else ("out_golden",)):
        p = f"{outdir}/{n}.bin"
        print(f"  -> {p} ({os.path.getsize(p)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
