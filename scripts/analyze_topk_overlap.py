"""Measure Indexer top-k overlap between adjacent query positions.

This picks the sparse-prefill kernel structure. Three numbers, three designs:

  1. ADJACENT OVERLAP  |A_i & A_i+1| / topk
     How coherent the index sets are from one query to the next.

  2. TILE UNION FACTOR  |union over 16 queries| / topk
     Cost of union-gather: a 16-row AMX query tile gathers the union and masks
     per row. 1.0 means free; 16.0 means every query wants different tokens and
     the tile does 16x the necessary work.

  3. BLOCK TOUCH RATE  touched 64-token blocks / causally-reachable blocks
     Cost of mask-over-dense: compute blocks densely, null out non-top-k. Only
     pays if whole blocks can be skipped. Low rate = clustered = this wins.

Usage: python3 scripts/analyze_topk_overlap.py --dump-dir <dir with topk_layer*.npy>
"""
import argparse, glob, json, os
import numpy as np

TILE, BLOCK = 16, 64


def analyze(path: str) -> dict:
    a = np.load(path)                       # [bsz, seqlen, topk] int32
    if a.ndim == 2:
        a = a[None]
    bsz, seqlen, topk = a.shape
    Skv = int(a.max()) + 1
    out = {"path": os.path.basename(path), "bsz": bsz, "seqlen": seqlen,
           "topk": topk, "skv_implied": Skv}
    if topk >= seqlen:
        out["verdict"] = ("topk >= seqlen: NOT sparse. topk(min(index_topk, end_pos)) "
                          "selected every position. Re-run with Lin > 2048.")
        return out

    b = 0
    # boolean membership [seqlen, Skv]; rows before index_topk are full-causal
    M = np.zeros((seqlen, Skv), dtype=bool)
    rows = np.repeat(np.arange(seqlen), topk)
    M[rows, a[b].reshape(-1)] = True

    # 1. adjacent overlap
    inter = np.logical_and(M[:-1], M[1:]).sum(axis=1)
    denom = np.minimum(M[:-1].sum(axis=1), M[1:].sum(axis=1)).clip(min=1)
    adj = inter / denom
    out["adjacent_overlap"] = {
        "mean": float(adj.mean()), "p10": float(np.percentile(adj, 10)),
        "median": float(np.median(adj)), "p90": float(np.percentile(adj, 90))}

    # 2. tile union factor
    nt = seqlen // TILE
    uf, touch = [], []
    for t in range(nt):
        sl = M[t * TILE:(t + 1) * TILE]
        used = sl.sum(axis=1).max()
        if used == 0:
            continue
        u = sl.any(axis=0)
        uf.append(u.sum() / used)
        # 3. block touch rate, against causally reachable blocks only
        nb = (Skv + BLOCK - 1) // BLOCK
        pad = np.zeros(nb * BLOCK, dtype=bool); pad[:Skv] = u
        touched = pad.reshape(nb, BLOCK).any(axis=1).sum()
        # Denominator from the data, not from assumed causality: blocks spanned
        # up to the largest index this tile references. On real (causal) top-k
        # that equals the causal limit, and it keeps the rate <= 1 even if the
        # input is not causal -- which a hardcoded causal bound does not.
        hi = int(np.flatnonzero(u).max()) if u.any() else 0
        reachable = max(1, (hi + BLOCK) // BLOCK)
        touch.append(touched / reachable)
    out["tile_union_factor"] = {
        "mean": float(np.mean(uf)), "median": float(np.median(uf)),
        "p90": float(np.percentile(uf, 90)), "max": float(np.max(uf))}
    out["block_touch_rate"] = {
        "mean": float(np.mean(touch)), "median": float(np.median(touch)),
        "p90": float(np.percentile(touch, 90))}

    u, tu = out["tile_union_factor"]["mean"], out["block_touch_rate"]["mean"]
    if u < 2.0:
        v = f"union-gather: tile union is only {u:.2f}x topk, so a 16-row tile stays dense and cheap"
    elif tu < 0.5:
        v = f"mask-over-dense: union is {u:.2f}x (too wide to gather) but only {tu:.0%} of reachable blocks are touched, so block-skipping pays"
    else:
        v = (f"neither: union {u:.2f}x AND {tu:.0%} of blocks touched -- index sets are "
             f"incoherent across a tile; per-query gather (M=1) may be unavoidable")
    out["verdict"] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", required=True)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.dump_dir, "topk_layer*.npy")))
    if not files:
        raise SystemExit(f"no topk_layer*.npy in {a.dump_dir}")
    res = []
    for f in files:
        r = analyze(f)
        res.append(r)
        print(f"\n=== {r['path']}  seqlen={r['seqlen']} topk={r['topk']} ===")
        for k in ("adjacent_overlap", "tile_union_factor", "block_touch_rate"):
            if k in r:
                print(f"  {k:20s} " + "  ".join(f"{kk}={vv:.3f}" for kk, vv in r[k].items()))
        print(f"  -> {r['verdict']}")
    with open(os.path.join(a.dump_dir, "overlap_analysis.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {os.path.join(a.dump_dir, 'overlap_analysis.json')}")


if __name__ == "__main__":
    main()
