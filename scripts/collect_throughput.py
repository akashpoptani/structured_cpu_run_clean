"""Assemble the TPOT-vs-batch-size matrix from a BS_SWEEP run.

Each sweep point writes results_clean/results/<TAG>/bench_bs<N>.json as it
finishes, so this reports whatever has landed -- a sweep still climbing, or one
killed by the time limit, both give a usable table. The largest BS present per
mode is the measured ceiling reached, which is a LOWER bound on the max batch
size unless the log shows an OOM.
"""
import json, re, sys
from pathlib import Path

MODES = [("HCTP2", "TP2"), ("HCDPEP", "DP2_EPON"), ("HCDPTR", "DP2_True")]
root = Path(__file__).resolve().parent.parent / "results_clean" / "results"

rows = {}
for tag, label in MODES:
    for f in sorted((root / tag).glob("bench_bs*.json")) if (root / tag).is_dir() else []:
        bs = int(re.search(r"bench_bs(\d+)\.json", f.name).group(1))
        try:
            agg = json.loads(f.read_text()).get("bench_aggregate") or {}
        except Exception as exc:
            print(f"  !! {tag} bs={bs}: unreadable ({exc})", file=sys.stderr); continue
        tpot = agg.get("tpot_seconds_mean")
        if tpot:
            rows.setdefault(bs, {})[label] = (tpot, agg.get("ttft_seconds_mean"))

if not rows:
    sys.exit("no bench_bs*.json yet")

hdr = [l for _, l in MODES]
print(f"{'BS':>5} | " + " | ".join(f"{h:>22}" for h in hdr))
print(f"{'':>5} | " + " | ".join(f"{'TPOT ms   tok/s':>22}" for _ in hdr))
print("-" * (7 + 25 * len(hdr)))
for bs in sorted(rows):
    cells = []
    for h in hdr:
        v = rows[bs].get(h)
        cells.append(f"{v[0]*1000:9.0f} {bs/v[0]:9.1f}" + " " * 3 if v else f"{'-':>22}")
    print(f"{bs:5d} | " + " | ".join(cells))
print("\nTPOT = mean seconds per decode step for the whole batch.")
print("tok/s = BS / TPOT, the aggregate decode throughput.")
for _, h in MODES:
    got = [b for b in rows if h in rows[b]]
    if got:
        print(f"  {h:10s} reached BS={max(got)}")
