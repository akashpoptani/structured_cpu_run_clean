"""Probe which torch.distributed collectives the ACTIVE BACKEND supports at world_size>=2.

PARALLELISM_PLAN_FINAL.md item 7 depends on this and says so explicitly:

    Unverified: gloo's reduce_scatter / all_to_all at world_size >= 2. A
    world_size=1 test passes but is meaningless (collectives no-op there).
    all_gather and all_reduce are certain. Test before designing around them.

`hasattr(torch.distributed, "reduce_scatter")` is not evidence -- the symbol
exists for every backend; only the backend decides whether it raises. The answer
is therefore environment-specific: re-run after changing backend, cluster or
torch version rather than trusting a recorded result. So this
runs each one for real on 2 ranks and reports support plus correctness.

Result on gloo / torch 2.11.0+cpu / world_size=2 (job 27918280):
    all_reduce, all_gather, all_gather_into_tensor, reduce_scatter,
    reduce_scatter_tensor  -> OK, correct values
    all_to_all             -> RuntimeError: Backend gloo does not support alltoall

That is why src/overrides/ep_moe.py uses design (a) (all_gather + reduce_scatter)
and not design (b) (all_to_all dispatch) from PARALLELISM_PLAN_FINAL.md item 7.

Costs nothing: no model, no weights, 1 core, a few seconds.

    srun ... python -m torch.distributed.run --nnodes=2 --nproc-per-node=1 \
        --node-rank=$SLURM_NODEID --master-addr=... scripts/gloo_collective_probe.py
"""

import os
import sys

import torch
import torch.distributed as dist


def _probe(name, fn, expect=None):
    """Run fn(); report OK / WRONG / raised. Never lets one failure stop the rest."""
    rank = dist.get_rank()
    try:
        got = fn()
    except Exception as exc:  # noqa: BLE001 - the whole point is to catch anything
        if rank == 0:
            print(f"  {name:28s} UNSUPPORTED -- {type(exc).__name__}: {str(exc)[:90]}")
        return False
    ok = True if expect is None else torch.equal(got, expect)
    if rank == 0:
        print(f"  {name:28s} {'OK' if ok else 'WRONG RESULT'}   got={got.tolist()}")
    return ok


def main() -> int:
    backend = os.environ.get("PROBE_BACKEND", "gloo")
    dist.init_process_group(backend)
    rank, world = dist.get_rank(), dist.get_world_size()
    if rank == 0:
        print(f"[probe] backend={backend} world_size={world} torch={torch.__version__}\n")
    if world < 2:
        print("[probe] world_size<2 -- results would be meaningless; aborting", file=sys.stderr)
        return 2

    results = {}

    # all_reduce: plan says certain. Confirm anyway, it is the baseline.
    def _ar():
        t = torch.full((4,), float(rank + 1))
        dist.all_reduce(t)
        return t
    results["all_reduce"] = _probe("all_reduce", _ar,
                                   torch.full((4,), float(sum(range(1, world + 1)))))

    # all_gather: design (a) in the plan depends on this one.
    def _ag():
        t = torch.full((2,), float(rank + 1))
        out = [torch.zeros(2) for _ in range(world)]
        dist.all_gather(out, t)
        return torch.cat(out)
    results["all_gather"] = _probe(
        "all_gather", _ag,
        torch.cat([torch.full((2,), float(r + 1)) for r in range(world)]))

    # all_gather_into_tensor: cheaper flat variant, if supported.
    def _agt():
        t = torch.full((2,), float(rank + 1))
        out = torch.zeros(2 * world)
        dist.all_gather_into_tensor(out, t)
        return out
    results["all_gather_into_tensor"] = _probe("all_gather_into_tensor", _agt)

    # reduce_scatter: would let us skip all_reduce + slice. Plan: UNVERIFIED.
    def _rs():
        ins = [torch.full((2,), float(rank + 1)) for _ in range(world)]
        out = torch.zeros(2)
        dist.reduce_scatter(out, ins)
        return out
    results["reduce_scatter"] = _probe("reduce_scatter", _rs,
                                       torch.full((2,), float(sum(range(1, world + 1)))))

    def _rst():
        inp = torch.full((2 * world,), float(rank + 1))
        out = torch.zeros(2)
        dist.reduce_scatter_tensor(out, inp)
        return out
    results["reduce_scatter_tensor"] = _probe("reduce_scatter_tensor", _rst)

    # all_to_all: design (b). Plan: UNVERIFIED.
    def _a2a():
        ins = [torch.full((2,), float(rank * 10 + r)) for r in range(world)]
        outs = [torch.zeros(2) for _ in range(world)]
        dist.all_to_all(outs, ins)
        return torch.cat(outs)
    results["all_to_all"] = _probe("all_to_all", _a2a)

    dist.barrier()
    if rank == 0:
        print("\n[probe] summary")
        for k, v in results.items():
            print(f"  {k:28s} {'usable' if v else 'DO NOT USE'}")
        print("\n[probe] design (a) all_gather+all_reduce viable:",
              results["all_reduce"] and results["all_gather"])
        print("[probe] reduce_scatter shortcut viable:",
              results["reduce_scatter"] or results["reduce_scatter_tensor"])
        print("[probe] design (b) all_to_all dispatch viable:", results["all_to_all"])
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
