"""ShardingPlan — the single source of truth for how a run is sharded.

The clean lane intentionally does NOT decide sharding behavior from upstream's
`world_size` / `rank` module globals scattered across `model.py`. Instead a
single `ShardingPlan`, built once per process from `SHARDING_MODE` + the
rank/world env, answers every "is X partitioned / replicated / does this mode
need collectives" question.

Two import styles both resolve to this one file:
  - `from src.overrides.sharding_plan import ShardingPlan`  (namespace pkg,
    used by tests run from the repo root)
  - `from sharding_plan import get_plan`                    (flat module, used
    by `src/overrides/model.py` once `src/overrides/` is on sys.path)

This module is data + policy only; it imports nothing from torch and performs
no collectives. Runtime wiring (set_plan before construction, the model
override reading get_plan) lands in later stages.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

SUPPORTED_MODES = ("tp2", "dp2", "dp2_epon")

# Modes where routed experts are partitioned across ranks.
_EXPERTS_PARTITIONED = frozenset({"tp2", "dp2_epon"})
# Modes where attention/index heads are partitioned across ranks.
_HEADS_PARTITIONED = frozenset({"tp2"})
# Modes where the token embedding is partitioned across ranks.
_EMBEDDING_PARTITIONED = frozenset({"tp2"})
# Modes where dense (non-expert) layers are replicated on every rank.
_DENSE_REPLICATED = frozenset({"dp2", "dp2_epon"})
# Modes that require a torch.distributed process group at runtime.
_NEEDS_DIST = frozenset({"tp2", "dp2_epon"})


@dataclass
class ShardingPlan:
    """Describes the sharding for one process (rank) of a run.

    Args:
      mode: one of SUPPORTED_MODES.
      rank: this process's rank in [0, world_size).
      world_size: total number of ranks.
      indexer_cross_rank_check: tri-state. None -> use the per-mode default
        (True for tp2, False for dp2 / dp2_epon). Pass an explicit bool to
        override for debugging. After __post_init__ this attribute is always
        a concrete bool.
    """

    mode: str
    rank: int
    world_size: int
    indexer_cross_rank_check: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.mode not in SUPPORTED_MODES:
            raise ValueError(f"unsupported sharding mode {self.mode!r}; expected one of {SUPPORTED_MODES}")
        if not isinstance(self.rank, int) or not isinstance(self.world_size, int):
            raise TypeError(
                f"rank and world_size must be int; got rank={self.rank!r} "
                f"world_size={self.world_size!r}"
            )
        if self.world_size < 1:
            raise ValueError(f"world_size must be >= 1; got {self.world_size}")
        if self.rank < 0:
            raise ValueError(f"rank must be >= 0; got {self.rank}")
        if self.rank >= self.world_size:
            raise ValueError(f"rank ({self.rank}) must be < world_size ({self.world_size})")

        # Per-mode world_size expectations.
        if self.mode == "dp2":
            if self.world_size not in (1, 2):
                raise ValueError(f"dp2 expects world_size 1 or 2 (expected 2); got {self.world_size}")
        elif self.mode == "dp2_epon":
            if self.world_size < 2:
                raise ValueError(f"dp2_epon requires world_size >= 2; got {self.world_size}")
        elif self.mode == "tp2":
            if self.world_size < 1:
                raise ValueError(f"tp2 requires world_size >= 1; got {self.world_size}")

        # Resolve the tri-state indexer flag to a concrete bool.
        if self.indexer_cross_rank_check is None:
            self.indexer_cross_rank_check = self._default_indexer_cross_rank_check()
        else:
            self.indexer_cross_rank_check = bool(self.indexer_cross_rank_check)

    def _default_indexer_cross_rank_check(self) -> bool:
        # True only for tp2 (the known-good path keeps the determinism check).
        # dp2 has no process group; dp2_epon skips the per-layer broadcast as
        # a perf win on a replicated, deterministic Indexer. See docs/DP_DESIGN.md.
        return self.mode == "tp2"

    # ---- topology properties (derived from mode) ----

    @property
    def experts_partitioned(self) -> bool:
        return self.mode in _EXPERTS_PARTITIONED

    @property
    def heads_partitioned(self) -> bool:
        return self.mode in _HEADS_PARTITIONED

    @property
    def embedding_partitioned(self) -> bool:
        return self.mode in _EMBEDDING_PARTITIONED

    @property
    def dense_replicated(self) -> bool:
        return self.mode in _DENSE_REPLICATED

    @property
    def needs_dist(self) -> bool:
        return self.mode in _NEEDS_DIST

    # ---- expert range ----

    def expert_range(self, n_routed_experts: int) -> Tuple[int, int]:
        """Half-open [start, end) range of routed experts owned by this rank.

        Partitioned modes (tp2, dp2_epon) give each rank a contiguous block:
        rank 0 owns the first block, rank 1 the second, etc. Requires
        n_routed_experts to be divisible by world_size. Non-partitioned mode
        (dp2) returns the full range on every rank (experts are replicated).
        """
        if not self.experts_partitioned:
            return (0, int(n_routed_experts))
        if n_routed_experts % self.world_size != 0:
            raise ValueError(
                f"n_routed_experts ({n_routed_experts}) must be divisible by "
                f"world_size ({self.world_size}) for mode {self.mode!r}"
            )
        per_rank = n_routed_experts // self.world_size
        start = self.rank * per_rank
        return (start, start + per_rank)


# ---- module-level active plan (set once per process) ----

_ACTIVE_PLAN: Optional[ShardingPlan] = None


def set_plan(plan: ShardingPlan) -> None:
    """Install the process-wide active ShardingPlan."""
    if not isinstance(plan, ShardingPlan):
        raise TypeError(f"set_plan expects a ShardingPlan; got {type(plan).__name__}")
    global _ACTIVE_PLAN
    _ACTIVE_PLAN = plan


def get_plan() -> ShardingPlan:
    """Return the active ShardingPlan. Fails clearly if none was set."""
    if _ACTIVE_PLAN is None:
        raise RuntimeError(
            "no active ShardingPlan; call sharding_plan.set_plan(...) before "
            "constructing the model"
        )
    return _ACTIVE_PLAN


def clear_plan() -> None:
    """Drop the active ShardingPlan (mainly for tests)."""
    global _ACTIVE_PLAN
    _ACTIVE_PLAN = None
