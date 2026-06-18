"""Selection: turn a scored, deduped candidate pool into a final ordered photo set.

Goals:
  * cover the whole period (distribute the quota across sub-groups, e.g. days/events),
  * within each group prefer the highest-scoring shots,
  * return results in chronological order for the renderer.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Selected:
    media_id: int
    path: str
    capture_dt: str
    group: str          # the sub-group it represents (day or event label)
    composite: float
    rank: float = 0.0   # ordering score (composite + family boost); falls back to composite
    people: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.rank:
            self.rank = self.composite


def distribute(pool: list[Selected], target: int) -> list[Selected]:
    """Pick ~target items spread across groups, best-within-group, chronological out.

    Uses round-robin over groups (each ordered best-first), so a busy day can't crowd
    out the rest of the period until every group has had a fair share.
    """
    if target <= 0 or not pool:
        return []
    if len(pool) <= target:
        return sorted(pool, key=lambda s: s.capture_dt)

    groups: dict[str, list[Selected]] = defaultdict(list)
    for s in pool:
        groups[s.group].append(s)
    for g in groups.values():
        g.sort(key=lambda s: s.rank, reverse=True)

    # Order groups by chronology of their best shot so round-robin feels temporal.
    ordered_groups = sorted(
        groups.values(), key=lambda g: min(s.capture_dt for s in g)
    )

    chosen: list[Selected] = []
    idx = 0
    while len(chosen) < target and any(idx < len(g) for g in ordered_groups):
        for g in ordered_groups:
            if idx < len(g):
                chosen.append(g[idx])
                if len(chosen) >= target:
                    break
        idx += 1

    return sorted(chosen, key=lambda s: s.capture_dt)
