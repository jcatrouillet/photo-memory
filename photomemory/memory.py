"""Memory builder: assemble the final ordered photo set for a month, year, or trip.

Pipeline per memory:
  gather candidates -> ensure scored (cached) -> dedupe near-duplicates ->
  distribute selection across sub-groups -> return (title, [Selected]).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date

from . import db
from .config import get_config
from .dedupe import Candidate, dedupe
from .grouping import Trip, month_events, year_events
from .selector import Selected, distribute


@dataclass
class Memory:
    title: str
    subtitle: str
    selection: list[Selected]
    captions: dict[str, str] = field(default_factory=dict)  # group -> friendly label
    extras: list[Selected] = field(default_factory=list)    # next-best, ranked (for "fill")

    def __len__(self) -> int:
        return len(self.selection)

MONTH_NAMES = ["", "January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]


def _fetch(media_ids: list[int]) -> dict[int, dict]:
    if not media_ids:
        return {}
    with db.connect() as conn:
        rows = conn.execute(
            f"""SELECT m.id, m.path, m.capture_dt, s.composite, s.phash
                FROM media m LEFT JOIN scores s ON s.media_id = m.id
                WHERE m.id IN ({','.join('?' * len(media_ids))})""",
            media_ids,
        ).fetchall()
    return {r["id"]: dict(r) for r in rows}


def _sample_capped(media_to_group: dict[int, str], cap: int) -> list[int]:
    """Reduce the candidate pool to ~cap ids, sampled evenly across groups (so we don't
    pay to score thousands of photos over SMB for one video). Sampling is uniform within
    each group, allocating the cap proportionally to group size."""
    groups: dict[str, list[int]] = {}
    for mid, g in media_to_group.items():
        groups.setdefault(g, []).append(mid)
    total = len(media_to_group)
    chosen: list[int] = []
    for mids in groups.values():
        mids.sort()
        k = max(1, round(cap * len(mids) / total))
        if len(mids) <= k:
            chosen.extend(mids)
        else:
            step = len(mids) / k
            chosen.extend(mids[int(i * step)] for i in range(k))
    return chosen


def _assemble(media_to_group: dict[int, str], target: int,
              score_cap: int | None = None) -> tuple[list[Selected], list[Selected]]:
    """Score (if needed), dedupe, distribute. Returns (selection, ranked next-best extras)."""
    from .quality.score import ensure_scored

    ids = list(media_to_group.keys())
    if score_cap and len(ids) > score_cap:
        ids = _sample_capped(media_to_group, score_cap)
    ensure_scored(ids)
    info = _fetch(ids)

    cands = [
        Candidate(media_id=i, path=info[i]["path"], phash=info[i]["phash"],
                  composite=info[i]["composite"] or 0.0)
        for i in ids if i in info
    ]
    kept = dedupe(cands)

    # Family boost: if any people are enrolled, extract+match faces on the kept set and
    # strongly favor photos featuring family members (config scoring.w_family).
    family = _family_for([c.media_id for c in kept])
    w_family = float(get_config().scoring.get("w_family", 0.6))

    pool = []
    for c in kept:
        fam_score, people = family.get(c.media_id, (0.0, []))
        pool.append(Selected(
            media_id=c.media_id, path=c.path,
            capture_dt=info[c.media_id]["capture_dt"] or "",
            group=media_to_group[c.media_id], composite=c.composite,
            rank=c.composite + w_family * fam_score,
            people=tuple(people),
        ))
    selection = distribute(pool, target)
    chosen = {s.media_id for s in selection}
    extras = sorted((p for p in pool if p.media_id not in chosen),
                    key=lambda s: s.rank, reverse=True)[: max(target * 4, 60)]
    return selection, extras


def _family_for(media_ids: list[int]) -> dict[int, tuple[float, list[str]]]:
    """Return family scores for the given media, or all-zero if no people are enrolled."""
    from . import faceindex

    if not faceindex.list_people():
        return {}
    faceindex.extract(media_ids)            # ensure faces extracted (uses cached frames)
    return faceindex.family_scores(media_ids)


def build_month(year: int, month: int, target: int) -> Memory:
    events = month_events(year, month)
    media_to_group = {mid: e.date_start for e in events for mid in e.media_ids}
    selection, extras = _assemble(media_to_group, target, score_cap=max(target * 15, 300))
    top = [e.label for e in sorted(events, key=lambda e: e.count, reverse=True)[:3]]
    captions = {e.date_start: e.label for e in events}
    return Memory(
        title=f"{MONTH_NAMES[month]} {year}",
        subtitle=", ".join(top),
        selection=selection,
        captions=captions,
        extras=extras,
    )


def build_year(year: int, target: int) -> Memory:
    months = year_events(year)
    media_to_group: dict[int, str] = {}
    captions: dict[str, str] = {}
    highlights: list[str] = []
    for m, events in months.items():
        for e in events:
            for mid in e.media_ids:
                media_to_group[mid] = f"{year}-{m:02d}"
        top = max(events, key=lambda e: e.count, default=None)
        captions[f"{year}-{m:02d}"] = f"{MONTH_NAMES[m]}" + (f" - {top.label}" if top else "")
        if top:
            highlights.append(top.label)
    selection, extras = _assemble(media_to_group, target, score_cap=max(target * 20, 800))
    return Memory(
        title=f"{year}",
        subtitle="A Year in Review",
        selection=selection,
        captions=captions,
        extras=extras,
    )


def build_trip(trip: Trip, target: int) -> Memory:
    with db.connect() as conn:
        rows = conn.execute(
            f"""SELECT id, capture_dt FROM media
                WHERE id IN ({','.join('?' * len(trip.media_ids))})""",
            trip.media_ids,
        ).fetchall()
    media_to_group = {r["id"]: (r["capture_dt"] or "")[:10] for r in rows}
    selection, extras = _assemble(media_to_group, target, score_cap=max(target * 15, 400))
    captions = {(r["capture_dt"] or "")[:10]: trip.label for r in rows}
    return Memory(
        title=trip.label,
        subtitle=f"{trip.date_start} - {trip.date_end}",
        selection=selection,
        captions=captions,
        extras=extras,
    )


def target_count(length_seconds: float, seconds_per_photo: float) -> int:
    return max(5, round(length_seconds / max(0.5, seconds_per_photo)))


def memory_from_ids(title: str, subtitle: str, media_ids: list[int]) -> Memory:
    """Build a Memory from an explicit, user-edited set of photos (chronological).

    Used by the web editor so photos can be freely added/removed before rendering.
    Captions come from each day's folder title or reverse-geocoded place.
    """
    if not media_ids:
        return Memory(title=title, subtitle=subtitle, selection=[], captions={})
    with db.connect() as conn:
        rows = conn.execute(
            f"""SELECT m.id, m.path, m.capture_dt, m.place, f.title
                FROM media m LEFT JOIN folders f ON f.id = m.folder_id
                WHERE m.id IN ({','.join('?' * len(media_ids))})""",
            media_ids,
        ).fetchall()
    rows = sorted(rows, key=lambda r: r["capture_dt"] or "")
    captions: dict[str, str] = {}
    selection: list[Selected] = []
    for r in rows:
        day = (r["capture_dt"] or "")[:10]
        captions.setdefault(day, r["title"] or r["place"] or day)
        selection.append(Selected(media_id=r["id"], path=r["path"],
                                  capture_dt=r["capture_dt"] or "", group=day, composite=0.0))
    return Memory(title=title, subtitle=subtitle, selection=selection, captions=captions)
