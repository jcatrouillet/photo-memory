"""Near-duplicate clustering via perceptual hash.

Groups visually similar shots (burst frames, slight reframings, edited copies) and
keeps the best member of each group. When an edited "-LR" (Lightroom) sibling is part
of a cluster it is preferred over the original.
"""
from __future__ import annotations

from dataclasses import dataclass

import imagehash

HAMMING_THRESHOLD = 8  # pHash distance below which two images are "the same shot"


@dataclass
class Candidate:
    media_id: int
    path: str
    phash: str | None
    composite: float


def _is_edited(path: str) -> bool:
    name = path.rsplit("\\", 1)[-1].rsplit("/", 1)[-1].lower()
    return "-lr" in name or "_lr" in name or "-edit" in name


def _better(a: Candidate, b: Candidate) -> Candidate:
    """Pick the keeper between two near-duplicates."""
    ae, be = _is_edited(a.path), _is_edited(b.path)
    if ae != be:
        return a if ae else b
    return a if a.composite >= b.composite else b


def dedupe(candidates: list[Candidate], threshold: int = HAMMING_THRESHOLD) -> list[Candidate]:
    """Return one keeper per near-duplicate cluster (order not guaranteed)."""
    keepers: list[tuple[imagehash.ImageHash | None, Candidate]] = []
    # Process best-first so cluster representatives start strong.
    for cand in sorted(candidates, key=lambda c: c.composite, reverse=True):
        h = None
        if cand.phash:
            try:
                h = imagehash.hex_to_hash(cand.phash)
            except Exception:
                h = None
        placed = False
        if h is not None:
            for i, (kh, keep) in enumerate(keepers):
                if kh is not None and (h - kh) <= threshold:
                    keepers[i] = (kh, _better(keep, cand))
                    placed = True
                    break
        if not placed:
            keepers.append((h, cand))
    return [k for _, k in keepers]
