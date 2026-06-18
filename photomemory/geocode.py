"""Offline reverse geocoding (GPS -> human place label) and small geo helpers.

Wraps the `reverse_geocoder` package, which ships an offline city database (no
network). Lookups are batched and the module is imported lazily so the rest of the
pipeline works even if it isn't installed.
"""
from __future__ import annotations

import math
from functools import lru_cache

_rg = None


def _get_rg():
    global _rg
    if _rg is None:
        import reverse_geocoder as rg  # lazy import (slow first load)

        _rg = rg
    return _rg


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def reverse_batch(coords: list[tuple[float, float]]) -> list[str | None]:
    """Map a list of (lat, lon) to place labels like 'City, Region, CC'."""
    if not coords:
        return []
    try:
        rg = _get_rg()
    except Exception:
        return [None] * len(coords)
    results = rg.search(coords)  # list of dicts with name/admin1/cc
    labels: list[str | None] = []
    for r in results:
        parts = [r.get("name"), r.get("admin1"), r.get("cc")]
        labels.append(", ".join(p for p in parts if p) or None)
    return labels


@lru_cache(maxsize=4096)
def reverse_one(lat: float, lon: float) -> str | None:
    return reverse_batch([(lat, lon)])[0]
