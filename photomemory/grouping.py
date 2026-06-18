"""Grouping: turn catalog rows into events, monthly/yearly buckets, and trips.

An *event* is a single folder (a day with a human title, e.g. "Lassen Volcanic
National Park"). A *trip* is a run of consecutive days that hang together — detected
from date adjacency and strengthened by GPS (a location away from home, or a change of
reverse-geocoded place, promotes and names a trip).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from . import db
from .config import get_config
from .geocode import haversine_km


@dataclass
class Event:
    folder_id: int | None
    title: str | None
    date_start: str          # ISO date
    date_end: str
    count: int
    place: str | None
    media_ids: list[int] = field(default_factory=list)

    @property
    def label(self) -> str:
        return self.title or self.place or self.date_start


@dataclass
class Trip:
    date_start: str
    date_end: str
    count: int
    place: str | None
    titles: list[str]
    media_ids: list[int] = field(default_factory=list)

    @property
    def span_days(self) -> int:
        a = date.fromisoformat(self.date_start)
        b = date.fromisoformat(self.date_end)
        return (b - a).days + 1

    @property
    def label(self) -> str:
        if self.place:
            return self.place.split(",")[0]
        if self.titles:
            return Counter(self.titles).most_common(1)[0][0]
        return f"{self.date_start} – {self.date_end}"


def _day(iso: str) -> str:
    return iso[:10]


# --- Events --------------------------------------------------------------------

def _events_from_rows(rows) -> list[Event]:
    by_folder: dict[int, list] = defaultdict(list)
    for r in rows:
        by_folder[r["folder_id"]].append(r)
    events: list[Event] = []
    for fid, items in by_folder.items():
        items.sort(key=lambda r: r["capture_dt"] or "")
        places = [r["place"] for r in items if r["place"]]
        events.append(
            Event(
                folder_id=fid,
                title=items[0]["title"],
                date_start=_day(items[0]["capture_dt"]),
                date_end=_day(items[-1]["capture_dt"]),
                count=len(items),
                place=Counter(places).most_common(1)[0][0] if places else None,
                media_ids=[r["id"] for r in items],
            )
        )
    events.sort(key=lambda e: e.date_start)
    return events


def _query_range(conn, start: str, end: str):
    return conn.execute(
        """
        SELECT m.id, m.capture_dt, m.folder_id, m.place, m.gps_lat, m.gps_lon,
               m.media_type, f.title
        FROM media m LEFT JOIN folders f ON f.id = m.folder_id
        WHERE m.capture_dt >= ? AND m.capture_dt < ? AND m.media_type='image'
        ORDER BY m.capture_dt
        """,
        (start, end),
    ).fetchall()


def month_events(year: int, month: int) -> list[Event]:
    start = date(year, month, 1).isoformat()
    end = (date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)).isoformat()
    with db.connect() as conn:
        return _events_from_rows(_query_range(conn, start, end))


def year_events(year: int) -> dict[int, list[Event]]:
    start = date(year, 1, 1).isoformat()
    end = date(year + 1, 1, 1).isoformat()
    with db.connect() as conn:
        rows = _query_range(conn, start, end)
    by_month: dict[int, list] = defaultdict(list)
    for r in rows:
        by_month[int(r["capture_dt"][5:7])].append(r)
    return {m: _events_from_rows(rows) for m, rows in sorted(by_month.items())}


# --- Trips ---------------------------------------------------------------------

def _estimate_home(day_coords: dict[str, tuple[float, float]]) -> tuple[float, float] | None:
    """Home = centroid of the densest ~0.5deg GPS cell (by number of distinct days)."""
    cfg = get_config()
    if cfg.home.get("lat") is not None and cfg.home.get("lon") is not None:
        return float(cfg.home["lat"]), float(cfg.home["lon"])
    if not day_coords:
        return None
    cells: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)
    for lat, lon in day_coords.values():
        cells[(round(lat * 2), round(lon * 2))].append((lat, lon))
    best = max(cells.values(), key=len)
    return (sum(c[0] for c in best) / len(best), sum(c[1] for c in best) / len(best))


def _estimate_home_global() -> tuple[float, float] | None:
    """Home = centroid of the densest ~0.5deg GPS cell across the whole library."""
    cfg = get_config()
    if cfg.home.get("lat") is not None and cfg.home.get("lon") is not None:
        return float(cfg.home["lat"]), float(cfg.home["lon"])
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT gps_lat, gps_lon FROM media WHERE gps_lat IS NOT NULL AND gps_lon IS NOT NULL"
        ).fetchall()
    if not rows:
        return None
    cells: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        cells[(round(r["gps_lat"] * 2), round(r["gps_lon"] * 2))].append(
            (r["gps_lat"], r["gps_lon"])
        )
    best = max(cells.values(), key=len)
    return (sum(c[0] for c in best) / len(best), sum(c[1] for c in best) / len(best))


def detect_trips(year: int | None = None) -> list[Trip]:
    cfg = get_config()
    gap = int(cfg.grouping["trip_max_gap_days"])
    min_span = int(cfg.grouping["trip_min_span_days"])
    home_radius = float(cfg.grouping["home_radius_km"])

    where = "m.media_type='image' AND m.capture_dt IS NOT NULL"
    params: list = []
    if year is not None:
        where += " AND m.capture_dt >= ? AND m.capture_dt < ?"
        params = [date(year, 1, 1).isoformat(), date(year + 1, 1, 1).isoformat()]

    with db.connect() as conn:
        rows = conn.execute(
            f"""
            SELECT m.id, m.capture_dt, m.place, m.gps_lat, m.gps_lon, f.title
            FROM media m LEFT JOIN folders f ON f.id = m.folder_id
            WHERE {where} ORDER BY m.capture_dt
            """,
            params,
        ).fetchall()

    # Aggregate per day.
    days: dict[str, dict] = defaultdict(lambda: {"ids": [], "places": [], "titles": set(),
                                                 "lat": [], "lon": []})
    for r in rows:
        d = days[_day(r["capture_dt"])]
        d["ids"].append(r["id"])
        if r["place"]:
            d["places"].append(r["place"])
        if r["title"]:
            d["titles"].add(r["title"])
        if r["gps_lat"] is not None:
            d["lat"].append(r["gps_lat"])
            d["lon"].append(r["gps_lon"])

    if not days:
        return []

    day_coords = {
        d: (sum(v["lat"]) / len(v["lat"]), sum(v["lon"]) / len(v["lon"]))
        for d, v in days.items() if v["lat"]
    }
    home = _estimate_home_global()  # stable, library-wide home location

    # Classify each day: away (GPS far from home), home (GPS near home), or unknown.
    def status(d: str) -> str:
        if d not in day_coords or not home:
            return "unknown"
        far = haversine_km(home[0], home[1], *day_coords[d]) > home_radius
        return "away" if far else "home"

    sorted_days = sorted(days.keys())

    # Build runs of consecutive non-home days (gap tolerant). Home days are separators,
    # so everyday-at-home stretches never become trips. A run is a trip if it contains
    # a real away day, or (pre-GPS era) is entirely unknown but a bounded multi-day burst.
    MAX_UNKNOWN_SPAN = 30   # cap date-only "trips" so home life doesn't merge for years
    MIN_UNKNOWN_PHOTOS = 15

    runs: list[list[str]] = []
    cur: list[str] = []
    prev_day = None
    for d in sorted_days:
        gap_ok = prev_day is None or (date.fromisoformat(d) - date.fromisoformat(prev_day)).days <= gap + 1
        if status(d) == "home" or not gap_ok:
            if cur:
                runs.append(cur)
            cur = [] if status(d) == "home" else [d]
        else:
            cur.append(d)
        prev_day = d
    if cur:
        runs.append(cur)

    trips: list[Trip] = []
    for run in runs:
        statuses = {status(d) for d in run}
        span = (date.fromisoformat(run[-1]) - date.fromisoformat(run[0])).days + 1
        ids, places, titles = [], [], []
        for d in run:
            ids += days[d]["ids"]
            places += days[d]["places"]
            titles += list(days[d]["titles"])

        has_away = "away" in statuses
        all_unknown = statuses == {"unknown"}
        if has_away:
            if span < min_span:
                continue
        elif all_unknown:
            if span < min_span or span > MAX_UNKNOWN_SPAN or len(ids) < MIN_UNKNOWN_PHOTOS:
                continue
        else:
            continue  # nothing away and not a clean pre-GPS burst

        trips.append(
            Trip(
                date_start=run[0],
                date_end=run[-1],
                count=len(ids),
                place=Counter(places).most_common(1)[0][0] if places else None,
                titles=titles,
                media_ids=ids,
            )
        )
    trips.sort(key=lambda t: t.date_start)
    return trips
