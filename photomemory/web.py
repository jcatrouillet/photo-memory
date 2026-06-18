"""Local web UI backend (FastAPI).

Serves a single-page app that lets you browse months/years/trips, preview a memory's
auto-selection, deselect shots, render, and watch/download the results — plus a People
view. Long operations (selection scoring, rendering) run as background jobs the frontend
polls. Everything is local; bind to 127.0.0.1.
"""
from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import db, faceindex
from .config import get_config
from .grouping import detect_trips, year_events
from .memory import Memory, build_month, build_trip, build_year, target_count
from .quality.score import ensure_frame
from .render import render_memory

app = FastAPI(title="Photo Memory")
STATIC = Path(__file__).parent / "static"

# Single worker so heavy GPU jobs (scoring/rendering) run one at a time.
_pool = ThreadPoolExecutor(max_workers=1)
_jobs: dict[str, dict] = {}
_previews: dict[str, Memory] = {}
_lock = threading.Lock()

MONTHS = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# --- jobs ----------------------------------------------------------------------

def _submit(fn) -> str:
    jid = uuid.uuid4().hex[:12]
    _jobs[jid] = {"status": "running", "result": None, "error": None}

    def run():
        try:
            _jobs[jid]["result"] = fn()
            _jobs[jid]["status"] = "done"
        except Exception as e:  # surface errors to the UI
            _jobs[jid]["error"] = f"{type(e).__name__}: {e}"
            _jobs[jid]["status"] = "error"

    _pool.submit(run)
    return jid


@app.get("/api/jobs/{jid}")
def job_status(jid: str):
    j = _jobs.get(jid)
    if not j:
        raise HTTPException(404, "no such job")
    return j


# --- browse --------------------------------------------------------------------

@app.get("/api/stats")
def stats():
    db.init_db()
    with db.connect() as c:
        total = c.execute("SELECT COUNT(*) n FROM media").fetchone()["n"]
        gps = c.execute("SELECT COUNT(*) n FROM media WHERE gps_lat IS NOT NULL").fetchone()["n"]
        rng = c.execute("SELECT MIN(capture_dt) a, MAX(capture_dt) b FROM media "
                        "WHERE capture_dt IS NOT NULL").fetchone()
        years = [{"year": int(r["y"]), "count": r["n"]} for r in c.execute(
            "SELECT substr(capture_dt,1,4) y, COUNT(*) n FROM media "
            "WHERE capture_dt IS NOT NULL GROUP BY y ORDER BY y DESC")]
    people = [{"name": n, "faces": f} for n, f in faceindex.list_people()]
    return {"total": total, "gps": gps,
            "range": [rng["a"][:10] if rng["a"] else None, rng["b"][:10] if rng["b"] else None],
            "years": years, "people": people}


@app.get("/api/months/{year}")
def months(year: int):
    out = []
    for m, events in year_events(year).items():
        total = sum(e.count for e in events)
        top = sorted(events, key=lambda e: e.count, reverse=True)[:3]
        out.append({"key": f"{year}-{m:02d}", "label": f"{MONTHS[m]} {year}",
                    "count": total, "highlights": [e.label for e in top]})
    return out


@app.get("/api/trips")
def trips(year: int | None = None):
    return [{"index": i, "label": t.label, "start": t.date_start, "end": t.date_end,
             "days": t.span_days, "count": t.count, "place": t.place, "year": year}
            for i, t in enumerate(detect_trips(year=year))]


@app.get("/api/videos")
def videos():
    out_dir = get_config().output_dir
    vids = []
    for p in sorted(out_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True):
        st = p.stat()
        vids.append({"name": p.name, "size_mb": round(st.st_size / 1e6, 1), "mtime": st.st_mtime})
    return vids


# --- media serving -------------------------------------------------------------

@app.get("/api/thumb/{media_id}")
def thumb(media_id: int):
    with db.connect() as c:
        row = c.execute("SELECT path FROM media WHERE id=?", (media_id,)).fetchone()
    if not row:
        raise HTTPException(404)
    frame = ensure_frame(media_id, Path(row["path"]))
    if not frame:
        raise HTTPException(404)
    return FileResponse(frame, media_type="image/jpeg")


@app.get("/api/montages")
def montages():
    d = get_config().output_dir / "people_review"
    if not d.exists():
        return []
    return [p.name for p in sorted(d.glob("*.jpg"))]


@app.get("/api/montage/{name}")
def montage(name: str):
    p = get_config().output_dir / "people_review" / Path(name).name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/jpeg")


# --- preview & render ----------------------------------------------------------

class PreviewReq(BaseModel):
    type: str          # month | year | trip
    key: str = ""      # "2024-10" for month, "2024" for year, trip index for trip
    year: int | None = None
    length: float | None = None


class RenderReq(BaseModel):
    preview_id: str
    keep_ids: list[int]
    music: bool = True


def _items(mem: Memory):
    return [{"media_id": s.media_id, "date": (s.capture_dt or "")[:10],
             "group": s.group, "caption": mem.captions.get(s.group, ""),
             "people": list(s.people), "score": round(s.composite, 3)}
            for s in mem.selection]


@app.post("/api/preview")
def preview(req: PreviewReq):
    cfg = get_config().render

    def fn():
        length = req.length or cfg["default_length_seconds"]
        tgt = target_count(length, cfg["seconds_per_photo"])
        if req.type == "month":
            y, m = req.key.split("-")
            mem = build_month(int(y), int(m), tgt)
        elif req.type == "year":
            mem = build_year(int(req.key), max(tgt, target_count(120, cfg["seconds_per_photo"])))
        elif req.type == "trip":
            trs = detect_trips(year=req.year)
            mem = build_trip(trs[int(req.key)], tgt)
        else:
            raise ValueError("bad type")
        pid = uuid.uuid4().hex[:12]
        with _lock:
            _previews[pid] = mem
        return {"preview_id": pid, "title": mem.title, "subtitle": mem.subtitle,
                "items": _items(mem)}

    return {"job": _submit(fn)}


@app.post("/api/render")
def render(req: RenderReq):
    mem = _previews.get(req.preview_id)
    if not mem:
        raise HTTPException(404, "preview expired; re-run preview")
    keep = set(req.keep_ids)

    def fn():
        sel = [s for s in mem.selection if s.media_id in keep]
        if not sel:
            raise ValueError("nothing selected")
        sub = Memory(title=mem.title, subtitle=mem.subtitle, selection=sel, captions=mem.captions)
        import re
        slug = re.sub(r"[^A-Za-z0-9]+", "_", mem.title).strip("_").lower() or "memory"
        out = get_config().output_dir / f"{slug}.mp4"
        render_memory(sub, out, music=req.music)
        return {"video": out.name}

    return {"job": _submit(fn)}


# --- static --------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")


def _mount_media():
    out_dir = get_config().output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/media", StaticFiles(directory=str(out_dir)), name="media")


def serve(host: str = "127.0.0.1", port: int = 8000):
    import uvicorn
    db.init_db()
    _mount_media()
    print(f"Photo Memory web UI -> http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
