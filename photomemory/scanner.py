"""Filesystem scanner: walk the NAS roots and populate the catalog incrementally.

Strategy for 350K+ files over SMB:
  * Walk paths first (cheap), load existing (path -> mtime/size) signatures once, and
    skip unchanged files in the main thread.
  * Dispatch only new/changed files to a thread pool — the slow part is the per-image
    EXIF/GPS/dimension read over the network, which is I/O bound and parallelizes well.
  * The main thread is the single SQLite writer (upserts records as workers return).
  * A final pass reverse-geocodes any GPS rows that still lack a place label.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from rich.console import Console

from . import db
from .config import Root, get_config
from .dates import parse_filename, parse_folder
from .exif import read_exif

console = Console()


def _classify(path: Path, image_exts: set[str], video_exts: set[str]) -> str | None:
    ext = path.suffix.lower()
    if ext in image_exts:
        return "image"
    if ext in video_exts:
        return "video"
    return None


def _root_for(path: Path, roots: list[Root]) -> Root | None:
    for r in roots:
        try:
            path.relative_to(r.path)
            return r
        except ValueError:
            continue
    return None


def _resolve_date(
    *, layout: str, folder_date, filename_dt, exif_dt, mtime: float
) -> tuple[str | None, str]:
    """Return (iso_datetime, source) per the layout's preference order."""
    folder_dt = (
        datetime(folder_date.year, folder_date.month, folder_date.day)
        if folder_date else None
    )
    if layout == "filename":
        order = [("filename", filename_dt), ("exif", exif_dt), ("path", folder_dt)]
    else:  # foldername
        order = [("path", folder_dt), ("filename", filename_dt), ("exif", exif_dt)]
    for source, dt in order:
        if dt is not None:
            return dt.isoformat(), source
    return datetime.fromtimestamp(mtime).isoformat(), "mtime"


def _process_file(path: Path, layout: str, read_gps: bool, media_type: str) -> dict:
    """Worker: build a media record for one file (may read EXIF over the network)."""
    st = path.stat()
    folder_date, _ = parse_folder(path.parent.name)
    filename_dt = parse_filename(path.name)

    exif = {"capture_dt": None, "gps_lat": None, "gps_lon": None, "gps_alt": None,
            "width": None, "height": None}
    if media_type == "image" and read_gps:
        exif = read_exif(path)

    capture_iso, source = _resolve_date(
        layout=layout, folder_date=folder_date,
        filename_dt=filename_dt, exif_dt=exif["capture_dt"], mtime=st.st_mtime,
    )
    return {
        "path": str(path),
        "folder_id": None,  # filled by writer
        "capture_dt": capture_iso,
        "date_source": source,
        "gps_lat": exif["gps_lat"],
        "gps_lon": exif["gps_lon"],
        "gps_alt": exif["gps_alt"],
        "place": None,
        "media_type": media_type,
        "width": exif["width"],
        "height": exif["height"],
        "file_size": st.st_size,
        "mtime": st.st_mtime,
        "indexed_at": datetime.now().isoformat(),
        "_parent": str(path.parent),
    }


def _ensure_folder(conn, cache: dict[str, int], parent: Path, root: Root) -> int:
    key = str(parent)
    if key in cache:
        return cache[key]
    fdate, title = parse_folder(parent.name)
    # For filename-dated roots (e.g. Amazon phone dumps) the folder is a device name,
    # not an event — don't use it as a title; captions fall back to the GPS place label.
    if root.layout == "filename":
        title = None
    cache[key] = db.upsert_folder(
        conn,
        path=key,
        source=root.name,
        title=title,
        date_start=fdate.isoformat() if fdate else None,
        date_end=fdate.isoformat() if fdate else None,
        year=fdate.year if fdate else None,
        month=fdate.month if fdate else None,
    )
    return cache[key]


def scan(root_filter: str | None = None, subpath: str | None = None) -> dict:
    """Scan configured roots. Returns summary stats.

    root_filter: only scan the root with this name.
    subpath: restrict to a subdirectory (relative to the root or absolute), e.g. a single year.
    """
    cfg = get_config()
    db.init_db()
    image_exts, video_exts = cfg.image_exts, cfg.video_exts
    read_gps = bool(cfg.scan["read_exif_gps"])
    workers = int(cfg.scan["workers"])

    roots = [r for r in cfg.roots if root_filter in (None, r.name)]

    # 1. Collect candidate files.
    console.print("[bold]Walking directories...[/bold]")
    candidates: list[tuple[Path, Root, str]] = []
    for root in roots:
        base = root.path
        if subpath:
            sp = Path(subpath)
            base = sp if sp.is_absolute() else (root.path / sp)
        if not base.exists():
            console.print(f"[yellow]skip missing root/subpath: {base}[/yellow]")
            continue
        for dirpath, _dirs, files in os.walk(base):
            for fn in files:
                p = Path(dirpath) / fn
                mt = _classify(p, image_exts, video_exts)
                if mt:
                    candidates.append((p, root, mt))
    console.print(f"Found [cyan]{len(candidates):,}[/cyan] media files.")

    # 2. Incremental skip via existing signatures.
    with db.connect() as conn:
        existing = {
            row["path"]: (row["mtime"], row["file_size"])
            for row in conn.execute("SELECT path, mtime, file_size FROM media")
        }
    todo = []
    for p, root, mt in candidates:
        try:
            st = p.stat()
        except OSError:
            continue
        sig = existing.get(str(p))
        if sig and abs(sig[0] - st.st_mtime) < 1 and sig[1] == st.st_size:
            continue
        todo.append((p, root, mt))
    console.print(f"[cyan]{len(todo):,}[/cyan] new/changed (skipped {len(candidates) - len(todo):,}).")

    # 3. Process in parallel; main thread writes.
    from tqdm import tqdm

    inserted = 0
    with db.connect() as conn:
        folder_cache: dict[str, int] = {}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(_process_file, p, root.layout, read_gps, mt): root
                for (p, root, mt) in todo
            }
            for fut in tqdm(as_completed(futs), total=len(futs), unit="img"):
                root = futs[fut]
                try:
                    rec = fut.result()
                except Exception:
                    continue
                rec["folder_id"] = _ensure_folder(
                    conn, folder_cache, Path(rec.pop("_parent")), root
                )
                db.upsert_media(conn, rec)
                inserted += 1
                if inserted % 2000 == 0:
                    conn.commit()

    # 4. Reverse-geocode rows that have GPS but no place yet.
    geocoded = _geocode_missing()

    return {"candidates": len(candidates), "processed": inserted, "geocoded": geocoded}


def _geocode_missing() -> int:
    from .geocode import reverse_batch

    with db.connect() as conn:
        rows = conn.execute(
            "SELECT id, gps_lat, gps_lon FROM media "
            "WHERE gps_lat IS NOT NULL AND gps_lon IS NOT NULL AND place IS NULL"
        ).fetchall()
        if not rows:
            return 0
        console.print(f"Reverse-geocoding [cyan]{len(rows):,}[/cyan] located photos...")
        coords = [(r["gps_lat"], r["gps_lon"]) for r in rows]
        labels = reverse_batch(coords)
        for r, label in zip(rows, labels):
            conn.execute("UPDATE media SET place = ? WHERE id = ?", (label, r["id"]))
    return sum(1 for x in labels if x)
