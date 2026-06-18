"""Scoring orchestrator: load each candidate image once, compute all metrics, cache.

For each image we do a single network read, produce a render-ready downscaled frame
(cached to disk for the renderer), and a smaller array for metric computation. Results
are persisted in the ``scores`` table so repeated memory builds are cheap.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image, ImageOps

from .. import db
from ..config import get_config
from ..exif import pillow_heif  # noqa: F401 - ensures HEIF opener registered
from . import aesthetic, faces, heuristics

RENDER_MAX = 1920   # long side of cached render frame
SCORE_MAX = 1024    # long side used for metric computation


def frame_cache_path(media_id: int) -> Path:
    d = get_config().cache_dir / "frames"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{media_id}.jpg"


def ensure_frame(media_id: int, path: Path) -> Path | None:
    """Return a cached render-ready JPEG for this media, creating it if missing.

    Used by the renderer so it never reads originals (and never trips over HEIC) at
    render time. Returns None if the source can't be read.
    """
    cache = frame_cache_path(media_id)
    if cache.exists():
        return cache
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((RENDER_MAX, RENDER_MAX), Image.LANCZOS)
            im.save(cache, "JPEG", quality=90)
        return cache
    except Exception:
        return None


def _load(path: Path, media_id: int) -> np.ndarray | None:
    """Read the original once: write a cached render frame, return a small RGB array."""
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            # Render cache (long side <= RENDER_MAX).
            cache = frame_cache_path(media_id)
            if not cache.exists():
                r = im.copy()
                r.thumbnail((RENDER_MAX, RENDER_MAX), Image.LANCZOS)
                r.save(cache, "JPEG", quality=90)
            # Scoring array (long side <= SCORE_MAX).
            s = im.copy()
            s.thumbnail((SCORE_MAX, SCORE_MAX), Image.LANCZOS)
            return np.asarray(s, dtype=np.uint8)
    except Exception:
        return None


def _composite(sharp_n: float, expo: float, aes: float | None,
               face_count: int, face_area: float) -> float:
    cfg = get_config().scoring
    cap = max(1, int(cfg["face_bonus_cap"]))
    face_n = 0.7 * min(face_count, cap) / cap + 0.3 * min(1.0, face_area * 4)

    parts = [
        (cfg["w_sharpness"], sharp_n),
        (cfg["w_exposure"], expo),
        (cfg["w_faces"], face_n),
    ]
    if aes is not None:
        parts.append((cfg["w_aesthetic"], aes))
    total_w = sum(w for w, _ in parts) or 1.0
    return sum(w * v for w, v in parts) / total_w


def score_one(path: Path, media_id: int, use_ml: bool) -> dict | None:
    rgb = _load(path, media_id)
    if rgb is None:
        return None
    cpu = _cpu_metrics(rgb, media_id)
    aes = aesthetic.aesthetic_score(rgb) if use_ml else None
    return _finalize(cpu, aes)


def _cpu_metrics(rgb, media_id: int) -> dict:
    """Thread-safe metrics (everything except the GPU aesthetic model)."""
    sharp_n, _ = heuristics.sharpness(rgb)
    expo = heuristics.exposure(rgb)
    face_count, face_area = faces.detect_faces(rgb)
    phash = str(imagehash.phash(Image.fromarray(rgb)))
    return {"media_id": media_id, "sharpness": sharp_n, "exposure": expo,
            "face_count": face_count, "face_area": face_area, "phash": phash}


def _finalize(cpu: dict, aes: float | None) -> dict:
    return {
        "media_id": cpu["media_id"],
        "phash": cpu["phash"],
        "sharpness": cpu["sharpness"],
        "exposure": cpu["exposure"],
        "aesthetic": aes,
        "face_count": cpu["face_count"],
        "face_area": cpu["face_area"],
        "composite": _composite(cpu["sharpness"], cpu["exposure"], aes,
                                cpu["face_count"], cpu["face_area"]),
        "scored_at": datetime.now().isoformat(),
    }


def _load_and_cpu(media_id: int, path: Path) -> dict | None:
    """Worker: read image (over the network), build cache frame, compute CPU metrics."""
    rgb = _load(path, media_id)
    if rgb is None:
        return None
    cpu = _cpu_metrics(rgb, media_id)
    cpu["_rgb"] = rgb  # handed to the GPU step on the main thread
    return cpu


def ensure_scored(media_ids: list[int], force: bool = False, progress: bool = True) -> int:
    """Score any of the given media that lack a cached score. Returns count newly scored.

    Image loads (the SMB-bound bottleneck) and CPU metrics run in a thread pool; the GPU
    NIMA model runs on the main thread, fed by completed workers.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cfg = get_config()
    use_ml = bool(cfg.scoring["use_ml"])
    workers = int(cfg.scan["workers"])

    with db.connect() as conn:
        rows = conn.execute(
            f"SELECT id, path FROM media WHERE id IN ({','.join('?' * len(media_ids))})",
            media_ids,
        ).fetchall() if media_ids else []
        have = set()
        if not force and media_ids:
            have = {
                r["media_id"]
                for r in conn.execute(
                    f"SELECT media_id FROM scores WHERE media_id IN ({','.join('?' * len(media_ids))})",
                    media_ids,
                )
            }

    todo = [(r["id"], Path(r["path"])) for r in rows if force or r["id"] not in have]
    if not todo:
        return 0

    n = 0
    with db.connect() as conn, ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_load_and_cpu, mid, path): mid for mid, path in todo}
        it = as_completed(futs)
        if progress:
            from tqdm import tqdm
            it = tqdm(it, total=len(futs), unit="img", desc="scoring")
        for fut in it:
            cpu = fut.result()
            if cpu is None:
                continue
            rgb = cpu.pop("_rgb")
            aes = aesthetic.aesthetic_score(rgb) if use_ml else None
            db.upsert_score(conn, _finalize(cpu, aes))
            n += 1
            if n % 200 == 0:
                conn.commit()
    return n
