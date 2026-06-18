"""Face index: extract embeddings, cluster them, label clusters as people, and match.

Workflow:
  1. `extract`  - detect+embed faces for a set of media (reuses cached render frames).
  2. `cluster`  - DBSCAN (cosine) groups embeddings; montages saved for review.
  3. `label`    - assign a name to a cluster; people centroids rebuilt.
  4. `match`    - score how strongly a photo features enrolled people (family boost).
"""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from . import db, facerec
from .config import get_config
from .dates import parse_filename, parse_folder
from .quality.score import ensure_frame

MATCH_THRESHOLD = 0.45   # cosine similarity floor to call a face a known person
AGE_WINDOW = 2.5         # years: compare a candidate to each person's age-appropriate protos


def _decimal_year(iso: str | None) -> float:
    if not iso:
        return float("nan")
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        try:
            dt = datetime.strptime(iso[:10], "%Y-%m-%d")
        except ValueError:
            return float("nan")
    return dt.year + (dt.timetuple().tm_yday - 1) / 365.25


def _resolve_proto_date(path: Path) -> str | None:
    """Date for a reference photo: catalog capture_dt if known, else folder/file name."""
    with db.connect() as conn:
        row = conn.execute("SELECT capture_dt FROM media WHERE path = ?", (str(path),)).fetchone()
    if row and row["capture_dt"]:
        return row["capture_dt"]
    fdate, _ = parse_folder(path.parent.name)
    if fdate:
        return fdate.isoformat()
    fdt = parse_filename(path.name)
    return fdt.isoformat() if fdt else None


def emb_to_blob(e: np.ndarray) -> bytes:
    return np.asarray(e, dtype=np.float32).tobytes()


def blob_to_emb(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)


# --- Extraction ----------------------------------------------------------------

def _extract_one(media_id: int, path: Path):
    frame = ensure_frame(media_id, path)
    if frame is None:
        return media_id, []
    try:
        faces = facerec.detect_and_embed(Image.open(frame))
    except Exception:
        faces = []
    return media_id, faces


def extract(media_ids: list[int], force: bool = False, progress: bool = True) -> int:
    """Detect+embed faces for the given media. Skips already-scanned unless force."""
    if not media_ids:
        return 0
    db.init_db()
    facerec._ensure()  # warm up GPU models on main thread before workers load frames

    with db.connect() as conn:
        rows = conn.execute(
            f"SELECT id, path FROM media WHERE id IN ({','.join('?' * len(media_ids))})",
            media_ids,
        ).fetchall()
        done = set()
        if not force:
            done = {r["media_id"] for r in conn.execute(
                f"SELECT media_id FROM faces_scanned WHERE media_id IN ({','.join('?' * len(media_ids))})",
                media_ids,
            )}

    todo = [(r["id"], Path(r["path"])) for r in rows if force or r["id"] not in done]
    if not todo:
        return 0

    workers = int(get_config().scan["workers"])
    n = 0
    with db.connect() as conn, ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_extract_one, mid, p): mid for mid, p in todo}
        it = as_completed(futs)
        if progress:
            from tqdm import tqdm
            it = tqdm(it, total=len(futs), unit="img", desc="faces")
        for fut in it:
            mid, faces = fut.result()
            if force:
                conn.execute("DELETE FROM faces WHERE media_id = ?", (mid,))
            for f in faces:
                conn.execute(
                    "INSERT INTO faces (media_id,x1,y1,x2,y2,area,prob,embedding) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (mid, *f.bbox, f.area_frac, f.prob, emb_to_blob(f.embedding)),
                )
            conn.execute("INSERT OR IGNORE INTO faces_scanned (media_id) VALUES (?)", (mid,))
            n += 1
            if n % 200 == 0:
                conn.commit()
    return n


def sample_media(n: int) -> list[int]:
    """An even random sample of image media for enrollment clustering."""
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT id FROM media WHERE media_type='image' ORDER BY RANDOM() LIMIT ?", (n,)
        ).fetchall()
    return [r["id"] for r in rows]


# --- Clustering ----------------------------------------------------------------

def cluster(eps: float = 0.40, min_samples: int = 5, only_unlabeled: bool = False) -> dict[int, int]:
    """DBSCAN-cluster face embeddings; write cluster ids. Returns {cluster_id: size}."""
    from sklearn.cluster import DBSCAN

    db.init_db()
    where = "embedding IS NOT NULL AND media_id IS NOT NULL"  # exclude reference prototypes
    if only_unlabeled:
        where += " AND person IS NULL"
    with db.connect() as conn:
        rows = conn.execute(f"SELECT id, embedding FROM faces WHERE {where}").fetchall()
    if not rows:
        return {}
    X = np.stack([blob_to_emb(r["embedding"]) for r in rows])
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine", n_jobs=-1).fit_predict(X)

    sizes: dict[int, int] = defaultdict(int)
    with db.connect() as conn:
        for r, lab in zip(rows, labels):
            cid = int(lab)
            conn.execute("UPDATE faces SET cluster = ? WHERE id = ?",
                         (None if cid < 0 else cid, r["id"]))
            if cid >= 0:
                sizes[cid] += 1
    return dict(sorted(sizes.items(), key=lambda kv: kv[1], reverse=True))


def split_cluster(cluster_id: int, eps: float = 0.24, min_samples: int = 4) -> dict[int, int]:
    """Re-cluster only the faces in one (over-merged) cluster with a tighter threshold.
    New sub-clusters get fresh ids appended after the current max. Returns {id: size}."""
    from sklearn.cluster import DBSCAN

    db.init_db()
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT id, embedding FROM faces WHERE cluster = ?", (cluster_id,)
        ).fetchall()
        base = conn.execute("SELECT COALESCE(MAX(cluster), -1) FROM faces").fetchone()[0] + 1
    if not rows:
        return {}
    X = np.stack([blob_to_emb(r["embedding"]) for r in rows])
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine", n_jobs=-1).fit_predict(X)

    sizes: dict[int, int] = defaultdict(int)
    with db.connect() as conn:
        for r, lab in zip(rows, labels):
            cid = None if lab < 0 else base + int(lab)
            conn.execute("UPDATE faces SET cluster = ? WHERE id = ?", (cid, r["id"]))
            if lab >= 0:
                sizes[base + int(lab)] += 1
    return dict(sorted(sizes.items(), key=lambda kv: kv[1], reverse=True))


def save_montages(top_k: int = 25, per_cluster: int = 16) -> Path:
    """Save a face montage per top cluster to output/people_review for visual labeling."""
    out = get_config().output_dir / "people_review"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("cluster_*.jpg"):
        f.unlink()

    with db.connect() as conn:
        clusters = conn.execute(
            "SELECT cluster, COUNT(*) c FROM faces WHERE cluster IS NOT NULL "
            "GROUP BY cluster ORDER BY c DESC LIMIT ?", (top_k,)
        ).fetchall()
        for row in clusters:
            cid = row["cluster"]
            faces = conn.execute(
                "SELECT f.media_id, f.x1,f.y1,f.x2,f.y2, m.path FROM faces f "
                "JOIN media m ON m.id=f.media_id WHERE f.cluster=? ORDER BY f.area DESC LIMIT ?",
                (cid, per_cluster),
            ).fetchall()
            crops = []
            for fr in faces:
                frame = ensure_frame(fr["media_id"], Path(fr["path"]))
                if not frame:
                    continue
                try:
                    im = Image.open(frame).convert("RGB")
                    crop = im.crop((fr["x1"], fr["y1"], fr["x2"], fr["y2"])).resize((112, 112))
                    crops.append(crop)
                except Exception:
                    continue
            if not crops:
                continue
            cols = 4
            rows_n = (len(crops) + cols - 1) // cols
            grid = Image.new("RGB", (cols * 112, rows_n * 112), (20, 20, 20))
            for i, c in enumerate(crops):
                grid.paste(c, ((i % cols) * 112, (i // cols) * 112))
            grid.save(out / f"cluster_{cid:03d}_n{row['c']}.jpg", "JPEG", quality=88)
    return out


# --- People (labeling) ---------------------------------------------------------

def label_cluster(cluster_id: int, name: str) -> int:
    with db.connect() as conn:
        return conn.execute("UPDATE faces SET person=? WHERE cluster=?",
                            (name, cluster_id)).rowcount


def enroll_photos(name: str, paths: list[str], add: bool = False) -> int:
    """Enroll a person from reference photos (files or folders), stored as prototype faces
    (media_id NULL). Robust to event folders: rather than trusting the largest face, it finds
    the *dominant recurring face* across the photos (the person the folder is about).

    add=False replaces this person's existing reference prototypes; add=True appends.
    """
    db.init_db()
    facerec._ensure()
    files: list[Path] = []
    exts = get_config().image_exts
    for p in paths:
        path = Path(p)
        if path.is_dir():
            files += [f for f in path.rglob("*") if f.suffix.lower() in exts]
        elif path.exists():
            files.append(path)

    embs: list[np.ndarray] = []
    meta: list[tuple[Path, tuple[int, int, int, int]]] = []
    for f in files:
        try:
            faces = facerec.detect_and_embed(Image.open(f))
        except Exception:
            continue
        for fa in faces:
            embs.append(fa.embedding)
            meta.append((f, fa.bbox))
    if not embs:
        return 0

    X = np.stack(embs)
    idx = list(range(len(embs)))
    if len(embs) >= 6:
        from sklearn.cluster import DBSCAN
        labels = DBSCAN(eps=0.32, min_samples=2, metric="cosine", n_jobs=-1).fit_predict(X)
        valid = [l for l in set(labels) if l >= 0]
        if valid:
            best = max(valid, key=lambda l: int((labels == l).sum()))  # most recurring face
            idx = [i for i, l in enumerate(labels) if l == best]

    _save_enroll_montage(name, [meta[i] for i in idx])

    date_cache: dict[Path, str | None] = {}
    with db.connect() as conn:
        if not add:
            conn.execute("DELETE FROM faces WHERE media_id IS NULL AND person = ?", (name,))
        for i in idx:
            f = meta[i][0]
            if f not in date_cache:
                date_cache[f] = _resolve_proto_date(f)
            conn.execute(
                "INSERT INTO faces (media_id, person, embedding, prob, proto_date) "
                "VALUES (NULL, ?, ?, 1.0, ?)",
                (name, emb_to_blob(embs[i]), date_cache[f]),
            )
    return len(idx)


def _save_enroll_montage(name: str, faces: list[tuple[Path, tuple[int, int, int, int]]]) -> None:
    """Save a montage of the faces chosen for a reference enrollment, for verification."""
    out = get_config().output_dir / "people_review"
    out.mkdir(parents=True, exist_ok=True)
    crops = []
    for f, (x1, y1, x2, y2) in faces[:16]:
        try:
            im = Image.open(f).convert("RGB")
            crops.append(im.crop((x1, y1, x2, y2)).resize((112, 112)))
        except Exception:
            continue
    if not crops:
        return
    cols = 4
    rows_n = (len(crops) + cols - 1) // cols
    grid = Image.new("RGB", (cols * 112, rows_n * 112), (20, 20, 20))
    for i, c in enumerate(crops):
        grid.paste(c, ((i % cols) * 112, (i // cols) * 112))
    grid.save(out / f"enrolled_{name}.jpg", "JPEG", quality=88)


def list_people() -> list[tuple[str, int]]:
    """Enrolled people and their prototype counts (cluster-labeled + reference faces)."""
    db.init_db()
    with db.connect() as conn:
        return [(r["person"], r["c"]) for r in conn.execute(
            "SELECT person, COUNT(*) c FROM faces WHERE person IS NOT NULL "
            "GROUP BY person ORDER BY c DESC")]


def set_birthdate(name: str, iso_date: str) -> None:
    db.init_db()
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO person_birth (name, birthdate) VALUES (?, ?) "
            "ON CONFLICT(name) DO UPDATE SET birthdate=excluded.birthdate",
            (name, iso_date),
        )


def get_birthdates() -> dict[str, float]:
    with db.connect() as conn:
        return {r["name"]: _decimal_year(r["birthdate"]) for r in
                conn.execute("SELECT name, birthdate FROM person_birth WHERE birthdate IS NOT NULL")}


def set_active_range(name: str, active_from: str | None, active_until: str | None) -> None:
    db.init_db()
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO person_birth (name, active_from, active_until) VALUES (?,?,?) "
            "ON CONFLICT(name) DO UPDATE SET active_from=COALESCE(excluded.active_from, active_from), "
            "active_until=COALESCE(excluded.active_until, active_until)",
            (name, active_from, active_until),
        )


def get_active_ranges() -> dict[str, tuple[float, float]]:
    """Return {name: (from_year, until_year)} using active_from (or birthdate) and active_until."""
    out: dict[str, tuple[float, float]] = {}
    with db.connect() as conn:
        for r in conn.execute("SELECT name, birthdate, active_from, active_until FROM person_birth"):
            lo = _decimal_year(r["active_from"] or r["birthdate"])
            hi = _decimal_year(r["active_until"])
            out[r["name"]] = (lo if not np.isnan(lo) else -1e9,
                              hi if not np.isnan(hi) else 1e9)
    return out


# --- Matching (family score) ---------------------------------------------------

def _load_prototypes() -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (names[k], embeddings[k,512], proto_years[k]) for every labeled/reference face.

    proto_year is the decimal year of the prototype's source photo (from the catalog for
    cluster faces, or the stored proto_date for reference faces); NaN if unknown.
    """
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT f.person, f.embedding, "
            "       COALESCE(m.capture_dt, f.proto_date) AS dt "
            "FROM faces f LEFT JOIN media m ON m.id = f.media_id "
            "WHERE f.person IS NOT NULL AND f.embedding IS NOT NULL"
        ).fetchall()
    if not rows:
        return None
    names = np.array([r["person"] for r in rows])
    mat = np.stack([blob_to_emb(r["embedding"]) for r in rows])
    years = np.array([_decimal_year(r["dt"]) for r in rows])
    return names, mat, years


def family_scores(media_ids: list[int], threshold: float = MATCH_THRESHOLD
                  ) -> dict[int, tuple[float, list[str]]]:
    """For each media (already face-extracted), return (family_score, [people present]).

    Each candidate face is matched against enrolled prototypes. When birthdates are known,
    a candidate from a photo dated D is compared to each person's prototypes taken near that
    person's age at D (AGE_WINDOW) — so look-alike siblings are separated by age, not just face.
    """
    proto = _load_prototypes()
    if proto is None or not media_ids:
        return {mid: (0.0, []) for mid in media_ids}
    names, mat, pyears = proto
    uniq = list(dict.fromkeys(names.tolist()))
    masks = {p: (names == p) for p in uniq}
    births = get_birthdates()
    active = get_active_ranges()

    with db.connect() as conn:
        rows = conn.execute(
            f"SELECT f.media_id, f.area, f.embedding, m.capture_dt "
            f"FROM faces f JOIN media m ON m.id = f.media_id "
            f"WHERE f.media_id IN ({','.join('?' * len(media_ids))})",
            media_ids,
        ).fetchall()

    per_media: dict[int, list] = defaultdict(list)
    for r in rows:
        per_media[r["media_id"]].append(r)

    def best_person(e: np.ndarray, photo_year: float) -> tuple[str | None, float]:
        sims = mat @ e
        best_p, best_s = None, -1.0
        for p in uniq:
            if p in active and not np.isnan(photo_year):
                lo, hi = active[p]
                if photo_year < lo or photo_year > hi:
                    continue  # person couldn't appear at this photo's date
            m = masks[p]
            psims = sims[m]
            if p in births and not np.isnan(photo_year):
                target_age = photo_year - births[p]
                proto_ages = pyears[m] - births[p]
                win = np.abs(proto_ages - target_age) <= AGE_WINDOW
                s = float(psims[win].max()) if win.any() else float(psims.max())
            else:
                s = float(psims.max())
            if s > best_s:
                best_p, best_s = p, s
        return best_p, best_s

    out: dict[int, tuple[float, list[str]]] = {}
    for mid in media_ids:
        present: dict[str, float] = {}
        for r in per_media.get(mid, []):
            e = blob_to_emb(r["embedding"])
            person, s = best_person(e, _decimal_year(r["capture_dt"]))
            if person is not None and s >= threshold:
                present[person] = max(present.get(person, 0.0), min(1.0, (r["area"] ** 0.5) * 3))
        if present:
            base = min(1.0, 0.6 + 0.2 * (len(present) - 1))   # 1 person .6, 2 .8, 3+ -> 1.0
            prom = max(present.values())
            out[mid] = (min(1.0, base * 0.6 + prom * 0.4), sorted(present))
        else:
            out[mid] = (0.0, [])
    return out
