"""SQLite catalog: schema, connection helper, and upsert/query helpers.

Three tables:
  folders  — one row per directory that holds media (carries parsed event title + dates)
  media    — one row per media file (cheap metadata + GPS, populated by the scanner)
  scores   — lazy per-image quality scores (phash, sharpness, aesthetic, faces), cached

The catalog lives on a local disk (see config.paths.db_path), never on the NAS.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .config import get_config

SCHEMA = """
CREATE TABLE IF NOT EXISTS folders (
    id          INTEGER PRIMARY KEY,
    path        TEXT UNIQUE NOT NULL,
    source      TEXT,                 -- root name: triees | amazon
    title       TEXT,                 -- folder name minus date prefix (event title)
    date_start  TEXT,                 -- ISO date (folder's earliest dated media / folder name date)
    date_end    TEXT,
    year        INTEGER,
    month       INTEGER
);

CREATE TABLE IF NOT EXISTS media (
    id           INTEGER PRIMARY KEY,
    path         TEXT UNIQUE NOT NULL,
    folder_id    INTEGER REFERENCES folders(id),
    capture_dt   TEXT,                -- ISO datetime
    date_source  TEXT,               -- path | filename | exif | mtime
    gps_lat      REAL,
    gps_lon      REAL,
    gps_alt      REAL,
    place        TEXT,               -- reverse-geocoded label (city/region/country)
    media_type   TEXT,               -- image | video
    width        INTEGER,
    height       INTEGER,
    file_size    INTEGER,
    mtime        REAL,
    indexed_at   TEXT
);

CREATE TABLE IF NOT EXISTS scores (
    media_id    INTEGER PRIMARY KEY REFERENCES media(id) ON DELETE CASCADE,
    phash       TEXT,
    sharpness   REAL,
    exposure    REAL,
    aesthetic   REAL,
    face_count  INTEGER,
    face_area   REAL,
    composite   REAL,
    scored_at   TEXT
);

CREATE TABLE IF NOT EXISTS faces (
    id          INTEGER PRIMARY KEY,
    media_id    INTEGER REFERENCES media(id) ON DELETE CASCADE,
    x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER,
    area        REAL,               -- face box area / image area
    prob        REAL,               -- detection confidence
    embedding   BLOB,               -- float32[512], L2-normalized
    cluster     INTEGER,            -- cluster id from `faces cluster`
    person      TEXT,               -- assigned name (after labeling)
    proto_date  TEXT                -- source photo date for reference protos (media_id NULL)
);

CREATE TABLE IF NOT EXISTS person_birth (
    name         TEXT PRIMARY KEY,
    birthdate    TEXT,              -- ISO date; enables age-based sibling disambiguation
    active_from  TEXT,              -- earliest date this person can appear (defaults to birthdate)
    active_until TEXT               -- latest date (e.g. deceased); NULL = no upper bound
);

CREATE TABLE IF NOT EXISTS people (
    name        TEXT PRIMARY KEY,
    embedding   BLOB,               -- centroid float32[512], L2-normalized
    n_faces     INTEGER,
    source      TEXT DEFAULT 'cluster'  -- 'cluster' (labeled) or 'reference' (example photos)
);

CREATE TABLE IF NOT EXISTS faces_scanned (
    media_id    INTEGER PRIMARY KEY REFERENCES media(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_media_capture ON media(capture_dt);
CREATE INDEX IF NOT EXISTS idx_media_folder  ON media(folder_id);
CREATE INDEX IF NOT EXISTS idx_media_type    ON media(media_type);
CREATE INDEX IF NOT EXISTS idx_media_gps     ON media(gps_lat, gps_lon);
CREATE INDEX IF NOT EXISTS idx_faces_media   ON faces(media_id);
CREATE INDEX IF NOT EXISTS idx_faces_person  ON faces(person);
CREATE INDEX IF NOT EXISTS idx_faces_cluster ON faces(cluster);
"""


@contextmanager
def connect(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    path = db_path or get_config().db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA busy_timeout=30000;")  # wait out concurrent writers
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path | None = None) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
        # Lightweight migration: add people.source to pre-existing tables.
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(people)")}
        if "source" not in cols:
            conn.execute("ALTER TABLE people ADD COLUMN source TEXT DEFAULT 'cluster'")
        fcols = {r["name"] for r in conn.execute("PRAGMA table_info(faces)")}
        if "proto_date" not in fcols:
            conn.execute("ALTER TABLE faces ADD COLUMN proto_date TEXT")
        bcols = {r["name"] for r in conn.execute("PRAGMA table_info(person_birth)")}
        for col in ("active_from", "active_until"):
            if col not in bcols:
                conn.execute(f"ALTER TABLE person_birth ADD COLUMN {col} TEXT")


def get_folder_id(conn: sqlite3.Connection, path: str) -> int | None:
    row = conn.execute("SELECT id FROM folders WHERE path = ?", (path,)).fetchone()
    return row["id"] if row else None


def upsert_folder(
    conn: sqlite3.Connection,
    *,
    path: str,
    source: str,
    title: str | None,
    date_start: str | None,
    date_end: str | None,
    year: int | None,
    month: int | None,
) -> int:
    conn.execute(
        """
        INSERT INTO folders (path, source, title, date_start, date_end, year, month)
        VALUES (:path, :source, :title, :date_start, :date_end, :year, :month)
        ON CONFLICT(path) DO UPDATE SET
            source=excluded.source, title=excluded.title,
            date_start=excluded.date_start, date_end=excluded.date_end,
            year=excluded.year, month=excluded.month
        """,
        dict(path=path, source=source, title=title, date_start=date_start,
             date_end=date_end, year=year, month=month),
    )
    return get_folder_id(conn, path)  # type: ignore[return-value]


def get_media_signature(conn: sqlite3.Connection, path: str) -> tuple[float, int] | None:
    """Return (mtime, file_size) of an existing row, for incremental skip."""
    row = conn.execute(
        "SELECT mtime, file_size FROM media WHERE path = ?", (path,)
    ).fetchone()
    return (row["mtime"], row["file_size"]) if row else None


def upsert_media(conn: sqlite3.Connection, rec: dict) -> None:
    conn.execute(
        """
        INSERT INTO media (path, folder_id, capture_dt, date_source, gps_lat, gps_lon,
                           gps_alt, place, media_type, width, height, file_size, mtime, indexed_at)
        VALUES (:path, :folder_id, :capture_dt, :date_source, :gps_lat, :gps_lon,
                :gps_alt, :place, :media_type, :width, :height, :file_size, :mtime, :indexed_at)
        ON CONFLICT(path) DO UPDATE SET
            folder_id=excluded.folder_id, capture_dt=excluded.capture_dt,
            date_source=excluded.date_source, gps_lat=excluded.gps_lat,
            gps_lon=excluded.gps_lon, gps_alt=excluded.gps_alt, place=excluded.place,
            media_type=excluded.media_type, width=excluded.width, height=excluded.height,
            file_size=excluded.file_size, mtime=excluded.mtime, indexed_at=excluded.indexed_at
        """,
        rec,
    )


def get_score(conn: sqlite3.Connection, media_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM scores WHERE media_id = ?", (media_id,)).fetchone()


def upsert_score(conn: sqlite3.Connection, rec: dict) -> None:
    conn.execute(
        """
        INSERT INTO scores (media_id, phash, sharpness, exposure, aesthetic,
                            face_count, face_area, composite, scored_at)
        VALUES (:media_id, :phash, :sharpness, :exposure, :aesthetic,
                :face_count, :face_area, :composite, :scored_at)
        ON CONFLICT(media_id) DO UPDATE SET
            phash=excluded.phash, sharpness=excluded.sharpness, exposure=excluded.exposure,
            aesthetic=excluded.aesthetic, face_count=excluded.face_count,
            face_area=excluded.face_area, composite=excluded.composite,
            scored_at=excluded.scored_at
        """,
        rec,
    )
