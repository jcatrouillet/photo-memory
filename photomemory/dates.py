"""Date + title parsing from folder names and file names.

Resolution order (cheap -> expensive) is orchestrated by the scanner; this module
provides the individual parsers.

Folder name formats seen in the library:
    20240615 Lassen Volcanic National Park   -> 2024-06-15, "Lassen Volcanic National Park"
    2005_01_03 Alexis                         -> 2005-01-03, "Alexis"
    20100107                                  -> 2010-01-07, ""  (no title)
    Famille                                   -> no date, "Famille"

File name formats:
    2024-06-15_06-55-26_000.jpeg              -> 2024-06-15 06:55:26
    IMG_20240615_065526.jpg                   -> 2024-06-15 06:55:26
"""
from __future__ import annotations

import re
from datetime import date, datetime

# --- Folder name parsing -------------------------------------------------------

# YYYYMMDD or YYYY_MM_DD or YYYY-MM-DD at the start, optionally followed by a title.
_FOLDER_DATE = re.compile(
    r"^(?P<y>\d{4})[_-]?(?P<m>\d{2})[_-]?(?P<d>\d{2})\b\s*(?P<title>.*)$"
)
# Year-only folder (e.g. "2024").
_YEAR_ONLY = re.compile(r"^(?P<y>\d{4})$")


def parse_folder(name: str) -> tuple[date | None, str | None]:
    """Return (date, title) parsed from a folder name. Either may be None."""
    name = name.strip()
    m = _FOLDER_DATE.match(name)
    if m:
        try:
            d = date(int(m["y"]), int(m["m"]), int(m["d"]))
        except ValueError:
            d = None
        title = (m["title"] or "").strip() or None
        return d, title
    if _YEAR_ONLY.match(name):
        return None, None  # a bare year folder is a container, not an event
    return None, name or None


# --- File name parsing ---------------------------------------------------------

_FILE_PATTERNS = [
    # 2024-06-15_06-55-26_000  /  2024-06-15_06-55-26
    re.compile(r"(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})[_ ](?P<H>\d{2})-(?P<M>\d{2})-(?P<S>\d{2})"),
    # 20240615_065526  (IMG_20240615_065526, PXL_20240615_065526123)
    re.compile(r"(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})[_-](?P<H>\d{2})(?P<M>\d{2})(?P<S>\d{2})"),
    # 2024-06-15 (date only)
    re.compile(r"(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})"),
]


def parse_filename(name: str) -> datetime | None:
    """Return a datetime parsed from a file name, or None."""
    for pat in _FILE_PATTERNS:
        m = pat.search(name)
        if not m:
            continue
        g = m.groupdict()
        try:
            return datetime(
                int(g["y"]), int(g["m"]), int(g["d"]),
                int(g.get("H") or 0), int(g.get("M") or 0), int(g.get("S") or 0),
            )
        except (ValueError, TypeError):
            continue
    return None


def parse_exif_datetime(value: str) -> datetime | None:
    """Parse an EXIF DateTimeOriginal string like '2024:06:15 06:55:26'."""
    if not value:
        return None
    value = value.strip().rstrip("\x00")
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None
