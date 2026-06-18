"""EXIF reading: capture datetime + GPS coordinates.

Uses Pillow (with pillow-heif registered for HEIC) and falls back to piexif for the
GPS IFD. Returns plain Python values so callers stay decoupled from imaging libs.
All failures are swallowed and reported as None — EXIF is best-effort.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .dates import parse_exif_datetime

try:  # HEIC/HEIF support (optional but recommended)
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover - optional dependency
    pass

from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS


def _to_degrees(value) -> float | None:
    """Convert EXIF GPS rational triplet (deg, min, sec) to decimal degrees."""
    try:
        d, m, s = value
        return float(d) + float(m) / 60.0 + float(s) / 3600.0
    except Exception:
        return None


def read_exif(path: Path) -> dict:
    """Return {capture_dt, gps_lat, gps_lon, gps_alt, width, height} (all optional)."""
    out = {"capture_dt": None, "gps_lat": None, "gps_lon": None, "gps_alt": None,
           "width": None, "height": None}
    try:
        with Image.open(path) as img:
            out["width"], out["height"] = img.width, img.height
            exif = img.getexif()
            if not exif:
                return out

            # DateTimeOriginal lives in the Exif sub-IFD (0x8769); DateTime in base IFD.
            dt_str = None
            try:
                sub = exif.get_ifd(0x8769)
                for tag_id, val in sub.items():
                    if TAGS.get(tag_id) == "DateTimeOriginal":
                        dt_str = val
                        break
            except Exception:
                pass
            if not dt_str:
                for tag_id, val in exif.items():
                    if TAGS.get(tag_id) in ("DateTimeOriginal", "DateTime"):
                        dt_str = val
                        break
            if isinstance(dt_str, str):
                dt = parse_exif_datetime(dt_str)
                if dt:
                    out["capture_dt"] = dt

            # GPS IFD (0x8825).
            try:
                gps = exif.get_ifd(0x8825)
            except Exception:
                gps = None
            if gps:
                g = {GPSTAGS.get(k, k): v for k, v in gps.items()}
                lat = _to_degrees(g.get("GPSLatitude"))
                lon = _to_degrees(g.get("GPSLongitude"))
                if lat is not None and g.get("GPSLatitudeRef") in ("S", "s"):
                    lat = -lat
                if lon is not None and g.get("GPSLongitudeRef") in ("W", "w"):
                    lon = -lon
                out["gps_lat"] = lat
                out["gps_lon"] = lon
                alt = g.get("GPSAltitude")
                if alt is not None:
                    try:
                        alt = float(alt)
                        if g.get("GPSAltitudeRef") in (1, b"\x01"):
                            alt = -alt
                        out["gps_alt"] = alt
                    except Exception:
                        pass
    except Exception:
        return out
    return out


def read_dimensions(path: Path) -> tuple[int | None, int | None]:
    try:
        with Image.open(path) as img:
            return img.width, img.height
    except Exception:
        return None, None
