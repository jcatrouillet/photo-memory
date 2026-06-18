"""Configuration loading: defaults merged with an optional config.toml.

Access the singleton via ``get_config()``. Paths are resolved to absolute, relative
to the project root (the directory containing config.toml / this package's parent).
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

# Project root = parent of the photomemory package directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = PROJECT_ROOT / "config.toml"

DEFAULTS: dict[str, Any] = {
    "paths": {
        "roots": [
            {"name": "triees", "path": "Z:\\Triees", "layout": "foldername"},
            {"name": "amazon", "path": "Z:\\Amazon Photos Downloads", "layout": "filename"},
        ],
        "db_path": "data/catalog.db",
        "output_dir": "output",
        "cache_dir": "data/cache",
    },
    "scan": {
        "workers": 16,
        "read_exif_gps": True,
        "image_exts": [".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff"],
        "video_exts": [".mp4", ".mov", ".m4v"],
    },
    "scoring": {
        "w_sharpness": 0.30,
        "w_exposure": 0.15,
        "w_aesthetic": 0.40,
        "w_faces": 0.15,
        "use_ml": True,
        "face_bonus_cap": 3,
        "w_family": 0.6,   # strong boost for photos featuring enrolled family members
    },
    "grouping": {
        "trip_max_gap_days": 1,
        "trip_min_span_days": 2,
        "home_radius_km": 50,
    },
    "home": {"lat": None, "lon": None},
    "render": {
        "width": 1920,
        "height": 1080,
        "fps": 30,
        "seconds_per_photo": 3.0,
        "transition_seconds": 0.6,
        "default_length_seconds": 90,
        "ken_burns_zoom": 0.04,   # gentle; whole photo fits with a blurred fill behind
        "music": True,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class Root:
    name: str
    path: Path
    layout: str  # "foldername" or "filename"


@dataclass
class Config:
    raw: dict[str, Any]
    roots: list[Root] = field(default_factory=list)

    def _resolve(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (PROJECT_ROOT / path)

    @property
    def db_path(self) -> Path:
        return self._resolve(self.raw["paths"]["db_path"])

    @property
    def output_dir(self) -> Path:
        return self._resolve(self.raw["paths"]["output_dir"])

    @property
    def cache_dir(self) -> Path:
        return self._resolve(self.raw["paths"]["cache_dir"])

    @property
    def scan(self) -> dict[str, Any]:
        return self.raw["scan"]

    @property
    def scoring(self) -> dict[str, Any]:
        return self.raw["scoring"]

    @property
    def grouping(self) -> dict[str, Any]:
        return self.raw["grouping"]

    @property
    def home(self) -> dict[str, Any]:
        return self.raw["home"]

    @property
    def render(self) -> dict[str, Any]:
        return self.raw["render"]

    @property
    def image_exts(self) -> set[str]:
        return {e.lower() for e in self.scan["image_exts"]}

    @property
    def video_exts(self) -> set[str]:
        return {e.lower() for e in self.scan["video_exts"]}


@lru_cache(maxsize=1)
def get_config() -> Config:
    raw = DEFAULTS
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "rb") as f:
            user = tomllib.load(f)
        raw = _deep_merge(DEFAULTS, user)
    cfg = Config(raw=raw)
    cfg.roots = [
        Root(name=r["name"], path=Path(r["path"]), layout=r.get("layout", "foldername"))
        for r in raw["paths"]["roots"]
    ]
    # Ensure local dirs exist.
    for d in (cfg.db_path.parent, cfg.output_dir, cfg.cache_dir):
        d.mkdir(parents=True, exist_ok=True)
    return cfg
