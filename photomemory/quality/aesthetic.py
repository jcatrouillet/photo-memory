"""Optional ML aesthetic scoring via pyiqa's NIMA model (GPU if available).

Lazily loaded and entirely optional: if torch/pyiqa aren't installed the rest of the
pipeline still runs on heuristics. ``aesthetic_score`` returns a value in [0, 1]
(NIMA's 1..10 mean opinion score, rescaled) or None when unavailable.
"""
from __future__ import annotations

import numpy as np

_model = None
_device = None
_unavailable = False


def available() -> bool:
    return _ensure_model() is not None


def _ensure_model():
    global _model, _device, _unavailable
    if _unavailable:
        return None
    if _model is not None:
        return _model
    try:
        import pyiqa
        import torch

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = pyiqa.create_metric("nima", device=_device)
        _model.eval()
        return _model
    except Exception:
        _unavailable = True
        return None


def device() -> str | None:
    return _device


def aesthetic_score(rgb: np.ndarray) -> float | None:
    """Score an HxWx3 uint8 RGB array. Returns [0,1] or None if ML unavailable."""
    model = _ensure_model()
    if model is None:
        return None
    try:
        import torch

        t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
        t = t.to(_device)
        with torch.no_grad():
            score = float(model(t).item())  # NIMA mean opinion score, ~1..10
        return max(0.0, min(1.0, score / 10.0))
    except Exception:
        return None
