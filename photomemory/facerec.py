"""Face detection + recognition embeddings (GPU) via facenet-pytorch.

MTCNN locates faces; InceptionResnetV1 (VGGFace2) turns each into a 512-d, L2-normalized
embedding suitable for cosine-similarity matching/clustering. Models are loaded lazily and
run on CUDA when available. Operates on already-decoded PIL images (the cached render
frames), so no NAS reads are needed at this stage.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

_device = None
_mtcnn = None
_resnet = None
_unavailable = False

MIN_PROB = 0.92        # detection confidence floor
MIN_AREA_FRAC = 0.004  # ignore tiny background faces


@dataclass
class Face:
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2
    prob: float
    area_frac: float                  # face box area / image area
    embedding: np.ndarray             # float32[512], L2-normalized


def available() -> bool:
    return _ensure() is not None


def device() -> str | None:
    return _device


def _ensure():
    global _device, _mtcnn, _resnet, _unavailable
    if _unavailable:
        return None
    if _resnet is not None:
        return _resnet
    try:
        import torch
        from facenet_pytorch import MTCNN, InceptionResnetV1

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _mtcnn = MTCNN(keep_all=True, post_process=True, select_largest=False, device=_device)
        _resnet = InceptionResnetV1(pretrained="vggface2").eval().to(_device)
        return _resnet
    except Exception:
        _unavailable = True
        return None


def detect_and_embed(img: Image.Image) -> list[Face]:
    resnet = _ensure()
    if resnet is None:
        return []
    try:
        import torch

        img = img.convert("RGB")
        W, H = img.size
        boxes, probs = _mtcnn.detect(img)
        if boxes is None:
            return []
        chips = _mtcnn.extract(img, boxes, save_path=None)  # tensor [n,3,160,160] or None
        if chips is None:
            return []
        if chips.ndim == 3:
            chips = chips.unsqueeze(0)
        with torch.no_grad():
            emb = resnet(chips.to(_device)).cpu().numpy()
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)

        out: list[Face] = []
        for i, box in enumerate(boxes):
            p = float(probs[i]) if probs[i] is not None else 0.0
            x1, y1, x2, y2 = (int(v) for v in box)
            area_frac = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1)) / float(W * H)
            if p < MIN_PROB or area_frac < MIN_AREA_FRAC:
                continue
            out.append(Face((x1, y1, x2, y2), p, area_frac, emb[i].astype(np.float32)))
        return out
    except Exception:
        return []


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # inputs are L2-normalized
