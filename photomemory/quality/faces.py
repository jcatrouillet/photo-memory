"""Face detection (count + relative area) used to favor good shots of people.

Uses OpenCV's bundled Haar cascade — zero downloads, CPU, fast on small frames.
It's only a weak signal (a bonus), so precision matters less than being free.
"""
from __future__ import annotations

import threading

import cv2
import numpy as np

# CascadeClassifier.detectMultiScale isn't guaranteed thread-safe on a shared object,
# so keep one classifier per thread (scoring loads images in a thread pool).
_local = threading.local()


def _get_cascade():
    c = getattr(_local, "cascade", None)
    if c is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        c = cv2.CascadeClassifier(path)
        _local.cascade = c
    return c


def detect_faces(rgb: np.ndarray) -> tuple[int, float]:
    """Return (face_count, fraction_of_frame_covered_by_faces)."""
    try:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        faces = _get_cascade().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                                minSize=(24, 24))
        if len(faces) == 0:
            return 0, 0.0
        h, w = gray.shape[:2]
        area = sum(fw * fh for (_, _, fw, fh) in faces) / float(w * h)
        return len(faces), min(1.0, area)
    except Exception:
        return 0, 0.0
