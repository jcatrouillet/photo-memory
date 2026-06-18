"""Cheap, dependency-light image-quality heuristics computed on a small RGB array.

All functions take an HxWx3 uint8 RGB numpy array and return a float in [0, 1]
(except raw sharpness variance, which is also returned for transparency).
"""
from __future__ import annotations

import cv2
import numpy as np

# Laplacian variance at/above this is considered fully "sharp".
SHARPNESS_FULL = 200.0
IDEAL_BRIGHTNESS = 0.46  # mid-ish, slightly bright reads well


def sharpness(rgb: np.ndarray) -> tuple[float, float]:
    """Return (normalized_0_1, raw_laplacian_variance)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return min(1.0, var / SHARPNESS_FULL), var


def exposure(rgb: np.ndarray) -> float:
    """Reward well-exposed images; penalize darkness, blowouts, and heavy clipping."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    mean = float(gray.mean())
    brightness = 1.0 - min(1.0, abs(mean - IDEAL_BRIGHTNESS) / IDEAL_BRIGHTNESS)
    clip_low = float((gray < 0.02).mean())
    clip_high = float((gray > 0.98).mean())
    clip_penalty = min(0.5, clip_low + clip_high)
    return max(0.0, brightness - clip_penalty)
