"""Field-condition degradations for portable fundus cameras.

Used three ways: mild versions as training augmentation for the DR classifier,
severe versions as synthetic "ungradable" examples for the quality gate, and
fixed-severity sweeps to measure robustness.

Every function takes a cropped RGB uint8 fundus image (before enhancement) and a
`severity` in [0, 1]; sizes are relative to the image so behaviour is the same
at any resolution.
"""

from __future__ import annotations

import cv2
import numpy as np

from .preprocess import channel_max, keep_where

KINDS = ("blur", "motion", "glare", "haze", "dark")

# Severity ranges. Mild images stay gradable; severe ones are treated as
# ungradable. The gap in between is deliberately never generated for training.
MILD = (0.05, 0.3)
SEVERE = (0.7, 1.0)


def _blur(img, severity, rng):
    return cv2.GaussianBlur(img, (0, 0), max(0.03 * severity**1.5 * img.shape[0], 0.1))


def _motion(img, severity, rng):
    length = max(3, int(round(0.12 * severity**1.5 * img.shape[0])) | 1)
    kernel = np.zeros((length, length), np.float32)
    kernel[length // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D(((length - 1) / 2, (length - 1) / 2), float(rng.uniform(0, 180)), 1.0)
    kernel = cv2.warpAffine(kernel, rot, (length, length))
    kernel /= max(float(kernel.sum()), 1e-6)
    return cv2.filter2D(img, -1, kernel)


def _glare(img, severity, rng):
    radius = 0.08 + 0.42 * severity  # as a fraction of the image width
    r, theta = 0.35 * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
    cx, cy = 0.5 + r * np.cos(theta), 0.5 + r * np.sin(theta)
    grid = (np.arange(64, dtype=np.float32) + 0.5) / 64  # the blob is smooth: build it small
    blob = np.exp(-((grid[None, :] - cx) ** 2 + (grid[:, None] - cy) ** 2) / (2 * radius**2))
    strength = 0.3 + 1.2 * severity
    veil = np.clip(255.0 * strength * cv2.resize(blob, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR), 0, 255).astype(np.uint8)
    return cv2.add(img, cv2.merge([veil, veil, veil]))


def _haze(img, severity, rng):
    # Media opacity / dirty lens: light scatters, contrast collapses towards a bright veil.
    keep = 1.0 - 0.95 * severity
    soft = cv2.GaussianBlur(img, (0, 0), max(0.012 * severity * img.shape[0], 0.1))
    veil = np.empty_like(img)
    veil[:] = (200, 185, 170)
    return cv2.addWeighted(soft, keep, veil, 1.0 - keep, 0.0)


def _dark(img, severity, rng):
    # Under-exposure: signal shrinks while sensor noise does not.
    noise = rng.standard_normal(img.shape[:2], dtype=np.float32) * (1.0 + 3.0 * severity)
    out = img.astype(np.float32) * (0.02**severity) + noise[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


_FUNCS = {"blur": _blur, "motion": _motion, "glare": _glare, "haze": _haze, "dark": _dark}


def degrade(img: np.ndarray, kind: str, severity: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Apply one degradation. The black area outside the fundus stays black."""
    if severity <= 0:
        return img.copy()
    rng = rng or np.random.default_rng(0)
    out = _FUNCS[kind](img, float(min(severity, 1.0)), rng)
    return keep_where(out, cv2.threshold(channel_max(img), 8, 255, cv2.THRESH_BINARY)[1])


def random_degradation(img: np.ndarray, rng: np.random.Generator, severity_range=MILD, kinds=KINDS):
    """Apply a randomly chosen degradation. Returns (image, kind, severity)."""
    kind = kinds[int(rng.integers(len(kinds)))]
    severity = float(rng.uniform(*severity_range))
    return degrade(img, kind, severity, rng), kind, severity
