"""Image-quality gate helpers: cheap rule checks and human-readable hints.

The learned part of the gate is a FundusNet with a single "ungradable" logit;
this module holds the parts that need no training.
"""

from __future__ import annotations

import cv2
import numpy as np

from .preprocess import CropGeometry, disc_mask, retina_mask

_STATS_SIZE = 256


def quality_stats(cropped: np.ndarray) -> dict:
    """Simple image statistics of a cropped fundus image, at a fixed scale."""
    small = cv2.resize(cropped, (_STATS_SIZE, _STATS_SIZE), interpolation=cv2.INTER_AREA)
    disc = disc_mask(_STATS_SIZE)
    valid = retina_mask(small) > 0
    stats = {"retina_fraction": float(valid.sum() / disc.sum())}
    if valid.sum() < 50:
        stats.update(brightness=0.0, contrast=0.0, sharpness=0.0, saturated_fraction=0.0, red_minus_blue=0.0)
        return stats
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # Stay away from the disc edge, whose hard step would dominate the Laplacian.
    inner = cv2.erode(valid.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    means = small[valid].mean(axis=0)
    stats.update(
        brightness=float(gray[valid].mean()),
        contrast=float(gray[valid].std()),
        sharpness=float(lap[inner].var()) if inner.any() else 0.0,
        saturated_fraction=float((gray[valid] >= 250).mean()),
        red_minus_blue=float(means[0] - means[2]),
    )
    return stats


def basic_checks(image: np.ndarray, geom: CropGeometry, stats: dict, min_side: int = 224) -> list[str]:
    """Hard failures that need no model. Returns a list of reasons (empty = pass)."""
    reasons = []
    if min(image.shape[:2]) < min_side:
        reasons.append(f"Image is too small ({image.shape[1]}x{image.shape[0]} px).")
    if not geom.found:
        reasons.append("No retina found - the image is too dark or is not a fundus photograph.")
    elif stats["retina_fraction"] < 0.5:
        reasons.append("Less than half of the retina is visible.")
    elif stats["red_minus_blue"] <= 0:
        reasons.append("This does not look like a colour fundus photograph.")
    return reasons


def quality_hints(stats: dict, ranges: dict | None) -> list[str]:
    """Explain a rejection by comparing statistics with those of good images.

    `ranges` maps a statistic to the [low, high] interval that covers nearly all
    good-quality training images (measured in the training notebook).
    """
    if not ranges:
        return []
    hints = []

    def below(key):
        return key in ranges and stats[key] < ranges[key][0]

    def above(key):
        return key in ranges and stats[key] > ranges[key][1]

    if below("sharpness"):
        hints.append("Looks blurred - refocus and hold the camera steady.")
    if below("brightness"):
        hints.append("Too dark - check pupil alignment and flash/illumination.")
    if above("brightness") or above("saturated_fraction"):
        hints.append("Over-exposed or glare - reduce illumination and avoid reflections.")
    if below("contrast"):
        hints.append("Low contrast - clean the lens; a cataract or other media opacity can also cause this.")
    return hints
