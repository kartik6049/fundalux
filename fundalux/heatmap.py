"""Heatmap drawing (no PyTorch): resizing and overlays for Grad-CAM maps in [0, 1]."""

from __future__ import annotations

import cv2
import numpy as np


def resize_cam(cam: np.ndarray, size: int) -> np.ndarray:
    return np.clip(cv2.resize(cam.astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC), 0.0, 1.0)


def overlay_heatmap(img: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Classic JET overlay (used in the training figures); weak activations stay transparent."""
    cam = resize_cam(cam, img.shape[0]) if cam.shape[:2] != img.shape[:2] else cam
    colour = cv2.cvtColor(cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    w = (alpha * cam)[..., None]
    return np.clip(img * (1 - w) + colour * w, 0, 255).astype(np.uint8)


def _lut(stops: list[tuple[float, str]]) -> np.ndarray:
    xs = np.linspace(0, 1, 256)
    rgb = [tuple(int(h[i : i + 2], 16) for i in (1, 3, 5)) for _, h in stops]
    pos = [p for p, _ in stops]
    return np.stack([np.interp(xs, pos, [c[k] for c in rgb]) for k in range(3)], axis=1)


# Blue never occurs in a fundus photograph, so the heat cannot be mistaken for a lesion
# (a white or yellow core could pass for an exudate). Powder blue rising to cobalt.
_HEAT = _lut([(0.0, "#8DB9F0"), (1.0, "#2350B8")])


def palette_overlay(img: np.ndarray, cam: np.ndarray, alpha: float = 0.62) -> np.ndarray:
    """Grad-CAM over the image in the app palette, with a contour at half the peak."""
    if cam.shape[:2] != img.shape[:2]:
        cam = cv2.resize(cam.astype(np.float32), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    cam = np.clip(cam, 0, 1)
    colour = _HEAT[(cam * 255).astype(np.uint8)]
    w = (alpha * np.clip((cam - 0.15) / 0.85, 0, 1) ** 0.8)[..., None]
    out = np.clip(img * (1 - w) + colour * w, 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours((cam >= 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (233, 232, 248), max(1, img.shape[0] // 256), lineType=cv2.LINE_AA)
    return out
