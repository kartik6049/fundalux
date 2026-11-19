"""Fundus preprocessing: crop to the retina, illumination correction, CLAHE.

All functions take and return RGB uint8 arrays.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CropGeometry:
    """Square region of the source image that holds the fundus disc."""

    x0: int
    y0: int
    side: int
    found: bool  # False when no fundus disc was detected (centre crop used)


def decode_image(data: bytes) -> np.ndarray:
    """Decode encoded image bytes (JPEG/PNG/TIFF...) to RGB uint8."""
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        import io

        from PIL import Image

        return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_image(path) -> np.ndarray:
    with open(path, "rb") as f:
        return decode_image(f.read())


def channel_max(img: np.ndarray) -> np.ndarray:
    """Per-pixel maximum over R, G, B (much faster than ndarray.max(axis=2))."""
    r, g, b = cv2.split(img)
    return cv2.max(cv2.max(r, g), b)


def find_fundus(img: np.ndarray) -> CropGeometry:
    """Locate the fundus disc and return the square that encloses it.

    The disc is often cut off at the top and bottom of the frame, so the square
    side is the larger bounding-box dimension and may extend past the image.
    """
    h, w = img.shape[:2]
    scale = min(1.0, 256.0 / max(h, w))
    small = img
    if scale < 1.0:
        small = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    gray = channel_max(small)
    thr = max(10.0, 0.1 * float(np.percentile(gray, 95)))
    mask = (gray > thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, bw, bh, area = (int(v) for v in stats[i])
        if area >= 0.08 * mask.size and min(bw, bh) >= 0.3 * max(bw, bh):
            side = max(bw, bh) / scale
            cx, cy = (x + bw / 2) / scale, (y + bh / 2) / scale
            return CropGeometry(round(cx - side / 2), round(cy - side / 2), max(1, round(side)), True)
    side = min(h, w)
    return CropGeometry((w - side) // 2, (h - side) // 2, side, False)


def apply_crop(img: np.ndarray, geom: CropGeometry, size: int, interpolation: int | None = None) -> np.ndarray:
    """Cut the square described by `geom` (zero-padded if needed) and resize it."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = geom.x0, geom.y0, geom.x0 + geom.side, geom.y0 + geom.side
    crop = img[max(0, y0) : min(h, y1), max(0, x0) : min(w, x1)]
    pad = (max(0, -y0), max(0, y1 - h), max(0, -x0), max(0, x1 - w))
    if any(pad):
        crop = cv2.copyMakeBorder(crop, *pad, cv2.BORDER_CONSTANT, value=0)
    if interpolation is None:
        interpolation = cv2.INTER_AREA if geom.side > size else cv2.INTER_LINEAR
    return cv2.resize(crop, (size, size), interpolation=interpolation)


@functools.lru_cache(maxsize=8)
def _disc_u8(size: int) -> np.ndarray:
    c = (size - 1) / 2
    yy, xx = np.ogrid[:size, :size]
    mask = ((xx - c) ** 2 + (yy - c) ** 2 <= (0.985 * size / 2) ** 2).astype(np.uint8) * 255
    mask.setflags(write=False)
    return mask


def disc_mask(size: int) -> np.ndarray:
    """Boolean mask of the inscribed fundus disc for a `size` x `size` crop."""
    return _disc_u8(size) > 0


def keep_where(img: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    """Copy of `img` with every pixel outside the 0/255 mask set to black."""
    return cv2.bitwise_and(img, img, mask=mask_u8)


def crop_fundus(img: np.ndarray, size: int) -> tuple[np.ndarray, CropGeometry]:
    """Crop to the fundus disc, resize to `size` and black out everything outside it."""
    geom = find_fundus(img)
    return keep_where(apply_crop(img, geom, size), _disc_u8(size)), geom


def retina_mask(img: np.ndarray) -> np.ndarray:
    """0/255 mask of disc pixels that actually contain retina (the disc may be cut off)."""
    lit = cv2.threshold(channel_max(img), 8, 255, cv2.THRESH_BINARY)[1]
    return cv2.bitwise_and(lit, _disc_u8(img.shape[0]))


def _flatten_lightness(L: np.ndarray, valid: np.ndarray, target: float) -> np.ndarray:
    size = L.shape[0]
    Lf = L.astype(np.float32)
    v = valid.astype(np.float32) / 255.0
    # The illumination field is smooth, so estimate it on a small copy.
    s = min(size, 96)
    Ls = cv2.resize(Lf * v, (s, s), interpolation=cv2.INTER_AREA)
    vs = cv2.resize(v, (s, s), interpolation=cv2.INTER_AREA)
    sigma = 0.12 * s
    field = cv2.GaussianBlur(Ls, (0, 0), sigma) / np.maximum(cv2.GaussianBlur(vs, (0, 0), sigma), 1e-3)
    field = cv2.resize(field, (size, size), interpolation=cv2.INTER_LINEAR)
    gain = np.clip(target / np.maximum(field, 1.0), 0.25, 4.0)
    return np.clip(Lf * gain, 0, 255).astype(np.uint8)


def _clahe(L: np.ndarray, clip_limit: float, tile_grid: int) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid)).apply(L)


def _on_lightness(img: np.ndarray, fn) -> np.ndarray:
    valid = retina_mask(img)
    if cv2.countNonZero(valid) < 0.01 * valid.size:
        return img.copy()
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[..., 0] = fn(np.ascontiguousarray(lab[..., 0]), valid)
    return keep_where(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), valid)


def correct_illumination(img: np.ndarray, target: float = 120.0) -> np.ndarray:
    """Flatten uneven illumination on a cropped fundus image.

    Lightness is divided by its own large-scale blur (the illumination field),
    so dark rims, vignetting and under-exposure are evened out while colour is kept.
    """
    return _on_lightness(img, lambda L, valid: _flatten_lightness(L, valid, target))


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation on the lightness channel."""
    return _on_lightness(img, lambda L, valid: _clahe(L, clip_limit, tile_grid))


def enhance_fundus(cropped: np.ndarray, target: float = 120.0, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """Illumination correction followed by CLAHE, on an image from `crop_fundus`."""
    return _on_lightness(cropped, lambda L, valid: _clahe(_flatten_lightness(L, valid, target), clip_limit, tile_grid))


def model_input(cropped: np.ndarray, size: int) -> np.ndarray:
    """Classifier input from a cropped image: resize to the model's own input size, then enhance.

    Same order as training (the cached crop is resized first, enhanced last).
    """
    if cropped.shape[0] != size or cropped.shape[1] != size:
        cropped = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
    return enhance_fundus(cropped)


def preprocess_fundus(img: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray, CropGeometry]:
    """Full pipeline on a raw photograph. Returns (cropped, enhanced, geometry).

    `cropped` feeds the quality gate, `enhanced` feeds the DR classifier.
    """
    cropped, geom = crop_fundus(img, size)
    return cropped, enhance_fundus(cropped), geom
