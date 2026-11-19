"""PyTorch datasets over the cache of cropped fundus images.

They live in the package (not the notebook) so DataLoader worker processes can
import them on any platform.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .degrade import MILD, SEVERE, degrade, random_degradation
from .models import to_tensor
from .preprocess import enhance_fundus, read_image


def worker_init(worker_id: int) -> None:
    """Keep OpenCV/torch from oversubscribing the CPU inside DataLoader workers."""
    cv2.setNumThreads(0)
    torch.set_num_threads(1)


def random_geometry(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Flip, rotate by any angle, small zoom and shift (the retina has no fixed 'up')."""
    size = img.shape[0]
    if rng.random() < 0.5:
        img = img[:, ::-1]
    c = (size - 1) / 2
    M = cv2.getRotationMatrix2D((c, c), float(rng.uniform(0, 360)), float(rng.uniform(0.9, 1.15)))
    M[:, 2] += rng.uniform(-0.03, 0.03, 2) * size
    return cv2.warpAffine(np.ascontiguousarray(img), M, (size, size), flags=cv2.INTER_LINEAR, borderValue=0)


def random_colour(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Per-channel gain and gamma jitter, imitating different cameras."""
    gamma = rng.uniform(0.8, 1.25)
    ramp = (np.arange(256, dtype=np.float32) / 255.0) ** gamma * 255.0
    out = np.empty_like(img)
    for ch, gain in enumerate(rng.uniform(0.9, 1.1, 3)):
        out[..., ch] = cv2.LUT(img[..., ch], np.clip(ramp * gain, 0, 255).astype(np.uint8))
    return out


class _CachedFundus(Dataset):
    """Shared loading and random-number handling.

    `fixed_degradation=(kind, severity)` applies the same degradation to every
    image; it is used to measure robustness, never for training.
    """

    def __init__(self, paths, size: int, train: bool, seed: int, fixed_degradation=None):
        self.paths = [str(p) for p in paths]
        self.size = size
        self.train = train
        self.seed = seed
        self.fixed_degradation = fixed_degradation
        self._train_rng = None

    def __len__(self) -> int:
        return len(self.paths)

    def _rng(self, index: int) -> np.random.Generator:
        if not self.train:  # evaluation is reproducible per image
            return np.random.default_rng((self.seed, index))
        if self._train_rng is None:  # one stream per worker process, reseeded each epoch by torch
            self._train_rng = np.random.default_rng(torch.initial_seed() % (2**32))
        return self._train_rng

    def _load(self, index: int) -> np.ndarray:
        img = read_image(self.paths[index])
        if img.shape[0] != self.size or img.shape[1] != self.size:
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return img


class DRDataset(_CachedFundus):
    """Yields (enhanced image tensor, DR grade)."""

    def __init__(self, paths, grades, size: int, train: bool = False, degrade_p: float = 0.4, seed: int = 0, fixed_degradation=None):
        super().__init__(paths, size, train, seed, fixed_degradation)
        self.grades = np.asarray(grades, dtype=np.int64)
        self.degrade_p = degrade_p

    def __getitem__(self, index: int):
        rng = self._rng(index)
        img = self._load(index)
        if self.train:
            img = random_colour(random_geometry(img, rng), rng)
            if rng.random() < self.degrade_p:
                img, _, _ = random_degradation(img, rng, MILD)
        if self.fixed_degradation:
            img = degrade(img, *self.fixed_degradation, rng)
        return to_tensor(enhance_fundus(img)), int(self.grades[index])


class QualityDataset(_CachedFundus):
    """Yields (image tensor, 1.0 if ungradable else 0.0) for the quality gate.

    Images are *not* enhanced: CLAHE would hide the very defects the gate must see.
    Gradable images flagged `clean` are turned into synthetic ungradable ones with
    probability `severe_p`; any gradable image gets a mild, still-gradable
    degradation with probability `mild_p` so the gate does not over-reject.
    """

    def __init__(self, paths, labels, clean, size: int, input_size: int, train: bool = False, severe_p: float = 0.25, mild_p: float = 0.25, seed: int = 0, fixed_degradation=None,
                 gradable_range=MILD, ungradable_range=SEVERE):
        super().__init__(paths, size, train, seed, fixed_degradation)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.clean = np.asarray(clean, dtype=bool)
        self.input_size = input_size
        self.severe_p = severe_p
        self.mild_p = mild_p
        self.gradable_range, self.ungradable_range = gradable_range, ungradable_range

    def __getitem__(self, index: int):
        rng = self._rng(index)
        img = self._load(index)
        label = float(self.labels[index])
        if self.train:
            img = random_colour(random_geometry(img, rng), rng)
        if label == 0 and not self.fixed_degradation:
            u = rng.random()
            if self.clean[index] and u < self.severe_p:
                img, _, _ = random_degradation(img, rng, self.ungradable_range)
                label = 1.0
            elif u < self.severe_p + self.mild_p:
                img, _, _ = random_degradation(img, rng, self.gradable_range)
        if self.fixed_degradation:
            img = degrade(img, *self.fixed_degradation, rng)
        img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        return to_tensor(img), label

