"""End-to-end inference: quality gate -> DR classifier -> Grad-CAM -> advice."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

from . import GRADE_NAMES
from .heatmap import palette_overlay, resize_cam
from .inference import ensemble_probabilities, load_backend, to_input
from .preprocess import crop_fundus, model_input
from .quality import basic_checks, quality_hints, quality_stats

DISCLAIMER = "Screening aid only - not a diagnosis. A qualified clinician must confirm every result."


@dataclass
class Prediction:
    gradable: bool
    ungradable_score: float | None  # quality gate output, P(ungradable)
    reasons: list[str] = field(default_factory=list)  # why the image was rejected
    quality: dict = field(default_factory=dict)  # raw image statistics
    referable: bool | None = None
    referable_score: float | None = None  # P(referable DR)
    threshold: float | None = None
    site: str | None = None  # whose calibrated threshold was used; None = the bundle's global threshold
    grade: int | None = None
    grade_probs: list[float] | None = None
    member_scores: dict[str, float] | None = None  # P(referable DR) of each model in the ensemble
    advice: str = ""
    cropped: np.ndarray | None = None  # what the quality gate saw
    enhanced: np.ndarray | None = None  # what the DR classifier saw
    heatmap: np.ndarray | None = None  # Grad-CAM in [0, 1], same size as `enhanced`
    overlay: np.ndarray | None = None

    @property
    def grade_name(self) -> str | None:
        return None if self.grade is None else GRADE_NAMES[self.grade]

    def with_threshold(self, threshold: float, site: str | None = None) -> "Prediction":
        """The same analysis decided at another threshold, e.g. a site's - no re-scoring needed."""
        out = replace(self, threshold=float(threshold), site=site)
        if self.referable_score is not None:
            out.referable = self.referable_score >= out.threshold
            if self.gradable:
                out.advice = referral_advice(True, out.referable, self.grade)
        return out

    def summary(self) -> dict:
        """JSON-friendly result without the image arrays."""
        keys = ("gradable", "ungradable_score", "reasons", "referable", "referable_score", "threshold", "site", "grade", "grade_probs", "member_scores", "advice")
        out = {k: getattr(self, k) for k in keys}
        out["grade_name"] = self.grade_name
        return out


def referral_advice(gradable: bool, referable: bool | None, grade: int | None) -> str:
    if not gradable:
        return "Image rejected - retake the photograph. If it is still ungradable after two more attempts, refer the patient for an in-person eye examination."
    if referable:
        if grade is not None and grade >= 3:
            return "Referable DR suspected, possibly severe - refer to an ophthalmologist urgently."
        return "Referable DR suspected - refer to an ophthalmologist."
    return "No referable DR detected - advise routine re-screening in 12 months, sooner if vision changes."


class Fundalux:
    """Loads the trained models and screens one photograph at a time.

    The DR classifier is an ensemble: every member scores the image and their
    probabilities are averaged. A bundle with a single model is an ensemble of one.
    Models run in ONNX Runtime when the bundle has an ONNX export (python -m
    fundalux.export), otherwise in PyTorch.
    """

    def __init__(self, members, gate, config: dict, backend: str = ""):
        self.members, self.gate, self.config, self.backend = members, gate, config, backend
        self.dr_members = [(m.name, m, m.size) for m in members]  # (architecture, member, input size)
        self.image_size = int(config["image_size"])
        self.dr_threshold = float(config["dr"]["threshold"])
        self.quality_size = int(config["quality"]["input_size"])
        self.quality_threshold = float(config["quality"]["threshold"])
        self.hint_ranges = config["quality"].get("hint_ranges")
        self.min_input_side = int(config.get("min_input_side", 224))

    @classmethod
    def load(cls, model_dir, backend: str = "auto") -> "Fundalux":
        model_dir = Path(model_dir)
        config_path = model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"{config_path} not found. Train the models with the Kaggle notebook and unzip fundalux_outputs.zip into {model_dir}/ (see models/README.md)."
            )
        config = json.loads(config_path.read_text())
        name, members, gate = load_backend(model_dir, config, backend)
        return cls(members, gate, config, name)

    def ungradable_score(self, cropped: np.ndarray) -> float:
        small = cv2.resize(cropped, (self.quality_size, self.quality_size), interpolation=cv2.INTER_AREA)
        return self.gate.ungradable(to_input([small]))

    def _gate(self, image: np.ndarray) -> Prediction:
        """Crop the photograph and run the quality gate (rule checks, then the learned model)."""
        cropped, geom = crop_fundus(image, self.image_size)
        stats = quality_stats(cropped)
        reasons = basic_checks(image, geom, stats, self.min_input_side)
        score = self.ungradable_score(cropped) if geom.found else None
        if score is not None and score >= self.quality_threshold:
            reasons.append(f"Image quality is too poor to grade (ungradable score {score:.2f}, limit {self.quality_threshold:.2f}).")
        gradable = not reasons
        if not gradable:
            reasons.extend(quality_hints(stats, self.hint_ranges))
        return Prediction(gradable=gradable, ungradable_score=score, reasons=reasons, quality=stats, cropped=cropped)

    def _score(self, cropped: np.ndarray, with_cam: bool):
        """Run every member at its own input size. Returns (logits per member, averaged heatmap or None)."""
        logits, total = [], np.zeros((self.image_size, self.image_size), np.float32)
        for m in self.members:
            out, cams = m.run(to_input([model_input(cropped, m.size)]), with_cam)
            logits.append(out)
            if with_cam:
                total += resize_cam(cams[0], self.image_size)
        heatmap = None
        if with_cam:
            peak = float(total.max())
            heatmap = total / peak if peak > 0 else total
        return logits, heatmap

    def referable_score(self, image: np.ndarray) -> tuple[bool, float | None]:
        """(passes the quality gate, P(referable DR) or None) without Grad-CAM - for calibrating a site."""
        gate = self._gate(image)
        if not gate.gradable:
            return False, None
        logits, _ = self._score(gate.cropped, with_cam=False)
        return True, float(ensemble_probabilities(logits)[0][0])

    def predict(self, image: np.ndarray, skip_gate: bool = False, site=None) -> Prediction:
        """Screen one RGB uint8 photograph.

        `site` is a calibrated SiteProfile (see sites.py); its threshold replaces the
        bundle's global one. With `skip_gate` the classifier runs even on a rejected
        image (the result keeps gradable=False) - only for demonstrating why the gate exists.
        """
        result = self._gate(image)
        gradable, cropped = result.gradable, result.cropped
        if not gradable and not skip_gate:
            result.advice = referral_advice(False, None, None)
            return result

        logits, heatmap = self._score(cropped, with_cam=True)
        referable, grades, per_member = ensemble_probabilities(logits)
        result.enhanced = model_input(cropped, self.image_size)
        result.referable_score = float(referable[0])
        result.member_scores = {m.name: float(p[0]) for m, p in zip(self.members, per_member)}
        result.threshold = self.dr_threshold if site is None else float(site.threshold)
        result.site = None if site is None else site.name
        result.referable = result.referable_score >= result.threshold
        if grades is not None:
            result.grade_probs = [float(p) for p in grades[0]]
            result.grade = int(np.argmax(grades[0]))
        result.heatmap = heatmap
        result.overlay = palette_overlay(result.enhanced, heatmap)
        result.advice = referral_advice(gradable, result.referable, result.grade)
        return result
