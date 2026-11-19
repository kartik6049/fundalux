"""Per-site decision thresholds.

The classifier ranks images well at every site, but its scores sit on a
different scale for each camera, so one threshold does not transfer. A site
profile stores a threshold calibrated on graded images from that site, tuned
for the same target sensitivity as the global threshold.

Profiles live in <model dir>/sites.json. A threshold only means something for
the model that produced the scores, so the file records a fingerprint of the
model bundle; profiles from another bundle are kept but marked stale.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import REFERABLE_MIN_GRADE
from .metrics import crossfit_operating_point, sens_spec, threshold_for_sensitivity

SITES_FILE = "sites.json"
MIN_PER_CLASS = 20  # below this a site cannot be calibrated at all
RECOMMENDED_PER_CLASS = 50  # below this the profile carries a warning


class CalibrationError(ValueError):
    pass


@dataclass
class SiteProfile:
    name: str
    threshold: float
    target_sensitivity: float
    n_images: int  # gate-accepted images the threshold was fitted on
    n_referable: int
    expected_sensitivity: float  # on new images from this site, estimated without reusing the fitting images
    expected_specificity: float
    sensitivity_range: tuple[float, float] = (float("nan"), float("nan"))
    specificity_range: tuple[float, float] = (float("nan"), float("nan"))
    estimate: str = ""  # how the expected values were obtained
    source: str = ""  # "built-in" for profiles written with the model bundle
    rejected_by_gate: int = 0
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%d"))
    stale: bool = False  # fitted for a different model bundle

    @property
    def warnings(self) -> list[str]:
        out = []
        if min(self.n_referable, self.n_images - self.n_referable) < RECOMMENDED_PER_CLASS:
            out.append(f"Fewer than {RECOMMENDED_PER_CLASS} images in one class: the threshold is imprecise.")
        if self.stale:
            out.append("Calibrated for a different model - recalibrate before use.")
        return out


def model_fingerprint(config: dict) -> str:
    """Identifies the scoring models; a new training run gives a new fingerprint."""
    dr = config["dr"]
    members = dr.get("models") or [{"arch": dr["arch"], "weights": dr["weights"], "input_size": config["image_size"]}]
    key = json.dumps({"members": members, "trained": config.get("trained_with", {}), "threshold": dr["threshold"]}, sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def patient_groups(source: str, image_ids) -> np.ndarray:
    """Group both eyes of one patient where the file names allow it (EyePACS: 123_left / 123_right)."""
    ids = np.asarray(image_ids, dtype=str)
    if source == "eyepacs":
        return np.array([i.rsplit("_", 1)[0] for i in ids])
    return ids


def calibrate(name: str, referable, scores, target_sensitivity: float, groups=None, source: str = "", rejected_by_gate: int = 0) -> SiteProfile:
    """Fit a site threshold on gate-accepted images; estimate its accuracy by cross-fitting."""
    y = np.asarray(referable).astype(bool)
    s = np.asarray(scores, dtype=np.float64)
    if y.size != s.size:
        raise CalibrationError("Labels and scores have different lengths.")
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if min(n_pos, n_neg) < MIN_PER_CLASS:
        raise CalibrationError(
            f"Calibration needs at least {MIN_PER_CLASS} referable and {MIN_PER_CLASS} non-referable gradable images; got {n_pos} and {n_neg}."
        )
    estimate = crossfit_operating_point(y, s, target_sensitivity, groups=groups)
    return SiteProfile(
        name=name,
        threshold=threshold_for_sensitivity(y, s, target_sensitivity),
        target_sensitivity=target_sensitivity,
        n_images=int(y.size),
        n_referable=n_pos,
        expected_sensitivity=estimate["sensitivity"],
        expected_specificity=estimate["specificity"],
        sensitivity_range=estimate["sensitivity_range"],
        specificity_range=estimate["specificity_range"],
        estimate=f"cross-fitted: tuned on one half of the site's images, tested on the other ({estimate['repeats']} random splits)",
        source=source,
        rejected_by_gate=rejected_by_gate,
    )


def calibrate_on_split(name: str, fit: dict, test: dict, target_sensitivity: float, source: str = "") -> SiteProfile:
    """Fit on one set of images and measure on a separate one (used when a site has a validation split).

    `fit` and `test` are {"referable": ..., "scores": ...}.
    """
    profile = calibrate(name, fit["referable"], fit["scores"], target_sensitivity, source=source)
    sens, spec = sens_spec(test["referable"], test["scores"], profile.threshold)
    profile.expected_sensitivity, profile.expected_specificity = sens, spec
    profile.sensitivity_range = profile.specificity_range = (float("nan"), float("nan"))
    profile.estimate = f"tuned on {profile.n_images} validation images, measured on {len(test['scores'])} separate test images"
    return profile


# Display names for the datasets the bundle is evaluated on.
BUILTIN_NAMES = {
    "eyepacs": "EyePACS (USA, many clinics)",
    "aptos": "APTOS 2019 (Aravind, India)",
    "idrid": "IDRiD (Nanded, India)",
    "messidor": "Messidor-2 (France)",
}


def builtin_profiles(predictions, target_sensitivity: float) -> list[SiteProfile]:
    """Profiles for the evaluation datasets, from the bundle's per-image predictions.

    `predictions` needs columns source, image_id, split, grade, gate_pass, p_ref.
    Sources with a validation split are tuned on it and measured on their test
    split; external sources are tuned on all their images and measured by cross-fitting.
    """
    df = predictions[(predictions["grade"] >= 0) & predictions["gate_pass"].astype(bool) & predictions["p_ref"].notna()]
    profiles = []
    for source, name in BUILTIN_NAMES.items():
        part = df[df["source"] == source]
        if part.empty:
            continue
        referable = (part["grade"] >= REFERABLE_MIN_GRADE).to_numpy()
        try:
            val, test = part[part["split"] == "val"], part[part["split"] == "test"]
            if len(val) and len(test):
                pack = lambda f: {"referable": (f["grade"] >= REFERABLE_MIN_GRADE).to_numpy(), "scores": f["p_ref"].to_numpy()}
                profiles.append(calibrate_on_split(name, pack(val), pack(test), target_sensitivity, source="built-in"))
            else:
                groups = patient_groups(source, part["image_id"])
                profiles.append(calibrate(name, referable, part["p_ref"].to_numpy(), target_sensitivity, groups=groups, source="built-in"))
        except CalibrationError:
            continue
    return profiles


class SiteRegistry:
    """The site profiles stored next to a model bundle."""

    def __init__(self, path: Path, fingerprint: str, profiles: list[SiteProfile]):
        self.path, self.fingerprint, self.profiles = Path(path), fingerprint, profiles

    @classmethod
    def load(cls, model_dir, config: dict) -> "SiteRegistry":
        path, fingerprint = Path(model_dir) / SITES_FILE, model_fingerprint(config)
        profiles = []
        if path.exists():
            data = json.loads(path.read_text())
            for raw in data.get("sites", []):
                raw = {k: tuple(v) if k.endswith("_range") and v is not None else v for k, v in raw.items() if k in SiteProfile.__dataclass_fields__}
                raw = {k: (float("nan"), float("nan")) if k.endswith("_range") and v is None else v for k, v in raw.items()}
                profile = SiteProfile(**raw)
                profile.stale = data.get("model_fingerprint") != fingerprint or raw.get("stale", False)
                profiles.append(profile)
        return cls(path, fingerprint, profiles)

    def get(self, name: str) -> SiteProfile | None:
        return next((p for p in self.profiles if p.name == name), None)

    def usable(self) -> list[SiteProfile]:
        return [p for p in self.profiles if not p.stale]

    def put(self, profile: SiteProfile) -> None:
        self.profiles = [p for p in self.profiles if p.name != profile.name] + [profile]
        self.save()

    def remove(self, name: str) -> None:
        self.profiles = [p for p in self.profiles if p.name != name]
        self.save()

    def save(self) -> None:
        def clean(value):
            if isinstance(value, float) and not np.isfinite(value):
                return None
            if isinstance(value, tuple):
                return [clean(v) for v in value]
            return value

        sites = [{k: clean(v) for k, v in asdict(p).items() if k != "stale"} for p in self.profiles if not p.stale]
        stale = [{**{k: clean(v) for k, v in asdict(p).items()}, "stale": True} for p in self.profiles if p.stale]
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"model_fingerprint": self.fingerprint, "sites": sites + stale}, indent=2))
        tmp.replace(self.path)
