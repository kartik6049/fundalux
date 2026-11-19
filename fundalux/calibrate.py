"""Calibrate a decision threshold for a new site from its graded photographs.

    python -m fundalux.calibrate --site "PHC Nanded" --labels labels.csv --images photos/ [--models models]

labels.csv needs an image column (file name, with or without extension) and
either a grade column (0-4) or a referable column (0/1). Images that fail the
quality gate are left out, as in use. Scoring runs on the CPU without Grad-CAM;
expect several seconds per image.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import REFERABLE_MIN_GRADE
from .datasets import IMAGE_EXT
from .pipeline import Fundalux
from .preprocess import decode_image
from .sites import CalibrationError, SiteRegistry, calibrate


def read_labels(table: pd.DataFrame) -> pd.DataFrame:
    """Normalise a labels table to columns key (file stem, lower case) and referable (0/1)."""
    columns = {c.lower().strip(): c for c in table.columns}
    image = next((columns[c] for c in ("image", "image_id", "id_code", "file", "filename", "image name") if c in columns), None)
    if image is None:
        raise CalibrationError("The labels file needs an 'image' column with the photo file names.")
    if "referable" in columns:
        referable = table[columns["referable"]].astype(int)
    elif "grade" in columns or "diagnosis" in columns or "level" in columns:
        grade_col = columns.get("grade") or columns.get("diagnosis") or columns.get("level")
        referable = (table[grade_col].astype(int) >= REFERABLE_MIN_GRADE).astype(int)
    else:
        raise CalibrationError("The labels file needs a 'grade' (0-4) or 'referable' (0/1) column.")
    keys = table[image].astype(str).str.strip().map(lambda s: Path(s).stem.lower())
    out = pd.DataFrame({"key": keys, "referable": referable}).dropna()
    if not out["referable"].isin([0, 1]).all():
        raise CalibrationError("'referable' must be 0 or 1.")
    return out.drop_duplicates("key")


def score_images(bundle: Fundalux, named_images, progress=None) -> pd.DataFrame:
    """Score (name, encoded bytes) pairs. Returns key, gradable, score."""
    rows = []
    items = list(named_images)
    for i, (name, data) in enumerate(items):
        try:
            gradable, score = bundle.referable_score(decode_image(data))
        except Exception:  # unreadable file: treat like an ungradable photo
            gradable, score = False, None
        rows.append({"key": Path(name).stem.lower(), "gradable": gradable, "score": score})
        if progress:
            progress(i + 1, len(items))
    return pd.DataFrame(rows, columns=["key", "gradable", "score"])


def calibrate_site(name: str, labels: pd.DataFrame, scored: pd.DataFrame, target: float, source: str):
    """Join labels with scores and fit the profile. Returns (profile, summary of what was used)."""
    joined = labels.merge(scored, on="key", how="inner")
    summary = {"labelled": len(labels), "matched": len(joined), "unmatched_labels": len(labels) - len(joined), "rejected_by_gate": int((~joined["gradable"]).sum())}
    usable = joined[joined["gradable"]]
    profile = calibrate(name, usable["referable"].to_numpy(), usable["score"].to_numpy(dtype=np.float64), target, source=source, rejected_by_gate=summary["rejected_by_gate"])
    return profile, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--site", required=True, help="name of the site / camera")
    parser.add_argument("--labels", required=True, help="CSV with image and grade (or referable) columns")
    parser.add_argument("--images", required=True, help="folder with the photographs")
    parser.add_argument("--models", default="models")
    parser.add_argument("--target", type=float, help="target sensitivity (default: the one the bundle was tuned for)")
    args = parser.parse_args()

    bundle = Fundalux.load(args.models)
    metrics_path = Path(args.models) / "metrics.json"
    default_target = json.loads(metrics_path.read_text())["dr"]["validation"]["target_sensitivity"] if metrics_path.exists() else 0.93
    target = args.target or default_target
    labels = read_labels(pd.read_csv(args.labels))
    wanted = set(labels["key"])
    files = [p for p in sorted(Path(args.images).rglob("*")) if p.suffix.lower() in IMAGE_EXT and p.stem.lower() in wanted]
    print(f"{len(labels)} labelled images, {len(files)} found in {args.images}. Scoring ...")

    def report(done, total):
        if done % 10 == 0 or done == total:
            print(f"  {done}/{total}", flush=True)

    scored = score_images(bundle, ((p.name, p.read_bytes()) for p in files), progress=report)
    try:
        profile, summary = calibrate_site(args.site, labels, scored, target, source="calibrated with the command line")
    except CalibrationError as exc:
        sys.exit(f"Not calibrated: {exc}")
    registry = SiteRegistry.load(args.models, bundle.config)
    registry.put(profile)
    print(json.dumps(summary))
    print(f"{profile.name}: threshold {profile.threshold:.3f}; expected sensitivity {profile.expected_sensitivity:.1%}, specificity {profile.expected_specificity:.1%} ({profile.estimate})")
    for warning in profile.warnings:
        print("Warning:", warning)
    print("Saved to", registry.path)


if __name__ == "__main__":
    main()
