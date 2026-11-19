"""Write models/sites.json with built-in profiles for the evaluation datasets.

    python scripts/build_site_profiles.py [model_dir]

Uses the bundle's reports/predictions.csv. New training runs write this file
themselves; this script is for bundles trained before they did. Profiles you
calibrated yourself are kept.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fundalux.sites import SiteRegistry, builtin_profiles  # noqa: E402


def main() -> None:
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "models"
    config = json.loads((model_dir / "config.json").read_text())
    metrics = json.loads((model_dir / "metrics.json").read_text())
    target = metrics["dr"]["validation"]["target_sensitivity"]
    predictions = pd.read_csv(model_dir / "reports" / "predictions.csv", dtype={"image_id": str})
    registry = SiteRegistry.load(model_dir, config)
    for profile in builtin_profiles(predictions, target):
        registry.put(profile)
        print(f"{profile.name:30s} threshold {profile.threshold:.3f}   expected sensitivity {profile.expected_sensitivity:.1%}, specificity {profile.expected_specificity:.1%}")
    print("Wrote", registry.path)


if __name__ == "__main__":
    main()
