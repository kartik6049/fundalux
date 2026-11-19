"""Render the pipeline illustrations used by the web pages from a real sample.

    python scripts/build_web_assets.py

Writes server/static/img/stage-*.jpg (photograph, crop, enhancement, heatmap)
and degradation examples. Re-run after retraining so the heatmap matches the models.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fundalux.degrade import degrade  # noqa: E402
from fundalux.pipeline import Fundalux  # noqa: E402
from fundalux.preprocess import correct_illumination, read_image  # noqa: E402

OUT = ROOT / "server" / "static" / "img"
SAMPLE = ROOT / "samples" / "idrid_test_IDRiD_008_grade2.jpg"


def save(name: str, rgb: np.ndarray, width: int = 640) -> None:
    scale = width / rgb.shape[1]
    small = cv2.resize(rgb, (width, round(rgb.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(OUT / name), cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 86, cv2.IMWRITE_JPEG_PROGRESSIVE, 1])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = Fundalux.load(ROOT / "models")
    photo = read_image(SAMPLE)
    result = bundle.predict(photo)
    save("stage-1-photo.jpg", photo, 760)
    save("stage-2-crop.jpg", result.cropped)
    save("stage-3-illumination.jpg", correct_illumination(result.cropped))
    save("stage-4-enhanced.jpg", result.enhanced)
    save("stage-5-heatmap.jpg", result.overlay)
    rng = np.random.default_rng(0)
    for kind, severity in (("blur", 0.8), ("dark", 0.8), ("glare", 0.7)):
        save(f"degraded-{kind}.jpg", degrade(result.cropped, kind, severity, rng), 320)
    (OUT / "hero.json").write_text(json.dumps({"sample": SAMPLE.name, "label": "IDRiD 008, graded moderate NPDR", "score": round(result.referable_score, 3), "grade": result.grade_name}))
    print(f"Wrote {len(list(OUT.glob('*.jpg')))} images to {OUT} (sample score {result.referable_score:.3f})")


if __name__ == "__main__":
    main()
