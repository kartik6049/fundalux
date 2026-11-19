"""Screen one photograph from the command line.

    python -m fundalux.cli photo.jpg [--models models] [--save-dir out]

Prints the result as JSON. With --save-dir it also writes cropped.png,
enhanced.png (the classifier input) and heatmap.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from .pipeline import DISCLAIMER, Fundalux
from .preprocess import read_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen a fundus photograph for referable diabetic retinopathy.")
    parser.add_argument("image")
    parser.add_argument("--models", default="models", help="folder with config.json and the .pt files")
    parser.add_argument("--save-dir", help="write the processed images here")
    args = parser.parse_args()

    result = Fundalux.load(args.models).predict(read_image(args.image))
    print(json.dumps({**result.summary(), "disclaimer": DISCLAIMER}, indent=2))
    if args.save_dir:
        out = Path(args.save_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name in ("cropped", "enhanced", "overlay"):
            img = getattr(result, name)
            if img is not None:
                cv2.imwrite(str(out / f"{'heatmap' if name == 'overlay' else name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
