"""Synthetic fundus-like images and fake dataset folders.

Only for testing the code path end to end without the real data (unit tests and
the notebook's smoke-test mode). Nothing learned from these images is meaningful.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from .degrade import degrade


def synthetic_fundus(rng: np.random.Generator, width: int = 240, height: int = 160, grade: int = 0, ungradable: bool = False):
    """Draw a retina-like disc; higher grades get more dark and bright spots.

    Returns (image RGB uint8, lesions) where lesions maps "HE"/"EX" to boolean masks.
    """
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    cx, cy, r = width / 2, height / 2, 0.46 * width  # wider than tall: the disc is cut off top and bottom
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    inside = dist < r
    shade = 1.0 - 0.35 * (dist / r) ** 2
    base = np.array([190.0, 95.0, 45.0]) * rng.uniform(0.8, 1.1)
    img = np.zeros((height, width, 3), np.float32)
    img[inside] = (shade[..., None] * base)[inside]
    # Optic disc and a few vessels radiating from it.
    side = 1 if rng.random() < 0.5 else -1
    od = (int(cx + side * 0.5 * r), int(cy + rng.uniform(-0.05, 0.05) * r))
    cv2.circle(img, od, max(2, int(0.13 * r)), (245.0, 215.0, 150.0), -1)
    for _ in range(6):
        pts, angle = [od], rng.uniform(0, 2 * np.pi)
        for _ in range(6):
            angle += rng.uniform(-0.5, 0.5)
            pts.append((int(pts[-1][0] + 0.25 * r * np.cos(angle)), int(pts[-1][1] + 0.25 * r * np.sin(angle))))
        cv2.polylines(img, [np.array(pts, np.int32)], False, (120.0, 40.0, 25.0), max(1, int(0.015 * r)))
    lesions = {"HE": np.zeros((height, width), np.uint8), "EX": np.zeros((height, width), np.uint8)}
    for _ in range(3 * grade):
        rho, phi = 0.8 * r * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
        centre = (int(cx + rho * np.cos(phi)), int(cy + rho * np.sin(phi)))
        radius = max(1, int(rng.uniform(0.02, 0.05) * r))
        kind, colour = ("HE", (95.0, 15.0, 15.0)) if rng.random() < 0.6 else ("EX", (250.0, 225.0, 120.0))
        cv2.circle(img, centre, radius, colour, -1)
        cv2.circle(lesions[kind], centre, radius, 1, -1)
    img[~inside] = 0
    img = np.clip(cv2.GaussianBlur(img, (0, 0), 0.6) + rng.normal(0, 2.0, img.shape), 0, 255).astype(np.uint8)
    img[~inside] = 0
    if ungradable:
        img = degrade(img, "blur", 1.0, rng)
    return img, {k: (v > 0) & inside for k, v in lesions.items()}


def _save(path, img_rgb) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))


def _grades(rng, n):
    return rng.choice(5, size=n, p=[0.45, 0.1, 0.25, 0.1, 0.1])


def make_fake_datasets(root, n_eyepacs_patients: int = 40, n_aptos: int = 40, n_idrid: int = 16, n_idrid_seg: int = 6, n_messidor: int = 16, seed: int = 0) -> dict:
    """Write tiny datasets that mirror the folder layout of the real Kaggle downloads.

    Returns {name: folder} plus "eyeq_csv": path of a fake EyeQ label file.
    """
    rng = np.random.default_rng(seed)
    roots = {k: os.path.join(root, k) for k in ("eyepacs", "aptos", "idrid", "messidor")}

    # EyePACS (tanlikesmath/diabetic-retinopathy-resized) + EyeQ quality labels
    rows, eyeq = [], []
    for pid in range(10, 10 + n_eyepacs_patients):
        for eye in ("left", "right"):
            grade = int(_grades(rng, 1)[0])
            bad = rng.random() < 0.2
            img, _ = synthetic_fundus(rng, grade=grade, ungradable=bad)
            name = f"{pid}_{eye}"
            _save(os.path.join(roots["eyepacs"], "resized_train", "resized_train", name + ".jpeg"), img)
            rows.append(f"{name},{grade}")
            if rng.random() < 0.8:
                eyeq.append(f"{len(eyeq)},{name}.jpeg,{2 if bad else int(rng.integers(2))},{grade}")
    _save(os.path.join(roots["eyepacs"], "resized_train_cropped", "resized_train_cropped", "10_left.jpeg"), img)
    with open(os.path.join(roots["eyepacs"], "trainLabels.csv"), "w") as f:
        f.write("image,level\n" + "\n".join(rows) + "\n")
    with open(os.path.join(roots["eyepacs"], "trainLabels_cropped.csv"), "w") as f:
        f.write(",Unnamed: 0,image,level\n0,0,10_left,0\n")
    eyeq_csv = os.path.join(root, "Label_EyeQ_train.csv")
    with open(eyeq_csv, "w") as f:
        f.write(",image,quality,DR_grade\n" + "\n".join(eyeq) + "\n")

    # APTOS 2019 (mariaherrerot/aptos2019): three folders, three CSVs, trailing blank lines
    parts = {"train_1.csv": ("train_images", []), "valid.csv": ("val_images", []), "test.csv": ("test_images", [])}
    for i, grade in enumerate(_grades(rng, n_aptos)):
        csv = "train_1.csv" if i % 5 < 3 else "valid.csv" if i % 5 == 3 else "test.csv"
        folder, lines = parts[csv]
        name = f"{int(rng.integers(16**12)):012x}"
        img, _ = synthetic_fundus(rng, width=200, height=200, grade=int(grade))
        _save(os.path.join(roots["aptos"], folder, folder, name + ".png"), img)
        lines.append(f"{name},{grade}")
    for csv, (_, lines) in parts.items():
        with open(os.path.join(roots["aptos"], csv), "w") as f:
            f.write("id_code,diagnosis\r\n" + "\r\n".join(lines) + "\r\n\r\n\r\n")

    # IDRiD (aaryapatel98/indian-diabetic-retinopathy-image-dataset): grading + segmentation
    grading = os.path.join(roots["idrid"], "B.%20Disease%20Grading", "B. Disease Grading")
    for folder, csv, n in (("a. Training Set", "a. IDRiD_Disease Grading_Training Labels.csv", n_idrid - n_idrid // 4), ("b. Testing Set", "b. IDRiD_Disease Grading_Testing Labels.csv", n_idrid // 4)):
        lines = []
        for i, grade in enumerate(_grades(rng, n), start=1):
            img, _ = synthetic_fundus(rng, grade=int(grade))
            _save(os.path.join(grading, "1. Original Images", folder, f"IDRiD_{i:03d}.jpg"), img)
            lines.append(f"IDRiD_{i:03d},{grade},0,,,")
        os.makedirs(os.path.join(grading, "2. Groundtruths"), exist_ok=True)
        with open(os.path.join(grading, "2. Groundtruths", csv), "w") as f:
            f.write("Image name,Retinopathy grade,Risk of macular edema ,,,\r\n" + "\r\n".join(lines) + "\r\n")
    seg = os.path.join(roots["idrid"], "A.%20Segmentation", "A. Segmentation")
    mask_dirs = {"HE": "2. Haemorrhages", "EX": "3. Hard Exudates"}
    for i in range(1, n_idrid_seg + 1):
        folder = "a. Training Set" if i <= n_idrid_seg - 2 else "b. Testing Set"
        img, lesions = synthetic_fundus(rng, grade=int(rng.integers(2, 5)))
        _save(os.path.join(seg, "1. Original Images", folder, f"IDRiD_{i:02d}.jpg"), img)
        for kind, mask in lesions.items():
            if mask.any():
                path = os.path.join(seg, "2. All Segmentation Groundtruths", folder, mask_dirs[kind], f"IDRiD_{i:02d}_{kind}.tif")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                cv2.imwrite(path, mask.astype(np.uint8) * 76)

    # Messidor-2 (mariaherrerot/messidor2preprocess): a few adjudicated-ungradable rows have no grade
    lines = []
    for i, grade in enumerate(_grades(rng, n_messidor)):
        gradable = i % 8 != 7
        img, _ = synthetic_fundus(rng, width=200, height=200, grade=int(grade), ungradable=not gradable)
        name = f"2005{i:04d}_{int(rng.integers(10**5)):05d}_0100_PP.png"
        _save(os.path.join(roots["messidor"], "messidor-2", "messidor-2", "preprocess", name), img)
        lines.append(f"{name},{grade},0,1" if gradable else f"{name},,,0")
    with open(os.path.join(roots["messidor"], "messidor_data.csv"), "w") as f:
        f.write("id_code,diagnosis,adjudicated_dme,adjudicated_gradable\n" + "\n".join(lines) + "\n")

    return {**roots, "eyeq_csv": eyeq_csv}
