"""Find images and labels inside the downloaded datasets and build one index.

The loaders search by file name instead of assuming fixed paths, because the
folder nesting differs between a Kaggle notebook Input, a download and a manual unzip.
Every loader returns a DataFrame with the columns in `COLUMNS`.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import REFERABLE_MIN_GRADE
from .preprocess import _disc_u8, apply_crop, crop_fundus, find_fundus, keep_where, read_image
from .quality import quality_stats

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
COLUMNS = ["source", "image_id", "path", "grade", "patient"]
TRAIN_SOURCES = ("eyepacs", "aptos")  # everything else is held out as external test data

# Bit flags inside the cached IDRiD lesion-mask PNGs.
LESION_BITS = {"MA": 1, "HE": 2, "EX": 4, "SE": 8, "OD": 16}
LESION_ANY = 1 | 2 | 4 | 8  # the optic disc is anatomy, not a lesion


def find_kaggle_input(slug: str, base="/kaggle/input") -> str | None:
    """Folder of a dataset attached to a Kaggle notebook ("Add Input"), or None.

    Kaggle has mounted datasets at /kaggle/input/<name> and, more recently, in
    deeper owner/name layouts, so the dataset name is searched a few levels down.
    """
    name, base = slug.split("/")[-1].lower(), Path(base)
    if not base.is_dir():
        return None
    for pattern in ("*", "*/*", "*/*/*"):
        for path in sorted(base.glob(pattern)):
            if path.is_dir() and path.name.lower() == name:
                return str(path)
    return None


def _files(root, suffixes) -> list[Path]:
    out = []
    for folder, _, names in os.walk(root):
        out.extend(Path(folder) / n for n in names if n.lower().endswith(suffixes))
    return sorted(out)


def _by_stem(paths) -> dict[str, Path]:
    return {p.stem.lower(): p for p in paths}


def _frame(source, ids, paths, grades, patients) -> pd.DataFrame:
    df = pd.DataFrame({"source": source, "image_id": list(ids), "path": [str(p) for p in paths], "grade": list(grades), "patient": list(patients)})
    return df.astype({"grade": int})[COLUMNS]


def index_eyepacs(root) -> pd.DataFrame:
    """EyePACS (Kaggle DR 2015 train set): trainLabels.csv with columns image, level."""
    csvs = [p for p in _files(root, (".csv",)) if p.name.lower() == "trainlabels.csv"]
    if not csvs:
        raise FileNotFoundError(f"trainLabels.csv not found under {root}")
    labels = pd.read_csv(csvs[0]).dropna(subset=["image", "level"])
    # The resized mirror ships a second, pre-cropped copy; we do our own cropping.
    images = _by_stem(p for p in _files(root, IMAGE_EXT) if "cropped" not in str(p.parent).lower())
    labels["path"] = labels["image"].astype(str).str.lower().map(images)
    labels = labels.dropna(subset=["path"])
    ids = labels["image"].astype(str)
    return _frame("eyepacs", ids, labels["path"], labels["level"], "eyepacs_" + ids.str.split("_").str[0])


def index_aptos(root) -> pd.DataFrame:
    """APTOS 2019: one or more CSVs with columns id_code, diagnosis."""
    frames = []
    for p in _files(root, (".csv",)):
        df = pd.read_csv(p)
        if {"id_code", "diagnosis"} <= set(df.columns):
            frames.append(df[["id_code", "diagnosis"]].dropna())
    if not frames:
        raise FileNotFoundError(f"No CSV with id_code/diagnosis columns under {root}")
    labels = pd.concat(frames).drop_duplicates("id_code")
    images = _by_stem(_files(root, IMAGE_EXT))
    labels["path"] = labels["id_code"].astype(str).str.lower().map(images)
    labels = labels.dropna(subset=["path"])
    ids = labels["id_code"].astype(str)
    return _frame("aptos", ids, labels["path"], labels["diagnosis"], "aptos_" + ids)


def index_idrid(root) -> pd.DataFrame:
    """IDRiD 'B. Disease Grading': training and testing sets reuse file names, so ids carry the set."""
    frames = []
    for csv in _files(root, (".csv",)):
        if "disease grading" not in csv.name.lower():
            continue
        subset = "train" if "training" in csv.name.lower() else "test"
        folder = "training set" if subset == "train" else "testing set"
        df = pd.read_csv(csv)
        df.columns = [str(c).strip().lower() for c in df.columns]
        df = df[["image name", "retinopathy grade"]].dropna()
        images = _by_stem(
            p for p in _files(root, IMAGE_EXT) if "disease grading" in str(p.parent).lower() and "original images" in str(p.parent).lower() and p.parent.name.lower().endswith(folder)
        )
        df["path"] = df["image name"].astype(str).str.strip().str.lower().map(images)
        df = df.dropna(subset=["path"])
        ids = subset + "_" + df["image name"].astype(str).str.strip()
        frames.append(_frame("idrid", ids, df["path"], df["retinopathy grade"], "idrid_" + ids))
    if not frames:
        raise FileNotFoundError(f"IDRiD disease-grading label CSVs not found under {root}")
    return pd.concat(frames, ignore_index=True)


def index_idrid_segmentation(root) -> list[dict]:
    """IDRiD 'A. Segmentation': [{image_id, path, masks: {MA|HE|EX|SE|OD: path}}]."""
    items = {}
    for p in _files(root, IMAGE_EXT):
        parent = str(p.parent).lower()
        if "segmentation" not in parent:
            continue
        if "original images" in parent:
            items.setdefault(p.stem, {"image_id": p.stem, "masks": {}})["path"] = str(p)
        elif "groundtruths" in parent:
            stem, _, kind = p.stem.rpartition("_")
            if kind in LESION_BITS:
                items.setdefault(stem, {"image_id": stem, "masks": {}})["masks"][kind] = str(p)
    return [items[k] for k in sorted(items) if "path" in items[k]]


def index_messidor(root) -> pd.DataFrame:
    """Messidor-2 with the adjudicated grades CSV. Images the adjudicators found
    ungradable have no grade; they get grade -1 and are kept for the quality gate."""
    csvs = [p for p in _files(root, (".csv",)) if "messidor" in p.name.lower()]
    if not csvs:
        raise FileNotFoundError(f"messidor_data.csv not found under {root}")
    df = pd.read_csv(csvs[0]).rename(columns={"image_id": "id_code", "adjudicated_dr_grade": "diagnosis"})
    images = _by_stem(_files(root, IMAGE_EXT))
    df["path"] = df["id_code"].astype(str).map(lambda s: images.get(Path(s).stem.lower()))
    df = df.dropna(subset=["path"])
    ids = df["id_code"].astype(str).map(lambda s: Path(s).stem)
    return _frame("messidor", ids, df["path"], df["diagnosis"].fillna(-1), "messidor_" + ids)


INDEXERS = {"eyepacs": index_eyepacs, "aptos": index_aptos, "idrid": index_idrid, "messidor": index_messidor}


def add_eyeq_labels(df: pd.DataFrame, eyeq_csvs) -> pd.DataFrame:
    """Attach EyeQ quality labels (0 good, 1 usable, 2 reject) to EyePACS rows; NaN = unlabelled."""
    df = df.copy()
    df["eyeq"] = np.nan
    frames = [pd.read_csv(p)[["image", "quality"]] for p in eyeq_csvs if p and os.path.exists(p)]
    if frames:
        eyeq = pd.concat(frames).drop_duplicates("image")
        lookup = dict(zip(eyeq["image"].astype(str).map(lambda s: Path(s).stem), eyeq["quality"]))
        is_eyepacs = df["source"] == "eyepacs"
        df.loc[is_eyepacs, "eyeq"] = df.loc[is_eyepacs, "image_id"].map(lookup)
    return df


def assign_splits(df: pd.DataFrame, seed: int = 42, val_frac: float = 0.1, test_frac: float = 0.1) -> pd.DataFrame:
    """Add `split` (train/val/test/external) and `referable` columns.

    Both eyes of a patient always land in the same split, and each split keeps
    the grade mix of every source (patients are stratified by their worst eye).
    """
    df = df.copy()
    df["referable"] = (df["grade"] >= REFERABLE_MIN_GRADE).astype(int)
    df["split"] = "external"
    pool = df[df["source"].isin(TRAIN_SOURCES)]
    patients = pool.groupby("patient").agg(source=("source", "first"), worst=("grade", "max")).sort_index()
    rng = np.random.default_rng(seed)
    assigned = {}
    for _, group in patients.groupby(["source", "worst"], sort=True):
        names = group.index.to_numpy().copy()
        rng.shuffle(names)
        n_test, n_val = int(round(len(names) * test_frac)), int(round(len(names) * val_frac))
        for i, name in enumerate(names):
            assigned[name] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    df.loc[pool.index, "split"] = pool["patient"].map(assigned)
    return df


def _cache_one(task):
    src, dst, size = task
    try:
        cropped, geom = crop_fundus(read_image(src), size)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not cv2.imwrite(dst, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"could not write {dst}")
        return {"ok": True, "fundus_found": geom.found, **quality_stats(cropped)}
    except Exception as exc:  # a corrupt file must not kill a 40k-image job
        return {"ok": False, "fundus_found": False, "error": f"{type(exc).__name__}: {exc}"}


def build_cache(df: pd.DataFrame, cache_dir, size: int, workers: int | None = None, progress=None) -> pd.DataFrame:
    """Crop every image to the fundus at `size` and store it as JPEG under cache_dir/images.

    Returns the index with `cache_path` (relative to cache_dir), `fundus_found`
    and the quality statistics added; unreadable images are dropped.
    """
    df = df.reset_index(drop=True).copy()
    df["cache_path"] = [f"images/{s}/{i}.jpg" for s, i in zip(df["source"], df["image_id"])]
    tasks = [(src, os.path.join(cache_dir, rel), size) for src, rel in zip(df["path"], df["cache_path"])]
    workers = workers or os.cpu_count() or 1
    if workers > 1 and len(tasks) > 1:
        # "spawn" avoids the OpenCV-after-fork deadlock; _cache_one is importable, so it works in notebooks.
        with mp.get_context("spawn").Pool(workers, initializer=cv2.setNumThreads, initargs=(1,)) as pool:
            it = pool.imap(_cache_one, tasks, chunksize=16)
            results = list(progress(it, total=len(tasks)) if progress else it)
    else:
        it = map(_cache_one, tasks)
        results = list(progress(it, total=len(tasks)) if progress else it)
    df = pd.concat([df, pd.DataFrame(results)], axis=1)
    failed = df[~df["ok"]]
    if len(failed):
        print(f"Skipped {len(failed)} unreadable images, e.g. {failed['path'].iloc[0]}: {failed['error'].iloc[0]}")
    return df[df["ok"]].drop(columns=["ok", "error"], errors="ignore").reset_index(drop=True)


def _read_mask(path) -> np.ndarray:
    """Read an IDRiD ground-truth mask as a boolean array (any non-zero pixel is lesion)."""
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        from PIL import Image

        arr = np.asarray(Image.open(path).convert("L"))
    if arr.ndim == 3:
        arr = arr[..., :3].max(axis=2)
    return arr > 0


def cache_idrid_masks(items: list[dict], cache_dir, size: int, progress=None) -> pd.DataFrame:
    """Cache IDRiD segmentation images and their lesion masks with identical geometry.

    Each mask PNG holds one bit per lesion type (see LESION_BITS). A cached pixel
    is marked when lesions cover at least 5% of it, so microaneurysms survive the
    downscale.
    """
    rows = []
    for item in progress(items) if progress else items:
        img = read_image(item["path"])
        geom = find_fundus(img)
        cropped = keep_where(apply_crop(img, geom, size), _disc_u8(size))
        packed = np.zeros((size, size), np.uint8)
        for kind, path in item["masks"].items():
            mask = _read_mask(path)
            if mask.shape != img.shape[:2]:
                mask = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            coverage = apply_crop(mask.astype(np.float32), geom, size, cv2.INTER_AREA if geom.side > size else cv2.INTER_LINEAR)
            packed[coverage >= 0.05] |= LESION_BITS[kind]
        rel_img, rel_mask = f"images/idrid_seg/{item['image_id']}.jpg", f"masks/idrid_seg/{item['image_id']}.png"
        for rel in (rel_img, rel_mask):
            os.makedirs(os.path.dirname(os.path.join(cache_dir, rel)), exist_ok=True)
        cv2.imwrite(os.path.join(cache_dir, rel_img), cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(os.path.join(cache_dir, rel_mask), packed)
        rows.append({"image_id": item["image_id"], "cache_path": rel_img, "mask_path": rel_mask, "lesion_types": "+".join(sorted(item["masks"]))})
    return pd.DataFrame(rows, columns=["image_id", "cache_path", "mask_path", "lesion_types"])
