"""Clinician review: blinded grading of a fixed set of photographs, and the analysis.

A review set is a folder with set.json and the photographs under opaque ids, so
a file name can never hint at the answer. The model's output for every photograph
is computed once, when the set is built, and stored with the set; reviewers never
see it unless they choose the teaching mode, which is recorded with each answer.

Answers live in a SQLite file next to the set (one row per reviewer and photograph).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import GRADE_NAMES, REFERABLE_MIN_GRADE
from . import metrics as M

MANIFEST = "set.json"
DATABASE = "responses.sqlite3"
MAX_SIDE = 1600  # photographs are stored at most this wide: enough to grade, small enough to serve


# --------------------------------------------------------------------------- building a set


def _save_jpeg(path: Path, rgb: np.ndarray, max_side: int = MAX_SIDE, quality: int = 90) -> None:
    scale = min(1.0, max_side / max(rgb.shape[:2]))
    if scale < 1:
        rgb = cv2.resize(rgb, (round(rgb.shape[1] * scale), round(rgb.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])


def stratified_sample(items: list[dict], n: int, seed: int = 0) -> list[dict]:
    """Up to n items, spread as evenly as possible over reference grades (unlabelled items count as one group)."""
    rng = np.random.default_rng(seed)
    groups: dict = {}
    for it in items:
        groups.setdefault(it.get("reference_grade"), []).append(it)
    for g in groups.values():
        rng.shuffle(g)
    chosen, keys = [], sorted(groups, key=lambda k: (k is None, k))
    while len(chosen) < n and any(groups[k] for k in keys):
        for k in keys:
            if groups[k] and len(chosen) < n:
                chosen.append(groups[k].pop())
    rng.shuffle(chosen)
    return chosen


def build_review_set(bundle, photos: list[tuple[str, bytes]], out_dir, name: str, references: dict | None = None,
                     site: str | None = None, progress=None) -> dict:
    """Score every photograph once and write the set (photographs, heatmaps, set.json) to out_dir.

    `photos` is [(original name, encoded bytes)]; `references` maps an original
    file stem (lower case) to a reference ICDR grade. Original names are kept in
    set.json for the organiser's export, but never sent to reviewers.
    """
    from .preprocess import decode_image
    from .sites import model_fingerprint

    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "heat").mkdir(parents=True, exist_ok=True)
    references = references or {}
    items = []
    for i, (original, data) in enumerate(photos):
        item_id = f"r{i + 1:03d}"
        image = decode_image(data)
        result = bundle.predict(image)
        _save_jpeg(out / "images" / f"{item_id}.jpg", image)
        heat = None
        if result.overlay is not None:
            heat = f"heat/{item_id}.jpg"
            _save_jpeg(out / heat, result.overlay, 768, 86)
        ref = references.get(Path(original).stem.lower())
        items.append({
            "id": item_id, "file": f"images/{item_id}.jpg", "heatmap": heat, "original": original,
            "reference_grade": None if ref is None else int(ref),
            "model": {"gradable": result.gradable, "ungradable_score": result.ungradable_score, "referable_score": result.referable_score,
                      "grade": result.grade, "grade_probs": result.grade_probs, "reasons": result.reasons},
        })
        if progress:
            progress(i + 1, len(photos))
    manifest = {
        "name": name, "created": time.strftime("%Y-%m-%d"), "site": site,
        "model_run": bundle.config.get("trained_with", {}).get("run"), "model_fingerprint": model_fingerprint(bundle.config),
        "items": items,
    }
    (out / MANIFEST).write_text(json.dumps(manifest, indent=1))
    return manifest


# --------------------------------------------------------------------------- the set and the answers


@dataclass
class ReviewSet:
    folder: Path
    manifest: dict

    @classmethod
    def load(cls, folder) -> "ReviewSet | None":
        path = Path(folder) / MANIFEST
        return cls(Path(folder), json.loads(path.read_text())) if path.exists() else None

    @property
    def items(self) -> list[dict]:
        return self.manifest["items"]

    def item(self, item_id: str) -> dict | None:
        return next((it for it in self.items if it["id"] == item_id), None)

    def order_for(self, reviewer: str) -> list[str]:
        """Each reviewer sees the photographs in their own fixed random order (order effects average out)."""
        ids = [it["id"] for it in self.items]
        seed = int.from_bytes(hashlib.sha256(reviewer.strip().lower().encode()).digest()[:8], "big")
        return [ids[i] for i in np.random.default_rng(seed).permutation(len(ids))]


class ResponseStore:
    COLUMNS = ["reviewer", "item_id", "gradable", "grade", "other_referral", "confidence", "note", "seconds", "mode", "created"]

    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.Lock()
        with self._connect() as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS responses (
                    reviewer TEXT NOT NULL, item_id TEXT NOT NULL, gradable INTEGER NOT NULL, grade INTEGER,
                    other_referral INTEGER NOT NULL DEFAULT 0, confidence TEXT, note TEXT, seconds REAL,
                    mode TEXT NOT NULL, created REAL NOT NULL, PRIMARY KEY (reviewer, item_id))"""
            )

    def _connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def save(self, reviewer: str, item_id: str, gradable: bool, grade: int | None, other_referral: bool, confidence: str | None,
             note: str, seconds: float | None, mode: str) -> None:
        if gradable and grade not in range(5):
            raise ValueError("A gradable photograph needs a grade from 0 to 4.")
        with self.lock, self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (reviewer.strip(), item_id, int(gradable), grade if gradable else None, int(other_referral), confidence, (note or "")[:2000], seconds, mode, time.time()),
            )

    def answered(self, reviewer: str) -> set[str]:
        with self._connect() as db:
            return {r[0] for r in db.execute("SELECT item_id FROM responses WHERE reviewer = ?", (reviewer.strip(),))}

    def frame(self) -> pd.DataFrame:
        with self._connect() as db:
            return pd.read_sql_query("SELECT * FROM responses ORDER BY reviewer, item_id", db)


# --------------------------------------------------------------------------- analysis


def _kappa_unweighted(a, b) -> float:
    a, b = np.asarray(a).astype(int), np.asarray(b).astype(int)
    if a.size == 0:
        return float("nan")
    labels = np.union1d(a, b)
    po = float((a == b).mean())
    pe = float(sum((a == k).mean() * (b == k).mean() for k in labels))
    return float("nan") if pe >= 1 else (po - pe) / (1 - pe)


def _pair(name_a, a, name_b, b) -> dict:
    """Agreement on referable / not referable between two raters on the same photographs."""
    a, b = np.asarray(a).astype(int), np.asarray(b).astype(int)
    return {
        "a": name_a, "b": name_b, "n": int(a.size),
        "agreement": float((a == b).mean()) if a.size else None, "kappa": _kappa_unweighted(a, b),
        "table": [[int(((a == i) & (b == j)).sum()) for j in (0, 1)] for i in (0, 1)],  # rows: a no/yes; columns: b no/yes
    }


def _accuracy(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true).astype(bool), np.asarray(y_pred).astype(bool)
    pos, neg = y_true.sum(), (~y_true).sum()
    return {"n": int(y_true.size), "referable": int(pos),
            "sensitivity": float((y_pred & y_true).sum() / pos) if pos else None,
            "specificity": float((~y_pred & ~y_true).sum() / neg) if neg else None}


def analyse(review_set: ReviewSet, responses: pd.DataFrame, threshold: float, threshold_label: str) -> dict:
    """Everything the results page shows. The model decides at `threshold` (global or a site's)."""
    items = {it["id"]: it for it in review_set.items}
    model = pd.DataFrame([{
        "item_id": i, "model_gradable": it["model"]["gradable"], "model_score": it["model"]["referable_score"], "model_grade": it["model"]["grade"],
        "reference_grade": it["reference_grade"],
    } for i, it in items.items()])
    model["model_referable"] = model["model_gradable"] & (model["model_score"].fillna(-1) >= threshold)
    out = {"threshold": threshold, "threshold_label": threshold_label, "items": len(items), "has_reference": bool(model["reference_grade"].notna().any()),
           "reviewers": [], "pairs": [], "versus_reference": [], "grade_kappa": [], "disagreements": [], "notes": [], "gradability": None}
    if responses.empty:
        return out
    df = responses.merge(model, on="item_id", how="inner")
    df["doctor_referable"] = df["gradable"].astype(bool) & (df["grade"].fillna(-1) >= REFERABLE_MIN_GRADE)

    for name, r in df.groupby("reviewer"):
        out["reviewers"].append({
            "name": name, "answered": int(len(r)), "of": len(items), "ungradable": int((~r.gradable.astype(bool)).sum()),
            "other_referral": int(r.other_referral.sum()), "median_seconds": float(r.seconds.median()) if r.seconds.notna().any() else None,
            "teaching_mode": int((r["mode"] == "teaching").sum()),
        })
        both = r[r.gradable.astype(bool) & r.model_gradable.astype(bool)]
        out["pairs"].append(_pair(name, both.doctor_referable, "Fundalux", both.model_referable))
        g = both[both.grade.notna() & both.model_grade.notna()]
        if len(g):
            out["grade_kappa"].append({"a": name, "b": "Fundalux", "n": int(len(g)), "kappa": M.quadratic_weighted_kappa(g.grade.astype(int), g.model_grade.astype(int), 5)})

    # Doctors with each other, on photographs both called gradable.
    names = sorted(df.reviewer.unique())
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            m = df[df.reviewer == a].merge(df[df.reviewer == b], on="item_id", suffixes=("_a", "_b"))
            m = m[m.gradable_a.astype(bool) & m.gradable_b.astype(bool)]
            if len(m):
                out["pairs"].append(_pair(a, m.doctor_referable_a, b, m.doctor_referable_b))
                out["grade_kappa"].append({"a": a, "b": b, "n": int(len(m)), "kappa": M.quadratic_weighted_kappa(m.grade_a.astype(int), m.grade_b.astype(int), 5)})

    # Against the reference grade, doctor and model on exactly the same photographs.
    if out["has_reference"]:
        for name, r in df.groupby("reviewer"):
            s = r[r.reference_grade.notna() & r.gradable.astype(bool) & r.model_gradable.astype(bool)]
            if len(s):
                truth = s.reference_grade >= REFERABLE_MIN_GRADE
                out["versus_reference"].append({"reviewer": name, "doctor": _accuracy(truth, s.doctor_referable), "model": _accuracy(truth, s.model_referable),
                                                "doctor_grade_kappa": M.quadratic_weighted_kappa(s.reference_grade.astype(int), s.grade.astype(int), 5),
                                                "model_grade_kappa": M.quadratic_weighted_kappa(s.reference_grade.astype(int), s.model_grade.astype(int), 5)})

    # Gradability: does the quality gate agree with the doctors about what can be graded?
    out["gradability"] = {
        "both_gradable": int((df.gradable.astype(bool) & df.model_gradable.astype(bool)).sum()),
        "doctor_only_ungradable": int((~df.gradable.astype(bool) & df.model_gradable.astype(bool)).sum()),
        "gate_only_refused": int((df.gradable.astype(bool) & ~df.model_gradable.astype(bool)).sum()),
        "both_ungradable": int((~df.gradable.astype(bool) & ~df.model_gradable.astype(bool)).sum()),
    }

    # Disagreements on referral, most informative first: the model said no where a doctor said yes.
    dis = df[df.gradable.astype(bool) & df.model_gradable.astype(bool) & (df.doctor_referable != df.model_referable)].copy()
    dis["kind"] = np.where(dis.doctor_referable, "Doctor refers, Fundalux does not", "Fundalux refers, doctor does not")
    dis = dis.sort_values(["doctor_referable", "item_id"], ascending=[False, True])
    for _, d in dis.iterrows():
        out["disagreements"].append({
            "item_id": d.item_id, "reviewer": d.reviewer, "kind": d.kind, "doctor_grade": GRADE_NAMES[int(d.grade)],
            "model_score": d.model_score, "model_grade": GRADE_NAMES[int(d.model_grade)] if pd.notna(d.model_grade) else None,
            "reference_grade": GRADE_NAMES[int(d.reference_grade)] if pd.notna(d.reference_grade) else None, "note": d.note or "",
        })
    out["notes"] = [{"item_id": n.item_id, "reviewer": n.reviewer, "note": n.note} for n in df[df.note.fillna("").str.strip() != ""].itertuples()]
    return out


def export_frame(review_set: ReviewSet, responses: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """One row per answer, with the model's output and the reference, for the organiser's own analysis."""
    rows = []
    for it in review_set.items:
        m = it["model"]
        rows.append({"item_id": it["id"], "original_file": it.get("original"), "reference_grade": it["reference_grade"],
                     "model_gradable": m["gradable"], "model_score": m["referable_score"], "model_grade": m["grade"],
                     "model_referable": bool(m["gradable"] and m["referable_score"] is not None and m["referable_score"] >= threshold)})
    return responses.merge(pd.DataFrame(rows), on="item_id", how="left")
