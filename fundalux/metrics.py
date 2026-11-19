"""Screening metrics in plain numpy (no scikit-learn needed at inference time)."""

from __future__ import annotations

import numpy as np


def roc_curve(y_true, y_score) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (fpr, tpr, thresholds); a sample is positive when score >= threshold."""
    y_true = np.asarray(y_true).astype(bool)
    y_score = np.asarray(y_score, dtype=np.float64)
    order = np.argsort(-y_score, kind="mergesort")
    y_true, y_score = y_true[order], y_score[order]
    last = np.r_[np.where(np.diff(y_score))[0], y_true.size - 1]  # last index of each distinct score
    tps = np.cumsum(y_true)[last]
    fps = (last + 1) - tps
    n_pos, n_neg = int(y_true.sum()), int((~y_true).sum())
    tpr = np.r_[0.0, tps / n_pos] if n_pos else np.full(last.size + 1, np.nan)
    fpr = np.r_[0.0, fps / n_neg] if n_neg else np.full(last.size + 1, np.nan)
    return fpr, tpr, np.r_[np.inf, y_score[last]]


def roc_auc(y_true, y_score) -> float:
    """Area under the ROC curve; NaN when only one class is present."""
    y_true = np.asarray(y_true).astype(bool)
    if y_true.all() or not y_true.any():
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))


def sens_spec(y_true, y_score, threshold: float) -> tuple[float, float]:
    """Sensitivity and specificity when predicting positive for score >= threshold."""
    y_true = np.asarray(y_true).astype(bool)
    pred = np.asarray(y_score) >= threshold
    n_pos, n_neg = int(y_true.sum()), int((~y_true).sum())
    sens = float((pred & y_true).sum() / n_pos) if n_pos else float("nan")
    spec = float((~pred & ~y_true).sum() / n_neg) if n_neg else float("nan")
    return sens, spec


def threshold_for_sensitivity(y_true, y_score, target: float) -> float:
    """Highest threshold whose sensitivity is at least `target` (best specificity)."""
    y_true = np.asarray(y_true).astype(bool)
    if not y_true.any():
        return 0.5
    _, tpr, thresholds = roc_curve(y_true, y_score)
    ok = np.where(tpr >= target)[0]
    return float(thresholds[ok[0]])


def youden_threshold(y_true, y_score) -> float:
    """Threshold that maximises sensitivity + specificity - 1."""
    y_true = np.asarray(y_true).astype(bool)
    if y_true.all() or not y_true.any():
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return float(thresholds[1:][int(np.argmax((tpr - fpr)[1:]))])


def bootstrap_ci(fn, y_true, y_score, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for fn(y_true, y_score). NaN resamples are ignored."""
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    rng = np.random.default_rng(seed)
    n = y_true.size
    if n == 0:
        return float("nan"), float("nan")
    vals = np.array([fn(y_true[i], y_score[i]) for i in rng.integers(0, n, size=(n_boot, n))], dtype=np.float64)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    """Rows = true class, columns = predicted class."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)), 1)
    return cm


def quadratic_weighted_kappa(y_true, y_pred, n_classes: int) -> float:
    """Agreement on an ordinal scale; 1 = perfect, 0 = chance."""
    cm = confusion_matrix(y_true, y_pred, n_classes).astype(np.float64)
    n = cm.sum()
    if n == 0:
        return float("nan")
    idx = np.arange(n_classes)
    w = (idx[:, None] - idx[None, :]) ** 2 / (n_classes - 1) ** 2
    expected = np.outer(cm.sum(1), cm.sum(0)) / n
    denom = (w * expected).sum()
    return float(1.0 - (w * cm).sum() / denom) if denom > 0 else float("nan")


def screening_report(y_true, y_score, threshold: float, n_boot: int = 1000, seed: int = 0) -> dict:
    """AUC, sensitivity and specificity at `threshold`, each with a 95% bootstrap CI."""
    y_true, y_score = np.asarray(y_true).astype(int), np.asarray(y_score, dtype=np.float64)
    sens, spec = sens_spec(y_true, y_score, threshold)
    report = {
        "n": int(y_true.size),
        "n_positive": int(y_true.sum()),
        "threshold": float(threshold),
        "auc": roc_auc(y_true, y_score),
        "sensitivity": sens,
        "specificity": spec,
    }
    if n_boot:
        report["auc_ci"] = bootstrap_ci(roc_auc, y_true, y_score, n_boot, seed)
        report["sensitivity_ci"] = bootstrap_ci(lambda t, s: sens_spec(t, s, threshold)[0], y_true, y_score, n_boot, seed)
        report["specificity_ci"] = bootstrap_ci(lambda t, s: sens_spec(t, s, threshold)[1], y_true, y_score, n_boot, seed)
    return report


def crossfit_operating_point(y_true, y_score, target: float, groups=None, repeats: int = 20, seed: int = 0) -> dict:
    """Honest estimate of what a threshold tuned for `target` sensitivity delivers on new data.

    Repeatedly splits the data in two halves (whole groups at a time, e.g. both
    eyes of a patient), tunes the threshold on one half, applies it to the other,
    and swaps. Returns the mean and range over repeats of the pooled
    sensitivity and specificity.
    """
    y_true, y_score = np.asarray(y_true).astype(bool), np.asarray(y_score, dtype=np.float64)
    groups = np.arange(y_true.size) if groups is None else np.asarray(groups)
    names = np.unique(groups)
    rng = np.random.default_rng(seed)
    sens, spec = [], []
    for _ in range(repeats):
        half = np.isin(groups, rng.permutation(names)[: names.size // 2])
        predicted = np.zeros(y_true.size, dtype=bool)
        for fit, apply in ((half, ~half), (~half, half)):
            predicted[apply] = y_score[apply] >= threshold_for_sensitivity(y_true[fit], y_score[fit], target)
        sens.append(float((predicted & y_true).sum() / max(y_true.sum(), 1)))
        spec.append(float((~predicted & ~y_true).sum() / max((~y_true).sum(), 1)))
    return {
        "sensitivity": float(np.mean(sens)), "specificity": float(np.mean(spec)),
        "sensitivity_range": (float(np.min(sens)), float(np.max(sens))), "specificity_range": (float(np.min(spec)), float(np.max(spec))),
        "repeats": repeats,
    }


def crossfit_referral_rate(y_true, y_score, target: float, groups=None, repeats: int = 20, seed: int = 0) -> np.ndarray:
    """For each image, the share of cross-fitting repeats in which it was referred.

    Same procedure as crossfit_operating_point, kept per image so results can be
    broken down (e.g. by DR grade) without ever using an image to tune its own threshold.
    """
    y_true, y_score = np.asarray(y_true).astype(bool), np.asarray(y_score, dtype=np.float64)
    groups = np.arange(y_true.size) if groups is None else np.asarray(groups)
    names = np.unique(groups)
    rng = np.random.default_rng(seed)
    referred = np.zeros(y_true.size)
    for _ in range(repeats):
        half = np.isin(groups, rng.permutation(names)[: names.size // 2])
        for fit, apply in ((half, ~half), (~half, half)):
            referred[apply] += y_score[apply] >= threshold_for_sensitivity(y_true[fit], y_score[fit], target)
    return referred / repeats
