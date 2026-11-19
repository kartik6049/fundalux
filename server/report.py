"""Everything the evidence and guide pages show, computed once from the model bundle.

Inputs: config.json, metrics.json, sites.json and reports/*.csv written by the
training notebook. Out-of-sample discipline follows the notebook: the global
threshold was frozen on validation data; per-site figures come from a separate
test split or from cross-fitting.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from fundalux import GRADE_NAMES, REFERABLE_MIN_GRADE
from fundalux import metrics as M
from fundalux.sites import BUILTIN_NAMES, SiteRegistry, patient_groups

# Evaluation sets: key -> (title, metrics.json name, rows of predictions.csv)
SETS = {
    "internal": ("Internal test (EyePACS + APTOS)", "Internal test (EyePACS + APTOS)", lambda p: p[p.split == "test"]),
    "eyepacs": ("EyePACS test", "EyePACS test", lambda p: p[(p.split == "test") & (p.source == "eyepacs")]),
    "aptos": ("APTOS 2019 test", "APTOS test", lambda p: p[(p.split == "test") & (p.source == "aptos")]),
    "idrid": ("IDRiD (external)", "IDRiD - external, India", lambda p: p[p.source == "idrid"]),
    "messidor": ("Messidor-2 (external)", "Messidor-2 - external", lambda p: p[p.source == "messidor"]),
}


def clean(obj):
    """JSON-safe: NaN/inf -> None, numpy -> Python."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _roc_points(y, s, max_points: int = 160) -> list[list[float]]:
    fpr, tpr, _ = M.roc_curve(y, s)
    idx = np.unique(np.linspace(0, len(fpr) - 1, min(max_points, len(fpr))).round().astype(int))
    return [[float(fpr[i]), float(tpr[i])] for i in idx]


def _site_for(key: str, registry: SiteRegistry | None):
    if registry is None or key == "internal":
        return None
    return registry.get(BUILTIN_NAMES[key]) if key in BUILTIN_NAMES else None


def _site_referral(key: str, frame: pd.DataFrame, site, target: float) -> np.ndarray | None:
    """Out-of-sample referral decision (or rate) per image at the site threshold."""
    if site is None:
        return None
    if key in ("eyepacs", "aptos"):  # site threshold tuned on the validation split; these are test images
        return (frame.p_ref.to_numpy() >= site.threshold).astype(float)
    return M.crossfit_referral_rate(frame.referable.to_numpy(), frame.p_ref.to_numpy(), target, groups=patient_groups(key, frame.image_id))


def build_report(model_dir) -> dict:
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    metrics = json.loads((model_dir / "metrics.json").read_text()) if (model_dir / "metrics.json").exists() else {}
    registry = SiteRegistry.load(model_dir, config)
    dr = metrics.get("dr", {})
    target = dr.get("validation", {}).get("target_sensitivity", 0.93)
    crit = metrics.get("success_criteria", {})
    members = config["dr"].get("models") or [{"arch": config["dr"]["arch"], "input_size": config["image_size"]}]
    training = metrics.get("dr_training", {})
    report = {
        "model": {
            "members": [{"arch": m["arch"], "input_size": m["input_size"]} for m in members],
            "trained": config.get("trained_with", {}),
            "global_threshold": config["dr"]["threshold"],
            "quality_threshold": config["quality"]["threshold"],
            "quality_arch": config["quality"]["arch"],
            "quality_input": config["quality"]["input_size"],
            "image_size": config["image_size"],
            "epochs": training.get("epochs"),
            "train_images": training.get("train_images"),
            "balanced_sources": training.get("balanced_sources"),
            "member_history": [{"arch": m["arch"], "best_auc": max((h["val_score"] for h in m["history"]), default=None), "epochs": len(m["history"])} for m in training.get("members", [])],
        },
        "targets": {"sensitivity": crit.get("target_sensitivity", 0.90), "specificity": crit.get("target_specificity", 0.85), "tuning_sensitivity": target},
        "data": metrics.get("data", {}),
        "validation": dr.get("validation", {}),
        "member_auc": dr.get("member_auc", {}),
        "quality_gate": metrics.get("quality_gate", {}),
        "gradcam": metrics.get("gradcam", {}),
        "robustness": metrics.get("robustness", {}).get("rows", []),
        "sites": [
            {"name": p.name, "threshold": p.threshold, "n_images": p.n_images, "n_referable": p.n_referable, "expected_sensitivity": p.expected_sensitivity,
             "expected_specificity": p.expected_specificity, "estimate": p.estimate, "source": p.source, "stale": p.stale, "warnings": p.warnings}
            for p in registry.profiles
        ],
        "sets": {},
    }

    pred_path = model_dir / "reports" / "predictions.csv"
    predictions = pd.read_csv(pred_path, dtype={"image_id": str}) if pred_path.exists() else None
    if predictions is not None:
        predictions["referable"] = (predictions["grade"] >= REFERABLE_MIN_GRADE).astype(int)
    for key, (title, metrics_name, select) in SETS.items():
        r = dr.get("results", {}).get(metrics_name)
        if r is None:
            continue
        entry = {
            "title": title,
            "graded": r["n"], "referable": r["n_positive"], "rejected_by_gate": r.get("rejected_by_gate"),
            "auc": r["auc"], "auc_ci": r.get("auc_ci"),
            "global": {"threshold": dr.get("threshold"), "sensitivity": r["sensitivity"], "sensitivity_ci": r.get("sensitivity_ci"), "specificity": r["specificity"], "specificity_ci": r.get("specificity_ci")},
            "kappa": r.get("grade_kappa"),
            "meets_target_global": bool(r["sensitivity"] > report["targets"]["sensitivity"] and r["specificity"] > report["targets"]["specificity"]),
        }
        site = _site_for(key, registry)
        if site is not None and not site.stale:
            entry["site"] = {"name": site.name, "threshold": site.threshold, "sensitivity": site.expected_sensitivity, "specificity": site.expected_specificity, "estimate": site.estimate,
                             "meets_target": bool(site.expected_sensitivity > report["targets"]["sensitivity"] and site.expected_specificity > report["targets"]["specificity"])}
        if predictions is not None:
            frame = select(predictions)
            frame = frame[(frame.grade >= 0) & frame.gate_pass.astype(bool) & frame.p_ref.notna()]
            y, s = frame.referable.to_numpy(), frame.p_ref.to_numpy()
            entry["roc"] = _roc_points(y, s)
            site_rate = _site_referral(key, frame, site if site is not None and not site.stale else None, target)
            by_grade = []
            for g in range(5):
                mask = frame.grade.to_numpy() == g
                row = {"grade": g, "name": GRADE_NAMES[g], "referable": g >= REFERABLE_MIN_GRADE, "n": int(mask.sum()),
                       "referred_global": float((s[mask] >= dr.get("threshold", 0.5)).mean()) if mask.any() else None}
                if site_rate is not None:
                    row["referred_site"] = float(site_rate[mask].mean()) if mask.any() else None
                by_grade.append(row)
            entry["by_grade"] = by_grade
            if "grade_pred" in frame:
                entry["confusion"] = M.confusion_matrix(frame.grade.astype(int), frame.grade_pred.astype(int), 5).tolist()
        report["sets"][key] = entry
    report["summary"] = _summary(report)
    return clean(report)


def _summary(report: dict) -> dict:
    """Headline figures, each traceable to a table on the evidence page."""
    sets = report["sets"]
    aucs = [e["auc"] for k, e in sets.items() if k != "internal" and e.get("auc") is not None]
    severe = [g["referred_global"] for e in sets.values() for g in e.get("by_grade", []) if g["grade"] >= 3 and g["n"] and g["referred_global"] is not None]
    with_site = [e["site"] for e in sets.values() if e.get("site")]
    gate = report["quality_gate"].get("test", {}).get("test: real EyeQ labels only", {})
    cam = report["gradcam"].get("trained", {})
    glob_spec = [e["global"]["specificity"] for k, e in sets.items() if k != "internal"]
    kappas = [e["kappa"] for k, e in sets.items() if k != "internal" and e.get("kappa") is not None]
    moderate = [r for r in report["robustness"] if abs(r["severity"] - 0.3) < 1e-6 and r.get("spec_with_gate") is not None]
    clean_rows = [r for r in report["robustness"] if r["severity"] == 0]
    return {
        "auc_min": min(aucs) if aucs else None, "auc_max": max(aucs) if aucs else None,
        "global_spec_min": min(glob_spec) if glob_spec else None, "global_spec_max": max(glob_spec) if glob_spec else None,
        "kappa_min": min(kappas) if kappas else None, "kappa_max": max(kappas) if kappas else None,
        "moderate_spec_min": min(r["spec_with_gate"] for r in moderate) if moderate else None,
        "moderate_spec_max": max(r["spec_with_gate"] for r in moderate) if moderate else None,
        "clean_spec": clean_rows[0]["spec_with_gate"] if clean_rows else None,
        "severe_referred_min": min(severe) if severe else None,
        "sites_meeting": sum(s["meets_target"] for s in with_site), "sites_total": len(with_site),
        "gate_auc": gate.get("AUC"), "pointing": cam.get("pointing hit rate"),
        "pointing_baseline": report["gradcam"].get("baseline", {}).get("pointing hit rate"),
    }
