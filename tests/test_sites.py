import json

import numpy as np
import pandas as pd
import pytest

from fundalux import metrics as M
from fundalux.calibrate import calibrate_site, read_labels
from fundalux.sites import CalibrationError, SiteRegistry, builtin_profiles, calibrate, model_fingerprint, patient_groups


def _scores(n=400, shift=0.0, seed=0):
    """Well-ranked scores whose scale is shifted per site, like the real cameras."""
    rng = np.random.default_rng(seed)
    y = rng.random(n) < 0.3
    s = 1 / (1 + np.exp(-(rng.normal(2.5 * y, 1.0) + shift)))
    return y.astype(int), s


def test_crossfit_is_honest():
    y, s = _scores(600)
    est = M.crossfit_operating_point(y, s, 0.93, repeats=10)
    assert 0.85 < est["sensitivity"] < 0.99 and 0 < est["specificity"] < 1
    lo, hi = est["specificity_range"]
    assert lo <= est["specificity"] <= hi
    # Tuning and testing on the same images would be optimistic; cross-fitting must not exceed that.
    t = M.threshold_for_sensitivity(y, s, 0.93)
    assert est["sensitivity"] <= M.sens_spec(y, s, t)[0] + 0.02


def test_crossfit_keeps_groups_together():
    y, s = _scores(200)
    groups = np.repeat(np.arange(100), 2)
    a = M.crossfit_operating_point(y, s, 0.9, groups=groups, repeats=5, seed=1)
    assert a == M.crossfit_operating_point(y, s, 0.9, groups=groups, repeats=5, seed=1)


def test_site_threshold_fixes_a_shifted_scale():
    """The motivating case: one site's scores run high, so the global threshold over-refers there."""
    y_a, s_a = _scores(800, shift=0.0, seed=1)
    y_b, s_b = _scores(800, shift=2.0, seed=2)
    global_t = M.threshold_for_sensitivity(y_a, s_a, 0.93)
    spec_global = M.sens_spec(y_b, s_b, global_t)[1]
    site = calibrate("B", y_b, s_b, 0.93)
    assert site.threshold > global_t
    assert site.expected_specificity > spec_global + 0.1
    assert site.expected_sensitivity > 0.88


def test_calibration_refuses_too_few_cases():
    y, s = _scores(60)
    with pytest.raises(CalibrationError, match="at least 20"):
        calibrate("tiny", y[:30], s[:30], 0.93)
    ok = calibrate("small", *_scores(90, seed=3), 0.93)
    assert ok.warnings and "imprecise" in ok.warnings[0]


def test_patient_groups():
    assert list(patient_groups("eyepacs", ["10_left", "10_right", "11_left"])) == ["10", "10", "11"]
    assert list(patient_groups("idrid", ["train_IDRiD_001", "test_IDRiD_001"])) == ["train_IDRiD_001", "test_IDRiD_001"]


def _predictions():
    rows = []
    for source, splits, shift in (("eyepacs", ("val", "test"), 0.0), ("aptos", ("val", "test"), 1.5), ("idrid", ("external",), 1.0), ("messidor", ("external",), 0.5)):
        for k, split in enumerate(splits):
            y, s = _scores(300, shift, seed=hash((source, split)) % 1000)
            ids = [f"{i // 2}_{'left' if i % 2 else 'right'}" if source == "eyepacs" else f"{source}{k}_{i}" for i in range(len(y))]
            rows.append(pd.DataFrame({"source": source, "image_id": ids, "split": split, "grade": np.where(y, 3, 0), "gate_pass": True, "p_ref": s}))
    df = pd.concat(rows, ignore_index=True)
    df.loc[df.index[:5], "gate_pass"] = False  # rejected images never enter a calibration
    return df


def test_builtin_profiles():
    profiles = {p.name.split(" ")[0]: p for p in builtin_profiles(_predictions(), 0.93)}
    assert set(profiles) == {"EyePACS", "APTOS", "IDRiD", "Messidor-2"}
    assert "validation" in profiles["APTOS"].estimate and "cross-fitted" in profiles["IDRiD"].estimate
    assert profiles["APTOS"].threshold > profiles["EyePACS"].threshold
    assert all(p.source == "built-in" for p in profiles.values())


def _config(run="r1"):
    return {"image_size": 96, "dr": {"models": [{"arch": "efficientnet_b0", "weights": "a.pt", "input_size": 96}], "num_grades": 5, "threshold": 0.2}, "trained_with": {"run": run}}


def test_registry_roundtrip_and_staleness(tmp_path):
    reg = SiteRegistry.load(tmp_path, _config())
    assert reg.profiles == []
    reg.put(calibrate("Camp A", *_scores(300), 0.93, source="calibrated in the app"))
    reg.put(calibrate("Camp A", *_scores(300, seed=5), 0.95))  # same name replaces
    again = SiteRegistry.load(tmp_path, _config())
    assert [p.name for p in again.usable()] == ["Camp A"] and again.get("Camp A").target_sensitivity == 0.95
    json.loads((tmp_path / "sites.json").read_text())  # valid JSON: no NaN
    # A retrained model scores on a different scale: old thresholds must not be offered.
    other = SiteRegistry.load(tmp_path, _config("r2"))
    assert other.usable() == [] and other.profiles[0].stale and any("different model" in w for w in other.profiles[0].warnings)
    other.save()
    assert SiteRegistry.load(tmp_path, _config("r1")).profiles[0].stale  # once stale, stays stale
    assert model_fingerprint(_config("r1")) != model_fingerprint(_config("r2"))
    again.remove("Camp A")
    assert SiteRegistry.load(tmp_path, _config()).profiles == []


def test_read_labels_formats():
    a = read_labels(pd.DataFrame({"Image": ["IMG_1.JPG", "img_2"], "grade": [3, 1]}))
    assert list(a.key) == ["img_1", "img_2"] and list(a.referable) == [1, 0]
    b = read_labels(pd.DataFrame({"id_code": ["x"], "referable": [1]}))
    assert list(b.referable) == [1]
    with pytest.raises(CalibrationError, match="image"):
        read_labels(pd.DataFrame({"name": ["x"], "grade": [1]}))
    with pytest.raises(CalibrationError, match="grade"):
        read_labels(pd.DataFrame({"image": ["x"], "label": [1]}))


def test_calibrate_site_joins_and_drops_rejected():
    y, s = _scores(200)
    keys = [f"p{i}" for i in range(200)]
    labels = pd.DataFrame({"key": keys + ["missing"], "referable": list(y) + [1]})
    gradable = np.ones(200, bool)
    gradable[:10] = False
    scored = pd.DataFrame({"key": keys, "gradable": gradable, "score": np.where(gradable, s, np.nan)})
    profile, summary = calibrate_site("Camp", labels, scored, 0.93, source="test")
    assert summary == {"labelled": 201, "matched": 200, "unmatched_labels": 1, "rejected_by_gate": 10}
    assert profile.n_images == 190 and profile.rejected_by_gate == 10
