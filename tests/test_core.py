import json
from dataclasses import replace

import cv2
import numpy as np
import pytest
import torch

from fundalux import degrade as dg
from fundalux import metrics as M
from fundalux.gradcam import ensemble_grad_cam, grad_cam, lesion_overlap, overlay_heatmap
from fundalux.models import FundusNet, ensemble_probabilities, load_weights, save_weights, to_tensor
from fundalux.pipeline import Fundalux
from fundalux.preprocess import apply_crop, crop_fundus, disc_mask, enhance_fundus, find_fundus, model_input
from fundalux.quality import basic_checks, quality_hints, quality_stats
from fundalux.synthetic import synthetic_fundus


@pytest.fixture
def fundus():
    return synthetic_fundus(np.random.default_rng(0), width=480, height=320, grade=3)


def test_find_fundus_handles_cut_off_disc(fundus):
    img, _ = fundus
    geom = find_fundus(img)
    assert geom.found
    # The disc (radius 0.46 * 480) is cut off vertically: the square must use the full width.
    assert abs(geom.side - 2 * 0.46 * 480) <= 6
    assert geom.y0 < 0 and abs((geom.x0 + geom.side / 2) - 240) <= 3


def test_find_fundus_ignores_position_and_scale(fundus):
    img, _ = fundus
    canvas = np.zeros((700, 900, 3), np.uint8)
    canvas[300:620, 350:830] = img
    geom = find_fundus(canvas)
    assert geom.found and abs((geom.x0 + geom.side / 2) - 590) <= 4 and abs((geom.y0 + geom.side / 2) - 460) <= 4


def test_black_image_is_not_a_fundus():
    geom = find_fundus(np.zeros((300, 400, 3), np.uint8))
    assert not geom.found and geom.side == 300


def test_crop_is_square_and_masked(fundus):
    cropped, _ = crop_fundus(fundus[0], 128)
    assert cropped.shape == (128, 128, 3) and cropped.dtype == np.uint8
    assert cropped[~disc_mask(128)].max() == 0 and cropped[64, 64].max() > 50


def test_mask_follows_image_through_crop(fundus):
    img, lesions = fundus
    geom = find_fundus(img)
    mask = apply_crop(lesions["EX"].astype(np.float32), geom, 256, cv2.INTER_AREA) >= 0.05
    cropped = apply_crop(img, geom, 256)
    assert mask.any()
    # Exudates are drawn bright yellow: masked pixels must be much brighter in green than the rest.
    assert cropped[..., 1][mask].mean() > cropped[..., 1][~mask & disc_mask(256)].mean() + 40


def test_enhance_evens_out_brightness(fundus):
    cropped, _ = crop_fundus(fundus[0], 128)
    dark = (cropped * 0.3).astype(np.uint8)
    a, b = enhance_fundus(cropped), enhance_fundus(dark)
    inside = disc_mask(128)
    assert abs(float(a[inside].mean()) - float(b[inside].mean())) < 12
    assert b[~inside].max() == 0


@pytest.mark.parametrize("kind", dg.KINDS)
def test_degradations(fundus, kind):
    cropped, _ = crop_fundus(fundus[0], 128)
    rng = np.random.default_rng(1)
    mild, severe = dg.degrade(cropped, kind, 0.3, rng), dg.degrade(cropped, kind, 1.0, rng)
    assert mild.shape == cropped.shape and severe.dtype == np.uint8
    assert severe[~disc_mask(128)].max() == 0
    diff = lambda x: np.abs(x.astype(int) - cropped.astype(int)).mean()
    assert 0 < diff(mild) < diff(severe)
    assert np.array_equal(dg.degrade(cropped, kind, 0.0), cropped)


@pytest.mark.parametrize("kind", dg.KINDS)
def test_degradations_accept_uncropped_photos(fundus, kind):
    img = fundus[0]  # 480 x 320, not square: the app degrades the raw upload
    out = dg.degrade(img, kind, 0.8, np.random.default_rng(0))
    assert out.shape == img.shape and not np.array_equal(out, img)


def test_severe_blur_is_flagged_by_stats(fundus):
    cropped, _ = crop_fundus(fundus[0], 256)
    good, bad = quality_stats(cropped), quality_stats(dg.degrade(cropped, "blur", 1.0))
    assert bad["sharpness"] < 0.2 * good["sharpness"]
    hints = quality_hints(bad, {"sharpness": [0.5 * good["sharpness"], 1e9]})
    assert any("blur" in h.lower() for h in hints) and quality_hints(good, {"sharpness": [0.5 * good["sharpness"], 1e9]}) == []


def test_basic_checks(fundus):
    img, _ = fundus
    cropped, geom = crop_fundus(img, 128)
    assert basic_checks(img, geom, quality_stats(cropped)) == []
    black = np.zeros((300, 300, 3), np.uint8)
    cropped, geom = crop_fundus(black, 128)
    assert basic_checks(black, geom, quality_stats(cropped))
    blue = img[..., ::-1].copy()
    cropped, geom = crop_fundus(blue, 128)
    assert any("fundus" in r for r in basic_checks(blue, geom, quality_stats(cropped)))


def test_metrics_match_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = np.round(rng.normal(y * 0.8, 1.0), 1)  # rounded: plenty of tied scores
    assert M.roc_auc(y, s) == pytest.approx(sk.roc_auc_score(y, s))
    g, p = rng.integers(0, 5, 300), rng.integers(0, 5, 300)
    assert M.quadratic_weighted_kappa(g, p, 5) == pytest.approx(sk.cohen_kappa_score(g, p, weights="quadratic"))
    assert np.array_equal(M.confusion_matrix(g, p, 5), sk.confusion_matrix(g, p, labels=range(5)))


def test_threshold_for_sensitivity():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    s = rng.normal(y * 1.5, 1.0)
    t = M.threshold_for_sensitivity(y, s, 0.9)
    sens, spec = M.sens_spec(y, s, t)
    assert sens >= 0.9
    # It is the *highest* such threshold: the next higher score drops below the target.
    higher = np.sort(s[s > t])[0]
    assert M.sens_spec(y, s, higher)[0] < 0.9
    assert 0 < spec < 1
    tj = M.youden_threshold(y, s)
    assert sum(M.sens_spec(y, s, tj)) >= sum(M.sens_spec(y, s, t))


def test_metrics_degenerate_inputs():
    assert np.isnan(M.roc_auc([1, 1], [0.2, 0.9]))
    assert M.threshold_for_sensitivity([0, 0], [0.1, 0.2], 0.9) == 0.5
    rep = M.screening_report([0, 0, 0], [0.1, 0.2, 0.7], 0.5, n_boot=20)
    assert np.isnan(rep["sensitivity"]) and rep["specificity"] == pytest.approx(2 / 3)
    lo, hi = M.bootstrap_ci(M.roc_auc, np.array([0, 1] * 50), np.linspace(0, 1, 100), n_boot=50)
    assert 0 <= lo <= hi <= 1


def test_model_normalises_internally_and_roundtrips(tmp_path, fundus):
    torch.manual_seed(0)
    model = FundusNet("efficientnet_b0", num_grades=5).eval()
    cropped, _ = crop_fundus(fundus[0], 96)
    x = to_tensor(cropped).unsqueeze(0)
    assert x.dtype == torch.uint8
    with torch.no_grad():
        out = model(x)
        assert torch.allclose(out, model(x.float()))
    assert out.shape == (1, 6)
    assert not any(k in ("mean", "std") for k in model.state_dict())
    save_weights(model, tmp_path / "m.pt")
    clone = load_weights(FundusNet("efficientnet_b0", num_grades=5), tmp_path / "m.pt")
    with torch.no_grad():
        assert torch.allclose(out, clone(x), atol=1e-5)


@pytest.mark.parametrize("arch", ["efficientnet_b0", "resnet18", "convnext_tiny"])
def test_grad_cam(arch, fundus):
    torch.manual_seed(0)
    model = FundusNet(arch, num_grades=5)
    cropped, _ = crop_fundus(fundus[0], 96)
    x = torch.stack([to_tensor(cropped), to_tensor(cropped[:, ::-1].copy())])
    cams, out = grad_cam(model, x)
    assert cams.shape[0] == 2 and cams.min() >= 0 and cams.max() <= 1 + 1e-6
    assert out.shape == (2, 6) and not out.requires_grad
    assert overlay_heatmap(cropped, cams[0]).shape == cropped.shape
    for p in model.parameters():  # works on a frozen model too
        p.requires_grad_(False)
    assert grad_cam(model, x)[0].shape == cams.shape


def test_lesion_overlap_scores():
    valid = np.ones((64, 64), bool)
    lesion = np.zeros((64, 64), bool)
    lesion[10:20, 10:20] = True
    on = np.zeros((64, 64))
    on[12:18, 12:18] = 1.0
    off = np.zeros((64, 64))
    off[40:50, 40:50] = 1.0
    good, bad = lesion_overlap(on, lesion, valid), lesion_overlap(off, lesion, valid)
    assert good["pointing_hit"] == 1 and good["energy_in_mask"] == 1 and good["pixel_auc"] > 0.6
    assert bad["pointing_hit"] == 0 and bad["energy_in_mask"] == 0 and bad["pixel_auc"] < 0.5
    uniform = lesion_overlap(np.ones((64, 64)), lesion, valid)
    assert uniform["energy_in_mask"] == pytest.approx(uniform["lesion_area"]) and uniform["pixel_auc"] == pytest.approx(0.5)
    blank = lesion_overlap(np.zeros((64, 64)), lesion, valid)  # an empty heatmap is a miss, not a missing value
    assert blank["empty"] and blank["pointing_hit"] == 0 and blank["pixel_auc"] == 0.5 and not good["empty"]
    assert np.isnan(lesion_overlap(on, np.zeros((64, 64), bool), valid)["pixel_auc"])


def test_convnext_keeps_its_pretrained_norm_layer(tmp_path):
    torch.manual_seed(0)
    model = FundusNet("convnext_tiny", num_grades=5).eval()
    assert {"norm.weight", "norm.bias"} <= set(model.state_dict())
    save_weights(model, tmp_path / "c.pt")
    clone = load_weights(FundusNet("convnext_tiny", num_grades=5), tmp_path / "c.pt")
    x = torch.randint(0, 255, (1, 3, 64, 64), dtype=torch.uint8)
    with torch.no_grad():
        assert torch.allclose(model(x), clone(x), atol=1e-5)


def test_ensemble_probabilities():
    a = np.array([[0.0, 2.0, 0.0, 0.0, 0.0, 0.0], [4.0, 0.0, 0.0, 0.0, 0.0, 9.0]])
    b = np.array([[2.0, 0.0, 2.0, 0.0, 0.0, 0.0], [-4.0, 0.0, 0.0, 0.0, 0.0, 9.0]])
    referable, grades, members = ensemble_probabilities([a, torch.tensor(b)])
    sig = lambda z: 1 / (1 + np.exp(-z))
    assert referable == pytest.approx([(0.5 + sig(2)) / 2, 0.5]) and members.shape == (2, 2)
    assert grades.sum(axis=1) == pytest.approx([1, 1]) and grades[0, 0] == pytest.approx(grades[0, 1]) and grades[1].argmax() == 4
    assert ensemble_probabilities([a[:, :1]])[1] is None  # the quality gate has no grade outputs


def test_ensemble_grad_cam_feeds_each_model_its_own_size(fundus):
    torch.manual_seed(0)
    cropped, _ = crop_fundus(fundus[0], 96)
    small, large = FundusNet("efficientnet_b0", num_grades=5), FundusNet("resnet18", num_grades=5)
    cams, logits = ensemble_grad_cam([(small, 64), (large, 96)], [cropped, cropped[:, ::-1].copy()], 96)
    assert cams.shape == (2, 96, 96) and 0 <= cams.min() and cams.max() == pytest.approx(1.0)
    with torch.no_grad():
        direct = small.eval()(to_tensor(model_input(cropped, 64)).unsqueeze(0))
    assert torch.allclose(logits[0][0], direct[0], atol=1e-5) and logits[1].shape == (2, 6)
    assert model_input(cropped, 64).shape == (64, 64, 3)
    assert np.array_equal(model_input(cropped, 96), enhance_fundus(cropped))


def _write_bundle(folder, quality_bias, ensemble=True):
    torch.manual_seed(0)
    quality = FundusNet("efficientnet_b0", num_grades=0)
    with torch.no_grad():
        quality.head.weight.zero_()
        quality.head.bias.fill_(quality_bias)
    save_weights(quality, folder / "q.pt")
    if ensemble:
        members = [("efficientnet_b0", 96), ("resnet18", 64)]
        for i, (arch, _) in enumerate(members):
            save_weights(FundusNet(arch, num_grades=5), folder / f"dr{i}.pt")
        dr = {"models": [{"arch": a, "weights": f"dr{i}.pt", "input_size": s} for i, (a, s) in enumerate(members)], "num_grades": 5, "threshold": 0.5}
    else:  # layout written by the first, single-model notebook
        save_weights(FundusNet("efficientnet_b0", num_grades=5), folder / "dr.pt")
        dr = {"arch": "efficientnet_b0", "weights": "dr.pt", "num_grades": 5, "threshold": 0.5}
    config = {
        "image_size": 96,
        "min_input_side": 64,
        "dr": dr,
        "quality": {"arch": "efficientnet_b0", "weights": "q.pt", "input_size": 64, "threshold": 0.5, "hint_ranges": {"sharpness": [1e9, 1e12]}},
    }
    (folder / "config.json").write_text(json.dumps(config))


def test_pipeline_accepts_then_predicts(tmp_path, fundus):
    _write_bundle(tmp_path, quality_bias=-10.0)  # gate always says "gradable"
    result = Fundalux.load(tmp_path).predict(fundus[0])
    assert result.gradable and result.reasons == []
    assert 0 <= result.referable_score <= 1 and result.referable == (result.referable_score >= 0.5)
    assert set(result.member_scores) == {"efficientnet_b0", "resnet18"}
    assert result.referable_score == pytest.approx(np.mean(list(result.member_scores.values())))
    assert result.grade in range(5) and sum(result.grade_probs) == pytest.approx(1, abs=1e-4)
    assert result.heatmap.shape == (96, 96) and result.overlay.shape == (96, 96, 3) and result.enhanced.shape == (96, 96, 3)
    json.dumps(result.summary())


def test_pipeline_applies_a_site_threshold(tmp_path, fundus):
    from fundalux.sites import SiteProfile

    _write_bundle(tmp_path, quality_bias=-10.0)
    net = Fundalux.load(tmp_path)
    base = net.predict(fundus[0])
    score = base.referable_score
    strict = SiteProfile("Strict camp", threshold=min(score + 0.01, 1.0), target_sensitivity=0.93, n_images=100, n_referable=40, expected_sensitivity=0.9, expected_specificity=0.9)
    lenient = replace(strict, name="Lenient camp", threshold=max(score - 0.01, 0.0))
    a, b = net.predict(fundus[0], site=strict), net.predict(fundus[0], site=lenient)
    assert (a.referable, a.site, a.threshold) == (False, "Strict camp", strict.threshold)
    assert b.referable and "refer" in b.advice.lower() and "no referable" in a.advice.lower()
    # Re-deciding a cached result gives exactly what predicting at that site gives.
    for site in (strict, lenient):
        again = base.with_threshold(site.threshold, site.name)
        assert (again.referable, again.advice, again.site) == (net.predict(fundus[0], site=site).referable, net.predict(fundus[0], site=site).advice, site.name)
    assert base.site is None and base.threshold == 0.5
    gradable, s = net.referable_score(fundus[0])
    assert gradable and s == pytest.approx(score, abs=1e-6)


def test_pipeline_reads_single_model_bundles(tmp_path, fundus):
    _write_bundle(tmp_path, quality_bias=-10.0, ensemble=False)
    net = Fundalux.load(tmp_path)
    result = net.predict(fundus[0])
    cropped, _ = crop_fundus(fundus[0], 96)
    with torch.no_grad():
        expected = torch.sigmoid(net.dr_members[0][1].model(to_tensor(enhance_fundus(cropped)).unsqueeze(0))[0, 0])
    assert result.referable_score == pytest.approx(float(expected), abs=1e-6) and list(result.member_scores) == ["efficientnet_b0"]


def test_pipeline_refuses_ungradable(tmp_path, fundus):
    _write_bundle(tmp_path, quality_bias=10.0)  # gate always says "ungradable"
    net = Fundalux.load(tmp_path)
    result = net.predict(fundus[0])
    assert not result.gradable and result.referable is None and result.heatmap is None
    assert "retake" in result.advice.lower() and any("blur" in r.lower() for r in result.reasons)
    forced = net.predict(fundus[0], skip_gate=True)
    assert not forced.gradable and forced.referable_score is not None
    # Not a fundus at all: rejected by the rule checks without consulting the model.
    assert not net.predict(np.zeros((200, 200, 3), np.uint8)).gradable
    with pytest.raises(FileNotFoundError, match="models/README.md"):
        Fundalux.load(tmp_path / "missing")
