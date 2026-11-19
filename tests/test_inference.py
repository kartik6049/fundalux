"""ONNX Runtime serving must give exactly what PyTorch gives, heatmaps included."""

import json

import numpy as np
import pytest
import torch

from fundalux.export import export_bundle, head_parameters
from fundalux.gradcam import grad_cam
from fundalux.inference import cam_from_features, ensemble_probabilities, normalise_cams
from fundalux.models import FundusNet, to_tensor
from fundalux.pipeline import Fundalux
from fundalux.preprocess import crop_fundus
from test_core import _write_bundle, fundus  # noqa: F401  (fixture)


@pytest.mark.parametrize("arch", ["efficientnet_b0", "convnext_tiny", "resnet18"])
def test_closed_form_grad_cam_matches_autograd(arch, fundus):
    torch.manual_seed(0)
    model = FundusNet(arch, num_grades=5).eval()
    with torch.no_grad():  # a non-trivial LayerNorm so its gradient term matters
        if hasattr(model.norm, "weight"):
            model.norm.weight.uniform_(0.5, 1.5)
    cropped, _ = crop_fundus(fundus[0], 96)
    x = torch.stack([to_tensor(cropped), to_tensor(cropped[::-1].copy())])
    reference, _ = grad_cam(model, x)
    with torch.no_grad():
        feats = model.forward_features(x).numpy()
    ours = cam_from_features(feats, head_parameters(model))
    assert np.allclose(ours, reference, atol=1e-4)


def test_normalise_and_ensemble_helpers():
    cams = normalise_cams(np.array([[[-1.0, 2.0]], [[0.0, 0.0]]]))
    assert cams.tolist() == [[[0.0, 1.0]], [[0.0, 0.0]]]
    p, grades, members = ensemble_probabilities([np.zeros((1, 6)), torch.zeros(1, 6)])
    assert p[0] == pytest.approx(0.5) and grades.shape == (1, 5) and members.shape == (2, 1)


def test_exported_bundle_matches_torch(tmp_path, fundus):
    _write_bundle(tmp_path, quality_bias=-10.0)
    spec = export_bundle(tmp_path)
    assert json.loads((tmp_path / "config.json").read_text())["onnx"] == spec
    torch_net, onnx_net = Fundalux.load(tmp_path, backend="torch"), Fundalux.load(tmp_path, backend="onnx")
    assert (torch_net.backend, onnx_net.backend, Fundalux.load(tmp_path).backend) == ("torch", "onnx", "onnx")
    a, b = torch_net.predict(fundus[0]), onnx_net.predict(fundus[0])
    assert b.referable_score == pytest.approx(a.referable_score, abs=1e-5)
    assert b.ungradable_score == pytest.approx(a.ungradable_score, abs=1e-6)
    assert np.allclose(a.heatmap, b.heatmap, atol=1e-4) and np.allclose(a.grade_probs, b.grade_probs, atol=1e-5)
    assert a.member_scores.keys() == b.member_scores.keys()
    assert onnx_net.referable_score(fundus[0])[1] == pytest.approx(a.referable_score, abs=1e-5)


def test_missing_onnx_files_fall_back_to_torch(tmp_path, fundus):
    _write_bundle(tmp_path, quality_bias=-10.0)
    export_bundle(tmp_path)
    (tmp_path / "onnx" / "quality.onnx").unlink()
    assert Fundalux.load(tmp_path).backend == "torch"
