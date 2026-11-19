"""Inference backends: ONNX Runtime (default, no PyTorch needed) and PyTorch.

Both expose the same small interface, so the pipeline does not care which one
runs. Each DR network is exported with two outputs, its last feature map and
its logits. Because the head after that feature map is global-average pooling
(optionally a LayerNorm, for ConvNeXt) and one linear layer, the Grad-CAM
gradient has a closed form, so heatmaps need no automatic differentiation:

    linear head:     d logit / d F[c,h,w] = W[c] / HW
    LayerNorm head:  d logit / d p = (g - mean(g) - p_hat * mean(g * p_hat)) / sigma,  g = W * gamma,
                     with p the pooled features, then divided by HW

Grad-CAM weights each channel by the spatial mean of that gradient, so this is
exactly what torch.autograd gives (tests/test_inference.py checks it).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

HEADS_FILE = "heads.npz"


def to_input(images: list[np.ndarray]) -> np.ndarray:
    """RGB uint8 HxWx3 images (same size) -> float32 NCHW batch in 0-255; the networks normalise internally."""
    return np.ascontiguousarray(np.stack(images).transpose(0, 3, 1, 2), dtype=np.float32)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def ensemble_probabilities(logits) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Combine the members of an ensemble by averaging their probabilities.

    `logits` is a list with one [N, 1 + grades] array (or tensor) per model.
    Returns (P(positive) [N], grade probabilities [N, grades] or None, each member's P(positive) [members, N]).
    """
    arrays = []
    for x in logits:
        if hasattr(x, "detach"):  # a torch tensor
            x = x.detach().cpu().numpy()
        arrays.append(np.asarray(x, dtype=np.float64))
    positive = np.stack([sigmoid(x[:, 0]) for x in arrays])
    grades = None
    if arrays[0].shape[1] > 1:
        shifted = [x[:, 1:] - x[:, 1:].max(axis=1, keepdims=True) for x in arrays]
        grades = np.mean([np.exp(s) / np.exp(s).sum(axis=1, keepdims=True) for s in shifted], axis=0)
    return positive.mean(axis=0), grades, positive


def normalise_cams(cams: np.ndarray) -> np.ndarray:
    """ReLU, then scale each map to a peak of 1 (an all-zero map stays zero)."""
    cams = np.maximum(cams, 0.0)
    peak = cams.reshape(len(cams), -1).max(axis=1).reshape(-1, 1, 1)
    return np.where(peak > 1e-12, cams / np.maximum(peak, 1e-12), 0.0).astype(np.float32)


def cam_from_features(feats: np.ndarray, head: dict, target: int = 0) -> np.ndarray:
    """Grad-CAM of logit `target` from a feature map [B, C, h, w] and the head parameters."""
    feats = feats.astype(np.float64)
    w = head["weight"][target].astype(np.float64)  # [C]
    if head["kind"] == "layernorm":
        pooled = feats.mean(axis=(2, 3))  # [B, C]
        mu = pooled.mean(axis=1, keepdims=True)
        sigma = np.sqrt(((pooled - mu) ** 2).mean(axis=1, keepdims=True) + float(head["eps"]))
        p_hat = (pooled - mu) / sigma
        g = w * head["gamma"].astype(np.float64)  # [C]
        weights = (g - g.mean() - p_hat * (g * p_hat).mean(axis=1, keepdims=True)) / sigma  # [B, C]
    else:
        weights = np.broadcast_to(w, (feats.shape[0], w.size))
    # The 1/HW factor is common to every channel and disappears in the normalisation.
    return normalise_cams(np.einsum("bc,bchw->bhw", weights, feats))


class OnnxMember:
    """One DR network in ONNX Runtime."""

    def __init__(self, name: str, path, size: int, head: dict, session_options=None):
        import onnxruntime as ort

        self.name, self.size, self.head = name, int(size), head
        self.session = ort.InferenceSession(str(path), sess_options=session_options, providers=["CPUExecutionProvider"])

    def run(self, x: np.ndarray, with_cam: bool) -> tuple[np.ndarray, np.ndarray | None]:
        """x: float32 NCHW. Returns (logits [B, outputs], heatmaps [B, h, w] or None)."""
        feats, logits = self.session.run(["features", "logits"], {"image": x})
        return logits, (cam_from_features(feats, self.head) if with_cam else None)


class OnnxGate:
    def __init__(self, path, session_options=None):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(path), sess_options=session_options, providers=["CPUExecutionProvider"])

    def ungradable(self, x: np.ndarray) -> float:
        return float(sigmoid(self.session.run(["logits"], {"image": x})[0][0, 0]))


class TorchMember:
    """One DR network in PyTorch (used for training-time checks and bundles without ONNX)."""

    def __init__(self, name: str, model, size: int):
        self.name, self.model, self.size = name, model.eval(), int(size)

    def run(self, x: np.ndarray, with_cam: bool):
        import torch

        from .gradcam import grad_cam

        xt = torch.from_numpy(x)
        if with_cam:
            cams, out = grad_cam(self.model, xt)
            return out.float().numpy(), cams
        with torch.no_grad():
            return self.model(xt).float().numpy(), None


class TorchGate:
    def __init__(self, model):
        self.model = model.eval()

    def ungradable(self, x: np.ndarray) -> float:
        import torch

        with torch.no_grad():
            return float(torch.sigmoid(self.model(torch.from_numpy(x))[0, 0]))


def session_options():
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # The networks run one after another, each in its own session. Spinning (busy-waiting) worker
    # threads would keep every idle session burning CPU and slow down the one that is working.
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
    threads = int(os.environ.get("FUNDALUX_THREADS", "0"))  # 0 = let ONNX Runtime decide
    if threads:
        opts.intra_op_num_threads = threads
    return opts


def onnx_available(model_dir, config: dict) -> bool:
    spec = config.get("onnx")
    if not spec:
        return False
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    folder = Path(model_dir) / spec["dir"]
    files = [m["file"] for m in spec["members"]] + [spec["quality"], spec["heads"]]
    return all((folder / f).exists() for f in files)


def load_onnx(model_dir, config: dict):
    """(members, gate) from the bundle's ONNX export (see fundalux/export.py)."""
    spec, folder = config["onnx"], Path(model_dir) / config["onnx"]["dir"]
    heads = np.load(folder / spec["heads"], allow_pickle=False)
    opts = session_options()
    members = []
    for i, m in enumerate(spec["members"]):
        head = {"weight": heads[f"m{i}_weight"], "kind": str(heads[f"m{i}_kind"]), "gamma": heads[f"m{i}_gamma"], "eps": float(heads[f"m{i}_eps"])}
        members.append(OnnxMember(m["arch"], folder / m["file"], m["input_size"], head, opts))
    return members, OnnxGate(folder / spec["quality"], opts)


def load_torch(model_dir, config: dict):
    from .models import FundusNet, load_weights

    model_dir, dr = Path(model_dir), config["dr"]
    # Bundles from the first, single-model notebook describe their one model directly under "dr".
    specs = dr.get("models") or [{"arch": dr["arch"], "weights": dr["weights"], "input_size": config["image_size"]}]
    members = [TorchMember(s["arch"], load_weights(FundusNet(s["arch"], num_grades=dr["num_grades"]), model_dir / s["weights"]), s["input_size"]) for s in specs]
    gate = TorchGate(load_weights(FundusNet(config["quality"]["arch"], num_grades=0), model_dir / config["quality"]["weights"]))
    return members, gate


def load_backend(model_dir, config: dict, backend: str = "auto"):
    """Returns (backend name, members, gate). `backend` is auto, onnx or torch (env FUNDALUX_BACKEND overrides auto)."""
    if backend == "auto":
        backend = os.environ.get("FUNDALUX_BACKEND", "auto")
    if backend == "onnx" or (backend == "auto" and onnx_available(model_dir, config)):
        return ("onnx", *load_onnx(model_dir, config))
    return ("torch", *load_torch(model_dir, config))


def read_config(model_dir) -> dict:
    return json.loads((Path(model_dir) / "config.json").read_text())
