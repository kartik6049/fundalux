"""Export a bundle's PyTorch weights for ONNX Runtime serving (needs PyTorch).

    python -m fundalux.export [model_dir]

Writes <model_dir>/onnx/: one file per DR network with two outputs (feature
map and logits, for the closed-form Grad-CAM in inference.py), the quality
gate, and heads.npz with the classifier parameters the heatmaps need. Records
the export under "onnx" in config.json. The server then needs no PyTorch.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .inference import HEADS_FILE, read_config
from .models import FundusNet, load_weights

OPSET = 17


class _FeaturesAndLogits(nn.Module):
    def __init__(self, model: FundusNet):
        super().__init__()
        self.model = model

    def forward(self, x):
        feats = self.model.forward_features(x)
        return feats, self.model.forward_head(feats)


class _Logits(nn.Module):
    def __init__(self, model: FundusNet):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


def _export(module: nn.Module, size: int, path: Path, outputs: list[str]) -> None:
    args = (module.eval(), torch.zeros(1, 3, size, size), str(path))
    try:  # the classic exporter: dynamic batch, no extra packages
        legacy = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
        torch.onnx.export(*args, input_names=["image"], output_names=outputs, dynamic_axes={"image": {0: "batch"}, **{o: {0: "batch"} for o in outputs}}, opset_version=OPSET, **legacy)
    except Exception as exc:  # newer PyTorch: the default exporter (batch of one, which is all the server uses)
        print(f"  classic ONNX exporter failed ({type(exc).__name__}); using the default exporter")
        torch.onnx.export(*args, input_names=["image"], output_names=outputs)


def head_parameters(model: FundusNet) -> dict:
    """What the closed-form Grad-CAM needs from the classifier head."""
    norm = model.norm
    layernorm = isinstance(norm, nn.Module) and hasattr(norm, "normalized_shape")
    channels = model.head.in_features
    return {
        "weight": model.head.weight.detach().cpu().numpy().astype(np.float32),
        "kind": "layernorm" if layernorm else "linear",
        "gamma": (norm.weight.detach().cpu().numpy() if layernorm else np.ones(channels)).astype(np.float32),
        "eps": float(norm.eps) if layernorm else 0.0,
    }


def export_bundle(model_dir, members: list[tuple[str, FundusNet, int]] | None = None, quality: FundusNet | None = None, quality_size: int | None = None) -> dict:
    """Export the bundle in `model_dir` (or the given in-memory models) and update its config.json."""
    model_dir = Path(model_dir)
    config = read_config(model_dir)
    dr = config["dr"]
    if members is None:
        specs = dr.get("models") or [{"arch": dr["arch"], "weights": dr["weights"], "input_size": config["image_size"]}]
        members = [(s["arch"], load_weights(FundusNet(s["arch"], num_grades=dr["num_grades"]), model_dir / s["weights"]), s["input_size"]) for s in specs]
    if quality is None:
        quality = load_weights(FundusNet(config["quality"]["arch"], num_grades=0), model_dir / config["quality"]["weights"])
    quality_size = quality_size or config["quality"]["input_size"]

    out = model_dir / "onnx"
    out.mkdir(exist_ok=True)
    heads, spec = {}, {"dir": "onnx", "opset": OPSET, "members": [], "heads": HEADS_FILE}
    for i, (arch, model, size) in enumerate(members):
        model = model.cpu().eval()
        name = f"dr_{i + 1}_{arch}.onnx"
        _export(_FeaturesAndLogits(model), size, out / name, ["features", "logits"])
        for k, v in head_parameters(model).items():
            heads[f"m{i}_{k}"] = np.asarray(v)
        spec["members"].append({"arch": arch, "file": name, "input_size": int(size)})
    _export(_Logits(quality.cpu().eval()), quality_size, out / "quality.onnx", ["logits"])
    spec["quality"] = "quality.onnx"
    np.savez(out / HEADS_FILE, **heads)
    config["onnx"] = spec
    (model_dir / "config.json").write_text(json.dumps(config, indent=2))
    return spec


def main() -> None:
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("models")
    spec = export_bundle(model_dir)
    size = sum((model_dir / spec["dir"] / f).stat().st_size for f in [m["file"] for m in spec["members"]] + [spec["quality"], spec["heads"]])
    print(f"Exported {len(spec['members'])} networks and the quality gate to {model_dir / spec['dir']} ({size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
