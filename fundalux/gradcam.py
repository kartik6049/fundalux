"""Grad-CAM for FundusNet, plus overlap scores against lesion masks."""

from __future__ import annotations

import cv2
import numpy as np
import torch

from .heatmap import overlay_heatmap, resize_cam  # noqa: F401  (re-exported for the notebook)
from .models import FundusNet, to_tensor
from .preprocess import model_input


def grad_cam(model: FundusNet, x: torch.Tensor, target: int = 0) -> tuple[np.ndarray, torch.Tensor]:
    """Grad-CAM of output column `target` w.r.t. the last conv feature map.

    Returns (cams, outputs): cams is float32 [B, h, w], each scaled to [0, 1]
    (all zeros when nothing pushes the output up); outputs are the raw logits.
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        feats = model.forward_features(x)
    with torch.enable_grad():
        feats = feats.detach().requires_grad_(True)
        out = model.forward_head(feats)
        grads = torch.autograd.grad(out[:, target].sum(), feats)[0]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * feats).sum(dim=1)).detach().float()
    peak = cam.flatten(1).max(dim=1).values.clamp_min(1e-12).view(-1, 1, 1)
    cam = torch.where(peak > 1e-12, cam / peak, torch.zeros_like(cam))
    model.train(was_training)
    return cam.cpu().numpy(), out.detach()


def ensemble_grad_cam(members, cropped: list[np.ndarray], out_size: int, device=None) -> tuple[np.ndarray, list[torch.Tensor]]:
    """Grad-CAM of an ensemble for a batch of cropped (not yet enhanced) images.

    `members` is [(FundusNet, input size), ...]. Every model explains its own
    prediction at its own input size; the heatmaps are averaged at `out_size`
    and rescaled to [0, 1]. Returns (heatmaps [B, out_size, out_size], one
    logits tensor [B, outputs] per member, on the CPU).
    """
    total = np.zeros((len(cropped), out_size, out_size), np.float32)
    logits = []
    for model, size in members:
        x = torch.stack([to_tensor(model_input(c, size)) for c in cropped])
        cams, out = grad_cam(model, x if device is None else x.to(device))
        total += np.stack([resize_cam(c, out_size) for c in cams])
        logits.append(out.float().cpu())
    peak = total.reshape(len(cropped), -1).max(axis=1).reshape(-1, 1, 1)
    return np.where(peak > 0, total / np.maximum(peak, 1e-12), 0.0).astype(np.float32), logits


def lesion_overlap(cam: np.ndarray, lesion: np.ndarray, valid: np.ndarray, tolerance_frac: float = 0.04) -> dict:
    """Score how well a heatmap agrees with a lesion mask (all arrays HxW).

    pointing_hit   - heatmap maximum falls on a lesion (within a small tolerance)
    energy_in_mask - share of heatmap mass that lies on lesion pixels
    lesion_area    - share of the retina covered by lesions (the chance level for
                     energy_in_mask: a uniform heatmap scores exactly this)
    pixel_auc      - AUC of the heatmap as a per-pixel lesion detector (0.5 = chance)
    empty          - the heatmap is all zeros; it then scores as a miss / chance level

    Scores are NaN only when the mask has no lesion (or nothing but lesion) to compare with.
    """
    from .metrics import roc_auc

    lesion = lesion.astype(bool) & valid
    cam = np.where(valid, cam, 0.0).astype(np.float64)
    n_valid, n_lesion = int(valid.sum()), int(lesion.sum())
    area = n_lesion / max(n_valid, 1)
    if n_lesion == 0 or n_lesion == n_valid:
        return {"pointing_hit": np.nan, "energy_in_mask": np.nan, "lesion_area": area, "pixel_auc": np.nan, "empty": bool(cam.sum() <= 0)}
    if cam.sum() <= 0:
        return {"pointing_hit": 0.0, "energy_in_mask": area, "lesion_area": area, "pixel_auc": 0.5, "empty": True}
    k = max(1, int(round(tolerance_frac * cam.shape[0])))
    near = cv2.dilate(lesion.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))) > 0
    peak = np.unravel_index(int(np.argmax(cam)), cam.shape)
    return {
        "pointing_hit": float(near[peak]),
        "energy_in_mask": float(cam[lesion].sum() / cam.sum()),
        "lesion_area": area,
        "pixel_auc": roc_auc(lesion[valid].astype(int), cam[valid]),
        "empty": False,
    }
