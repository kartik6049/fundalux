# Source of Fundalux_Kaggle_Training.ipynb. Edit this file, then run
#   python scripts/build_notebook.py
# Cells are separated by "# %%" lines; "# %% [package]" is replaced by the
# fundalux/ source files.

# %% [markdown]
# # Fundalux — train the screening models (Kaggle notebook)
#
# This notebook trains the two models behind Fundalux and measures them against the screening targets
# (sensitivity above 90% and specificity above 85% for referable DR on held-out photographs):
#
# 1. **Image-quality gate** — rejects photographs that cannot be graded.
# 2. **DR classifier** — an ensemble of three networks (EfficientNet-B3, EfficientNetV2-S, ConvNeXt-Tiny) for
#    referable DR (moderate NPDR or worse) plus the 5-level grade, with Grad-CAM heatmaps.
#
# **Before you run it** (right-hand sidebar):
#
# 1. *Session options → Accelerator →* **GPU T4 x2**
# 2. *Session options → Internet →* **On** (needed once, for the ImageNet starting weights)
# 3. *Input → Add Input* — add the four datasets listed in section 4. (If you skip this, the notebook attaches
#    or downloads them itself.)
#
# Then **Save Version → Save & Run All (Commit)**. The run continues on Kaggle's servers with the browser
# closed, so there is nothing to disconnect from. Nothing has to be uploaded and no cloud drive is involved.
# Expect several hours: three models are trained at up to 512 px. Training is cut short rather than run past
# `TIME_BUDGET_HOURS`, to stay inside Kaggle's 12-hour limit.
#
# **Result:** `fundalux_outputs.zip` in the notebook's **Output**. Download it and unzip it into the `models/`
# folder of the project; `models/README.md` lists what goes where.

# %% [markdown]
# ## 1. Settings

# %%
import os

# ---- Things you may want to change -----------------------------------------
FRESH_START = False       # True = ignore saved checkpoints and train from scratch

IMG_SIZE = 512            # every image is cropped to the retina and cached at this size
MODELS = [                # the DR ensemble; "size" is the input size of that model, "batch" and "lr" are its own
    {"arch": "efficientnet_b3", "size": 512, "batch": 12, "lr": 2e-4},    # the model of the first run, now at 512 px
    {"arch": "efficientnet_v2_s", "size": 448, "batch": 12, "lr": 2e-4},  # 2021: faster-training EfficientNet
    {"arch": "convnext_tiny", "size": 448, "batch": 12, "lr": 1e-4},      # 2022: the newest design of the three
]
EPOCHS = 8
BALANCE_SOURCES = True    # sample every source dataset with the same share of referable cases (see section 7)
MAX_EYEPACS = None        # e.g. 8000 to use a random part of EyePACS for a quick first run
TIME_BUDGET_HOURS = 10.5  # Kaggle stops a session at 12 h; training is cut short rather than cross this
EXPORT_ONNX = True        # ONNX copies for the web app (it runs without PyTorch); doubles the size of the zip
# What the quality gate learns to call gradable. Run 2 (16 Nov 2026) used 0.05-0.3 / 0.7-1.0: photographs degraded
# to severity 0.3 were accepted while the classifier's specificity fell to 53-79%. From run 3 on, the gate learns to
# refuse from 0.3. (Chosen after seeing run 2's test results; the robustness section re-measures it.)
GATE_GRADABLE_DEGRADATION = (0.03, 0.15)
GATE_UNGRADABLE_DEGRADATION = (0.3, 1.0)

QUALITY_ARCH = "efficientnet_b0"
QUALITY_SIZE = 256
QUALITY_EPOCHS = 4
QUALITY_BATCH = 32
QUALITY_LR = 3e-4

TARGET_SENSITIVITY = 0.90  # screening targets on held-out data
TARGET_SPECIFICITY = 0.85
VAL_SENSITIVITY = 0.93     # the decision threshold is tuned to this on validation data (safety margin)
SEED = 42

DATASETS = {               # public Kaggle datasets: name -> "owner/dataset"
    "eyepacs": "tanlikesmath/diabetic-retinopathy-resized",
    "aptos": "mariaherrerot/aptos2019",
    "idrid": "aaryapatel98/indian-diabetic-retinopathy-image-dataset",
    "messidor": "mariaherrerot/messidor2preprocess",
}
LOCAL_DATA = {}            # optional, e.g. {"aptos": "/kaggle/input/my-aptos-upload"} to use a copy you uploaded yourself
EYEQ_URL = "https://raw.githubusercontent.com/HzFu/EyeQ/master/data/Label_EyeQ_train.csv"

# ---- Smoke test: a tiny synthetic run that only checks the code. Leave it alone. ----
SMOKE_TEST = os.environ.get("FUNDALUX_SMOKE") == "1"
if SMOKE_TEST:
    IMG_SIZE, EPOCHS = 96, 2
    MODELS = [{"arch": "efficientnet_b0", "size": 96, "batch": 8, "lr": 3e-4}, {"arch": "efficientnet_v2_s", "size": 64, "batch": 8, "lr": 3e-4}, {"arch": "convnext_tiny", "size": 64, "batch": 8, "lr": 1e-4}]
    QUALITY_SIZE, QUALITY_EPOCHS, QUALITY_BATCH = 64, 1, 8

# %%
import gc
import json
import math
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

RUN_START = time.time()
IN_KAGGLE = os.path.isdir("/kaggle/working")
IN_COLAB = "google.colab" in sys.modules
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
print(f"torch {torch.__version__} | device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if USE_AMP else ""))


def online(host="download.pytorch.org"):
    try:
        socket.create_connection((host, 443), timeout=8).close()
        return True
    except OSError:
        return False


if not SMOKE_TEST:
    if not USE_AMP:
        raise RuntimeError("No GPU found. Kaggle: Session options > Accelerator > GPU T4 x2. (Colab: Runtime > Change runtime type > T4 GPU.) Then run everything again.")
    if not online():
        raise RuntimeError("No internet access. Kaggle: Session options > Internet > On (your account must be phone-verified). It is needed for the ImageNet starting weights.")

# WORK is what you get back at the end; SCRATCH holds the big temporary files (on Kaggle it is not saved).
WORK = Path("/kaggle/working") if IN_KAGGLE else Path("/content") if IN_COLAB else Path.cwd()
SCRATCH = Path("/kaggle/temp") if IN_KAGGLE else WORK
SCRATCH.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)
if str(WORK) not in sys.path:
    sys.path.insert(0, str(WORK))
os.makedirs("fundalux", exist_ok=True)

cv2.setNumThreads(0)  # OpenCV threads and forked DataLoader workers do not mix
torch.manual_seed(SEED)
np.random.seed(SEED)

# %% [markdown]
# ## 2. Project code
#
# The next cells write the `fundalux` Python package next to this notebook. It is the same code the local app
# uses, so an image is preprocessed in exactly the same way during training and at screening time.
# (These cells are generated from the project's `fundalux/` folder — edit the files there, not here.
# If you do change one of these cells, restart the session before running the rest.)

# %% [package]

# %%
import fundalux
from fundalux import GRADE_NAMES, REFERABLE_MIN_GRADE
from fundalux import datasets as D
from fundalux import metrics as M
from fundalux.data import DRDataset, QualityDataset, worker_init
from fundalux.degrade import KINDS, degrade
from fundalux.gradcam import ensemble_grad_cam, lesion_overlap, overlay_heatmap
from fundalux.models import FundusNet, ensemble_probabilities, load_weights, save_weights, to_tensor
from fundalux.export import export_bundle
from fundalux.pipeline import Fundalux
from fundalux.sites import SiteRegistry, builtin_profiles, model_fingerprint
from fundalux.preprocess import correct_illumination, crop_fundus, enhance_fundus, model_input, read_image, retina_mask

print("fundalux", fundalux.__version__)

# %% [markdown]
# ## 3. Storage
#
# * `/kaggle/working` — saved with the notebook: the result zip and the training checkpoints.
# * `/kaggle/temp` — discarded when the session ends: the cropped-image cache (about 2 GB).
#
# A checkpoint is written after every epoch. If an *interactive* session is interrupted, re-running the notebook
# continues from the last finished epoch, provided *Session options → Persistence* is set to **Files only**.
# A committed run (*Save & Run All*) is not affected by the browser at all.

# %%
RUN = "+".join(f"{m['arch']}@{m['size']}" for m in MODELS) + ("_balanced" if BALANCE_SOURCES else "") + (f"_sub{MAX_EYEPACS}" if MAX_EYEPACS else "") + ("_smoke" if SMOKE_TEST else "")
CACHE_TAG = f"cache_{IMG_SIZE}" + (f"_sub{MAX_EYEPACS}" if MAX_EYEPACS else "") + ("_smoke" if SMOKE_TEST else "")
DATA_DIR, CACHE_DIR, OUT_DIR = SCRATCH / "data", SCRATCH / CACHE_TAG, SCRATCH / "fundalux_outputs"
CKPT_DIR = WORK / "checkpoints"
FIG_DIR, REPORT_DIR = OUT_DIR / "figures", OUT_DIR / "reports"
for folder in (DATA_DIR, CKPT_DIR, FIG_DIR, REPORT_DIR):
    folder.mkdir(parents=True, exist_ok=True)
INDEX_CSV, SEG_CSV = CACHE_DIR / "index.csv", CACHE_DIR / "idrid_seg.csv"
print(f"Results and checkpoints: {WORK}\nTemporary files:         {SCRATCH}")

metrics = {"run": RUN, "smoke_test": SMOKE_TEST, "image_size": IMG_SIZE}  # everything measured below ends up in metrics.json


def show(fig, name):
    """Save a figure into the output bundle and display it."""
    fig.savefig(FIG_DIR / f"{name}.png", dpi=130, bbox_inches="tight")
    plt.show()
    plt.close(fig)

# %% [markdown]
# ## 4. Data
#
# | Dataset | Add this Kaggle dataset as Input | Role | Images |
# |---|---|---|---|
# | EyePACS (Kaggle 2015, resized) | `tanlikesmath/diabetic-retinopathy-resized` | training / validation / test | ~35,000 |
# | APTOS 2019 (Aravind Eye Hospital, India) | `mariaherrerot/aptos2019` | training / validation / test | ~3,660 |
# | IDRiD (India, with lesion masks) | `aaryapatel98/indian-diabetic-retinopathy-image-dataset` | **external test** + Grad-CAM check — never trained on | 516 + 81 |
# | Messidor-2 (France) | `mariaherrerot/messidor2preprocess` | **external test** — never trained on | 1,744 |
# | EyeQ | fetched from GitHub | quality labels (good / usable / reject) for 12,543 EyePACS images | labels only |
#
# Each dataset is looked for in this order: `LOCAL_DATA` (your own upload) → the notebook's Inputs
# (`/kaggle/input`) → attached or downloaded automatically.
# Every image is cropped to the retina once and stored at `IMG_SIZE`; training then reads the small copies.

# %%
def fetch_dataset(name):
    """Return a folder holding the dataset: your own copy, a notebook Input, else attach/download it."""
    if name in LOCAL_DATA:
        return LOCAL_DATA[name]
    slug = DATASETS[name]
    mounted = D.find_kaggle_input(slug)
    if mounted:
        return mounted
    try:
        import kagglehub

        return kagglehub.dataset_download(slug)  # inside a Kaggle notebook this attaches the dataset
    except Exception as exc:
        print(f"kagglehub could not fetch {slug} ({type(exc).__name__}: {exc}); trying a direct download.")
    dest, archive = DATA_DIR / name, DATA_DIR / f"{name}.zip"
    url = f"https://www.kaggle.com/api/v1/datasets/download/{slug}"
    subprocess.run(["curl", "-L", "--fail", "--retry", "3", "-sS", "-o", str(archive), url], check=True)
    subprocess.run(["unzip", "-q", "-o", str(archive), "-d", str(dest)], check=True)
    archive.unlink()
    return str(dest)


def save_raw_examples(index, n=4):
    """Keep a few small raw photographs so the preprocessing figure can show the 'before'."""
    (CACHE_DIR / "examples").mkdir(parents=True, exist_ok=True)
    picks = index.groupby("source").head(1).head(n)
    for i, path in enumerate(picks["path"]):
        raw = read_image(path)
        scale = 512 / max(raw.shape[:2])
        raw = cv2.resize(raw, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else raw
        cv2.imwrite(str(CACHE_DIR / "examples" / f"raw_{i}.jpg"), cv2.cvtColor(raw, cv2.COLOR_RGB2BGR))


def prepare_data():
    t0 = time.time()
    if SMOKE_TEST:
        from fundalux.synthetic import make_fake_datasets

        roots = make_fake_datasets(DATA_DIR / "fake", seed=SEED)
        eyeq_csv = roots["eyeq_csv"]
    else:
        roots = {}
        for name in DATASETS:
            roots[name] = fetch_dataset(name)
            print(f"{name}: {roots[name]}  ({time.time() - t0:.0f}s)")
        eyeq_csv = DATA_DIR / "Label_EyeQ_train.csv"
        try:
            urllib.request.urlretrieve(EYEQ_URL, eyeq_csv)
        except Exception as exc:
            print(f"EyeQ quality labels could not be downloaded ({exc}). The quality gate will learn from synthetic degradations only.")

    index = pd.concat([D.INDEXERS[name](roots[name]) for name in D.INDEXERS], ignore_index=True)
    if MAX_EYEPACS:
        patients = index.loc[index.source == "eyepacs", "patient"].drop_duplicates()
        keep = set(patients.sample(min(len(patients), MAX_EYEPACS // 2), random_state=SEED))
        index = index[(index.source != "eyepacs") | index.patient.isin(keep)]
    index = D.add_eyeq_labels(index, [eyeq_csv])
    held_out = 0.25 if SMOKE_TEST else 0.10
    index = D.assign_splits(index, seed=SEED, val_frac=held_out, test_frac=held_out)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    save_raw_examples(index)
    print(f"Cropping {len(index)} images to {IMG_SIZE}px ...")
    index = D.build_cache(index, CACHE_DIR, IMG_SIZE, progress=tqdm)
    seg = D.cache_idrid_masks(D.index_idrid_segmentation(roots["idrid"]), CACHE_DIR, IMG_SIZE, progress=tqdm)
    seg.to_csv(SEG_CSV, index=False)
    index.drop(columns=["path"]).to_csv(INDEX_CSV, index=False)  # written last: its presence marks a complete cache
    print(f"Cache ready in {(time.time() - t0) / 60:.1f} min")


if not INDEX_CSV.exists():
    prepare_data()

index = pd.read_csv(INDEX_CSV, dtype={"image_id": str, "patient": str})
index["file"] = [str(CACHE_DIR / p) for p in index["cache_path"]]
seg = pd.read_csv(SEG_CSV)
has_grade = index.grade >= 0

print(f"\n{len(index)} images. Images per source and split:")
display(index.groupby(["source", "split"]).agg(images=("grade", "size"), referable=("referable", "sum")).unstack("split").fillna(0).astype(int))
print("DR grades per source:")
display(index[has_grade].groupby(["source", "grade"]).size().unstack("grade").fillna(0).astype(int).rename(columns=dict(enumerate(GRADE_NAMES))))
print("EyeQ quality labels (0 good, 1 usable, 2 reject):", index.eyeq.value_counts().sort_index().to_dict())
print("No retina found in", int((~index.fundus_found).sum()), "images (kept; the quality gate must reject them).")
metrics["data"] = {
    "images": {f"{s}/{p}": int(n) for (s, p), n in index.groupby(["source", "split"]).size().items()},
    "eyeq_labelled": int(index.eyeq.notna().sum()),
    "idrid_lesion_mask_images": int(len(seg)),
}

# %% [markdown]
# ### Preprocessing, step by step
#
# Crop to the retina → illumination correction → CLAHE. The second figure shows the simulated field problems
# (blur, motion, glare, haze, under-exposure) used for augmentation and for testing the quality gate.

# %%
examples = sorted((CACHE_DIR / "examples").glob("raw_*.jpg"))
picks = index.groupby("source").head(1).head(len(examples)) if examples else index.groupby("source").head(1)
fig, axes = plt.subplots(len(picks), 4, figsize=(11, 2.8 * len(picks)), squeeze=False)
for row, (_, r) in enumerate(picks.iterrows()):
    cropped = read_image(r.file)
    flat = correct_illumination(cropped)
    stages = [read_image(examples[row]) if examples else cropped, cropped, flat, enhance_fundus(cropped)]
    for ax, img, title in zip(axes[row], stages, ["original", "cropped", "+ illumination correction", "+ CLAHE (model input)"]):
        ax.imshow(img)
        ax.set_title(f"{r.source}: {title}" if title == "original" else title, fontsize=9)
        ax.axis("off")
show(fig, "preprocessing_steps")

sample = read_image(index[index.split == "test"].file.iloc[0])
levels = [0.15, 0.3, 0.5, 0.7, 1.0]
fig, axes = plt.subplots(len(KINDS), len(levels), figsize=(10, 2 * len(KINDS)), squeeze=False)
for i, kind in enumerate(KINDS):
    for j, severity in enumerate(levels):
        axes[i, j].imshow(degrade(sample, kind, severity, np.random.default_rng(SEED)))
        axes[i, j].set_title(f"{kind}  severity {severity}", fontsize=8)
        axes[i, j].axis("off")
show(fig, "field_degradations")

# %% [markdown]
# ## 5. Training tools
#
# One training loop serves every model. It fine-tunes the whole network with AdamW, a warm-up + cosine
# learning-rate schedule and mixed precision, keeps the weights with the best validation score, and saves a
# checkpoint after every epoch so an interrupted run resumes.
#
# Two safety nets for a long unattended run: if a model does not fit in GPU memory its batch size is halved and
# it starts again, and training stops early rather than run past `TIME_BUDGET_HOURS`.

# %%
NUM_WORKERS = 0 if SMOKE_TEST else min(4, os.cpu_count() or 2)


def make_loader(ds, batch_size, shuffle=False, sampler=None):
    training = shuffle or sampler is not None
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle and sampler is None, sampler=sampler, drop_last=training, num_workers=NUM_WORKERS, pin_memory=USE_AMP, worker_init_fn=worker_init)


@torch.no_grad()
def predict(model, ds, batch_size, desc="predicting"):
    """Run `model` over a dataset. Returns (logits [N, outputs], targets [N]) as numpy arrays."""
    model.eval()
    outputs, targets = [np.zeros((0, model.head.out_features), np.float32)], [np.zeros(0)]
    for x, y in tqdm(make_loader(ds, batch_size), desc=desc, leave=False):
        with torch.autocast(DEVICE.type, enabled=USE_AMP):
            out = model(x.to(DEVICE, non_blocking=True))
        outputs.append(out.float().cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(outputs), np.concatenate(targets)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def make_scaler():
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=USE_AMP)
    return torch.cuda.amp.GradScaler(enabled=USE_AMP)  # torch < 2.3


def lr_at(step, total_steps, peak):
    """Linear warm-up over the first 10% of training, then cosine decay to 1% of the peak."""
    warmup = max(1, total_steps // 10)
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return peak * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress)))


def fit(model, train_ds, val_ds, loss_fn, score_fn, epochs, lr, batch_size, tag, sampler=None, deadline=None):
    """Train `model`, leave it holding its best-validation weights and return the per-epoch history.

    `sampler` replaces plain shuffling; `deadline` (a time.time() value) stops training
    before an epoch that would not finish in time.
    """
    last_path, best_path = CKPT_DIR / f"{tag}_last.pt", CKPT_DIR / f"{tag}_best.pt"
    loader = make_loader(train_ds, batch_size, shuffle=True, sampler=sampler)
    total_steps = epochs * len(loader)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scaler = make_scaler()
    state = {"epoch": 0, "best": -math.inf, "history": []}

    if last_path.exists() and not FRESH_START:
        ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        if ckpt["total_steps"] == total_steps:
            model.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["opt"])
            scaler.load_state_dict(ckpt["scaler"])
            state = ckpt["state"]
            print(f"[{tag}] resuming after epoch {state['epoch']} of {epochs}")
        else:
            print(f"[{tag}] saved checkpoint belongs to different settings - starting again")
    if state["epoch"] == 0:
        best_path.unlink(missing_ok=True)

    epoch_seconds = None
    for epoch in range(state["epoch"], epochs):
        if deadline and epoch_seconds and time.time() + 1.15 * epoch_seconds > deadline:
            print(f"[{tag}] stopping after {epoch} of {epochs} epochs to stay inside TIME_BUDGET_HOURS")
            break
        model.train()
        t0, loss_sum, seen = time.time(), 0.0, 0
        bar = tqdm(loader, desc=f"[{tag}] epoch {epoch + 1}/{epochs}", leave=False)
        for step, (x, y) in enumerate(bar, start=epoch * len(loader)):
            for group in opt.param_groups:
                group["lr"] = lr_at(step, total_steps, lr)
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            with torch.autocast(DEVICE.type, enabled=USE_AMP):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            loss_sum, seen = loss_sum + loss.item() * len(x), seen + len(x)
            bar.set_postfix(loss=f"{loss_sum / seen:.4f}")
        if not math.isfinite(loss_sum):  # diverged: keep the best earlier epoch if there is one
            print(f"[{tag}] the training loss became NaN/inf in epoch {epoch + 1} - stopping this model here")
            if not best_path.exists():
                raise FloatingPointError(f"[{tag}] diverged in its first epoch; lower this model's learning rate")
            break
        score = float(score_fn(*predict(model, val_ds, batch_size * 2, desc="validating")))
        epoch_seconds = time.time() - t0
        state["history"].append({"epoch": epoch + 1, "train_loss": loss_sum / seen, "val_score": score, "minutes": epoch_seconds / 60})
        state["epoch"] = epoch + 1
        # NaN (a validation set with one class, only in the smoke test) never counts as an improvement.
        if score > state["best"] or not best_path.exists():
            state["best"] = score if math.isfinite(score) else state["best"]
            save_weights(model, best_path)
        ckpt = {"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "state": state, "total_steps": total_steps}
        torch.save(ckpt, str(last_path) + ".tmp")
        os.replace(str(last_path) + ".tmp", last_path)
        print(f"[{tag}] epoch {epoch + 1}/{epochs}  train loss {loss_sum / seen:.4f}  val score {score:.4f}  ({epoch_seconds / 60:.1f} min)")

    load_weights(model, best_path).to(DEVICE)
    return state["history"]


OUT_OF_MEMORY = getattr(torch, "OutOfMemoryError", None) or torch.cuda.OutOfMemoryError


def fit_with_memory_fallback(make_model, batch_size, **fit_args):
    """Build a model and train it; on a GPU out-of-memory error halve the batch size and start that model again.

    Returns (model, history, batch size actually used).
    """
    while True:
        model, out_of_memory = make_model().to(DEVICE), False
        try:
            return model, fit(model, batch_size=batch_size, **fit_args), batch_size
        except OUT_OF_MEMORY:
            out_of_memory = True
        if out_of_memory:  # outside the except block, so the failed batch is no longer referenced
            del model
            gc.collect()
            torch.cuda.empty_cache()
            if batch_size <= 4:
                raise RuntimeError(f"[{fit_args['tag']}] does not fit in GPU memory even with batch size {batch_size}")
            batch_size //= 2
            print(f"[{fit_args['tag']}] out of GPU memory - retrying with batch size {batch_size}")


def plot_histories(histories, title, score_name):
    """`histories` maps a model name to its per-epoch history."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.2))
    for name, history in histories.items():
        h = pd.DataFrame(history)
        a.plot(h.epoch, h.train_loss, marker="o", label=name)
        b.plot(h.epoch, h.val_score, marker="o", label=name)
    a.set(title=f"{title}: training loss", xlabel="epoch")
    b.set(title=f"{title}: validation {score_name}", xlabel="epoch")
    b.legend(fontsize=8)
    return fig

# %% [markdown]
# ## 6. Image-quality gate
#
# A small network scores each photograph for "ungradable". It learns from two kinds of examples:
#
# * **real** — EyeQ's human labels on EyePACS images (*reject* = ungradable, *good*/*usable* = gradable);
# * **synthetic** — good images that we deliberately ruin with heavy blur, glare, haze or under-exposure.
#   Mildly degraded copies stay labelled gradable, so the gate does not reject everything imperfect.
#
# The gate sees the cropped image *without* CLAHE: enhancement would hide the defects it must detect.

# %%
pool = index[index.split.isin(["train", "val", "test"])]
labelled = pool[pool.eyeq.notna()]
if len(labelled):
    extra = pool[(pool.source != "eyepacs")]  # other cameras, assumed gradable: they teach the gate not to reject a camera type
    q_frame = pd.concat([labelled.assign(q_label=(labelled.eyeq == 2).astype(float), q_clean=labelled.eyeq == 0), extra.assign(q_label=0.0, q_clean=True)])
else:
    print("No EyeQ labels: training the gate on synthetic degradations only.")
    q_frame = pool.assign(q_label=0.0, q_clean=True)
metrics["quality_gate"] = {"real_labels": int(len(labelled)), "real_ungradable": int((labelled.eyeq == 2).sum())}


def quality_ds(frame, train=False, synthetic=True, fixed_degradation=None):
    p = 0.25 if synthetic else 0.0
    return QualityDataset(frame.file, frame.q_label, frame.q_clean, IMG_SIZE, QUALITY_SIZE, train=train, severe_p=p, mild_p=p, seed=SEED, fixed_degradation=fixed_degradation,
                          gradable_range=GATE_GRADABLE_DEGRADATION, ungradable_range=GATE_UNGRADABLE_DEGRADATION)


def bce(out, y):
    return F.binary_cross_entropy_with_logits(out[:, 0].float(), y.float())


q_train, q_val, q_test = (q_frame[q_frame.split == s] for s in ("train", "val", "test"))
print(f"Quality gate: {len(q_train)} train / {len(q_val)} val / {len(q_test)} test images")
quality_model, history, _ = fit_with_memory_fallback(
    lambda: FundusNet(QUALITY_ARCH, num_grades=0, pretrained=not SMOKE_TEST), QUALITY_BATCH,
    train_ds=quality_ds(q_train, train=True), val_ds=quality_ds(q_val), loss_fn=bce, score_fn=lambda out, y: M.roc_auc(y, out[:, 0]),
    epochs=QUALITY_EPOCHS, lr=QUALITY_LR, tag=f"quality_{QUALITY_ARCH}_{QUALITY_SIZE}_{CACHE_TAG}",
)
show(plot_histories({QUALITY_ARCH: history}, "Quality gate", "AUC"), "quality_training")

# %%
# Threshold: the point on the validation ROC that best separates gradable from ungradable (Youden's J).
out, y = predict(quality_model, quality_ds(q_val), QUALITY_BATCH * 2)
QUALITY_THRESHOLD = M.youden_threshold(y, sigmoid(out[:, 0]))
if not 0.0 < QUALITY_THRESHOLD < 1.0:
    QUALITY_THRESHOLD = 0.5
print(f"Quality threshold: reject when P(ungradable) >= {QUALITY_THRESHOLD:.3f}\n")

rows = {}
for label, frame, synthetic in [("test: real + synthetic", q_test, True), ("test: real EyeQ labels only", q_test[q_test.eyeq.notna()], False)]:
    if len(frame):
        out, y = predict(quality_model, quality_ds(frame, synthetic=synthetic), QUALITY_BATCH * 2)
        rep = M.screening_report(y, sigmoid(out[:, 0]), QUALITY_THRESHOLD, n_boot=0 if SMOKE_TEST else 500)
        rows[label] = {"images": rep["n"], "ungradable": rep["n_positive"], "AUC": rep["auc"], "ungradable caught": rep["sensitivity"], "gradable accepted": rep["specificity"]}
display(pd.DataFrame(rows).T.round(3))
metrics["quality_gate"].update(threshold=QUALITY_THRESHOLD, test=rows)

# Score every image once. The cached rule checks (retina found, enough retina visible, red fundus) are
# part of the gate too, exactly as in the app.
out, _ = predict(quality_model, QualityDataset(index.file, np.zeros(len(index)), np.zeros(len(index), bool), IMG_SIZE, QUALITY_SIZE, severe_p=0, mild_p=0), QUALITY_BATCH * 2, desc="scoring all images")
index["q_prob"] = sigmoid(out[:, 0])
if SMOKE_TEST:  # an untrained gate accepts or rejects everything; keep both branches alive for the code test
    QUALITY_THRESHOLD = float(index.q_prob.quantile(0.7))
    metrics["quality_gate"]["threshold"] = QUALITY_THRESHOLD
index["rule_pass"] = index.fundus_found & (index.retina_fraction >= 0.5) & (index.red_minus_blue > 0)
index["gate_pass"] = index.rule_pass & (index.q_prob < QUALITY_THRESHOLD)

held_out = index[index.split.isin(["test", "external"])]
by_source = held_out.groupby("source").agg(images=("gate_pass", "size"), rejected=("gate_pass", lambda s: 1 - s.mean()))
by_grade = held_out[held_out.grade >= 0].groupby("grade").agg(images=("gate_pass", "size"), rejected=("gate_pass", lambda s: 1 - s.mean())).rename(index=dict(enumerate(GRADE_NAMES)))
print("Share of held-out images the gate rejects. IDRiD and Messidor-2 are curated, gradable sets, so their rejections are false alarms.")
display(by_source.round(3))
print("Rejections by DR grade. A gate that rejects diseased eyes more often is safe only because a rejected patient is re-imaged or referred.")
display(by_grade.round(3))
metrics["quality_gate"].update(reject_rate_by_source=by_source.rejected.to_dict(), reject_rate_by_grade=by_grade.rejected.to_dict())

# What do good images look like? The app uses these ranges to tell the operator *why* an image was rejected.
good = index[index.eyeq == 0] if (index.eyeq == 0).sum() >= 20 else index[index.gate_pass]
HINT_RANGES = {k: [float(good[k].quantile(0.02)), float(good[k].quantile(0.98))] for k in ("sharpness", "brightness", "contrast")}
HINT_RANGES["saturated_fraction"] = [0.0, float(max(good.saturated_fraction.quantile(0.99), 0.01))]
print("Statistics of good images (2nd-98th percentile):", {k: [round(v, 2) for v in r] for k, r in HINT_RANGES.items()})

# %% [markdown]
# ## 7. DR classifier: an ensemble of three networks
#
# Each member is an ImageNet-pretrained backbone with one small head: output 0 is the **referable DR** logit,
# outputs 1–5 are the five DR grades. The three members (`MODELS` in the Settings) have different designs and
# see the image at different sizes; at prediction time their probabilities are averaged. Their mistakes are
# partly different, so the average is usually better and steadier than any single member.
#
# **Balanced sampling.** In the first training run the model's scores sat on a different scale for each camera,
# and one threshold did not transfer. A likely cause: the sources differ in how much disease they contain
# (about 20% referable in EyePACS, 41% in APTOS), so "which camera took this?" becomes a cheap hint about the
# answer. With `BALANCE_SOURCES` every source is sampled so that half of its training images are referable:
# the source no longer says anything about the label. Each source keeps its natural share of the batches.
#
# Images that EyeQ's graders rejected are left out of training (their DR labels are unreliable); validation and
# test keep every image. Training images are augmented with flips, rotation, colour jitter and *mild* field
# degradations.

# %%
dr_train = index[(index.split == "train") & (index.eyeq != 2)]
dr_val = index[index.split == "val"]
if BALANCE_SOURCES:
    # Weight n_source / (classes * n_cell): each source keeps its share, and inside it both classes get half.
    in_source = dr_train.groupby("source")["grade"].transform("size")
    in_cell = dr_train.groupby(["source", "referable"])["grade"].transform("size")
    classes = dr_train.groupby("source")["referable"].transform("nunique")
    sample_weight = (in_source / (classes * in_cell)).to_numpy(dtype=np.float64)
    pos_weight = 1.0
else:
    sample_weight = np.ones(len(dr_train))
    n_pos = float(dr_train.referable.sum())
    pos_weight = (len(dr_train) - n_pos) / max(n_pos, 1.0)
# Grade weights follow the grade mix the sampler will actually produce.
seen_grades = np.bincount(dr_train.grade, weights=sample_weight, minlength=5)
grade_weights = np.clip(seen_grades.sum() / (5 * np.maximum(seen_grades, 1e-9)), 0.5, 5.0)
per_source = dr_train.assign(w=sample_weight, wr=sample_weight * dr_train.referable).groupby("source").agg(images=("w", "size"), data=("referable", "mean"), w=("w", "sum"), wr=("wr", "sum"))
mix = pd.DataFrame({"images": per_source.images, "referable in the data": per_source.data, "referable as sampled": per_source.wr / per_source.w, "share of each batch": per_source.w / per_source.w.sum()})
print(f"DR classifier: {len(dr_train)} train / {len(dr_val)} val images")
display(mix.round(3))
print("Loss weights  grades:", grade_weights.round(2), " referable:", round(pos_weight, 2))
GRADE_W = torch.tensor(grade_weights, dtype=torch.float32, device=DEVICE)
POS_W = torch.tensor(pos_weight, dtype=torch.float32, device=DEVICE)


def dr_loss(out, grade):
    referable = (grade >= REFERABLE_MIN_GRADE).float()
    binary = F.binary_cross_entropy_with_logits(out[:, 0].float(), referable, pos_weight=POS_W)
    grading = F.cross_entropy(out[:, 1:].float(), grade, weight=GRADE_W, label_smoothing=0.05)
    return binary + 0.5 * grading


def dr_ds(frame, size, train=False, fixed_degradation=None):
    return DRDataset(frame.file, frame.grade.clip(lower=0), size, train=train, seed=SEED, fixed_degradation=fixed_degradation)


def train_sampler():
    if not BALANCE_SOURCES:
        return None
    return torch.utils.data.WeightedRandomSampler(torch.as_tensor(sample_weight), num_samples=len(dr_train), replacement=True)


RESERVE_HOURS = 1.0  # kept back for evaluation, the Grad-CAM check, the robustness sweep and the export
members = []         # one dict per trained model: arch, size, batch, model, history
for i, spec in enumerate(MODELS):
    now = time.time()
    share = (RUN_START + (TIME_BUDGET_HOURS - RESERVE_HOURS) * 3600 - now) / (len(MODELS) - i)  # this model's slice of the remaining time
    if members and share < 20 * 60:
        print(f"SKIPPING {spec['arch']}: only {share / 60:.0f} min of the time budget left for it. The ensemble will have {len(members)} member(s).")
        continue
    tag = f"dr_{spec['arch']}_{spec['size']}_{CACHE_TAG}" + ("_balanced" if BALANCE_SOURCES else "")
    try:
        model, history, batch = fit_with_memory_fallback(
            lambda: FundusNet(spec["arch"], num_grades=5, pretrained=not SMOKE_TEST), spec["batch"],
            train_ds=dr_ds(dr_train, spec["size"], train=True), val_ds=dr_ds(dr_val, spec["size"]), loss_fn=dr_loss,
            score_fn=lambda out, g: M.roc_auc(g >= REFERABLE_MIN_GRADE, out[:, 0]), epochs=EPOCHS, lr=spec["lr"], tag=tag,
            sampler=train_sampler(), deadline=now + share,
        )
    except FloatingPointError as exc:  # one model failing must not cost the whole run
        print(f"SKIPPING {spec['arch']}: {exc}")
        continue
    members.append({"arch": spec["arch"], "size": spec["size"], "batch": batch, "model": model, "history": history})
    print(f"{spec['arch']} done: best validation AUC {max(h['val_score'] for h in history):.4f} in {sum(h['minutes'] for h in history) / 60:.2f} h ({len(history)} of {EPOCHS} epochs)\n")

if not members:
    raise RuntimeError("No DR model could be trained - see the messages above.")
if len(members) < len(MODELS):
    print(f"WARNING: the ensemble has {len(members)} of the {len(MODELS)} requested models.")
show(plot_histories({m["arch"]: m["history"] for m in members}, "DR classifier", "AUC (referable DR)"), "dr_training")
metrics["dr_training"] = {
    "epochs": EPOCHS, "train_images": int(len(dr_train)), "balanced_sources": BALANCE_SOURCES, "requested_models": [s["arch"] for s in MODELS],
    "members": [{k: m[k] for k in ("arch", "size", "batch", "history")} for m in members],
}

# %% [markdown]
# ## 8. Decision threshold and results
#
# The ensemble score is the average of the members' probabilities. The threshold is chosen **on validation data
# only**: the highest threshold that still reaches `VAL_SENSITIVITY`. It is then frozen and applied to data the
# models have never seen: the internal test split, and the two external datasets from other hospitals and cameras.
#
# The headline numbers are for images the quality gate **accepts**, because those are the only images the
# classifier grades in use. Results on *all* images are shown next to them.

# %%
def ensemble_scores(frame, fixed_degradation=None, desc="scoring"):
    """Average the members. Returns (P(referable) [N], grade probabilities [N, 5], each member's P(referable) [members, N])."""
    logits = [predict(m["model"], dr_ds(frame, m["size"], fixed_degradation=fixed_degradation), m["batch"] * 2, desc=f"{desc}: {m['arch']}")[0] for m in members]
    return ensemble_probabilities(logits)


scored = index[index.split.isin(["val", "test", "external"]) & has_grade]
p_ref, p_grade, p_members = ensemble_scores(scored, desc="validation and test images")
index.loc[scored.index, "p_ref"] = p_ref
index.loc[scored.index, "grade_pred"] = p_grade.argmax(axis=1)
MEMBER_COLUMNS = [f"p_{m['arch']}" for m in members]
for column, p in zip(MEMBER_COLUMNS, p_members):
    index.loc[scored.index, column] = p

val = index[(index.split == "val") & has_grade & index.gate_pass]
DR_THRESHOLD = M.threshold_for_sensitivity(val.referable, val.p_ref, VAL_SENSITIVITY)
val_sens, val_spec = M.sens_spec(val.referable, val.p_ref, DR_THRESHOLD)
print(f"Threshold: refer when P(referable DR) >= {DR_THRESHOLD:.3f}   (validation: sensitivity {val_sens:.3f}, specificity {val_spec:.3f})")

EVAL_SETS = {
    "Internal test (EyePACS + APTOS)": index[(index.split == "test") & has_grade],
    "  EyePACS test": index[(index.split == "test") & (index.source == "eyepacs")],
    "  APTOS test": index[(index.split == "test") & (index.source == "aptos")],
    "IDRiD - external, India": index[(index.source == "idrid") & has_grade],
    "Messidor-2 - external": index[(index.source == "messidor") & has_grade],
}
N_BOOT = 50 if SMOKE_TEST else 1000
results, table, member_auc = {}, {}, {}
for name, frame in EVAL_SETS.items():
    accepted = frame[frame.gate_pass]
    rep = M.screening_report(accepted.referable, accepted.p_ref, DR_THRESHOLD, n_boot=N_BOOT, seed=SEED)
    rep_all = M.screening_report(frame.referable, frame.p_ref, DR_THRESHOLD, n_boot=0)
    rep["rejected_by_gate"] = 1 - len(accepted) / max(len(frame), 1)
    rep["grade_kappa"] = M.quadratic_weighted_kappa(accepted.grade, accepted.grade_pred, 5)
    rep["all_images"] = rep_all
    results[name.strip()] = rep
    member_auc[name.strip()] = {**{m["arch"]: M.roc_auc(accepted.referable, accepted[c]) for m, c in zip(members, MEMBER_COLUMNS)}, "ensemble": rep["auc"]}
    ci = lambda key: "[{:.3f}, {:.3f}]".format(*rep[key + "_ci"])
    table[name] = {
        "graded": rep["n"], "referable": rep["n_positive"], "rejected": f"{rep['rejected_by_gate']:.1%}",
        "AUC": round(rep["auc"], 3), "sensitivity": round(rep["sensitivity"], 3), "sens. 95% CI": ci("sensitivity"),
        "specificity": round(rep["specificity"], 3), "spec. 95% CI": ci("specificity"), "grade kappa": round(rep["grade_kappa"], 3),
        "sens. (all images)": round(rep_all["sensitivity"], 3), "spec. (all images)": round(rep_all["specificity"], 3),
    }
display(pd.DataFrame(table).T)
print("AUC of each member on its own, and of the ensemble (gate-accepted images):")
display(pd.DataFrame(member_auc).T.round(4))
metrics["dr"] = {
    "threshold": DR_THRESHOLD, "validation": {"sensitivity": val_sens, "specificity": val_spec, "target_sensitivity": VAL_SENSITIVITY},
    "results": results, "member_auc": member_auc,
}

print(f"\nScreening targets: sensitivity > {TARGET_SENSITIVITY:.0%} and specificity > {TARGET_SPECIFICITY:.0%} on held-out data")
criteria = {}
for name in ("Internal test (EyePACS + APTOS)", "IDRiD - external, India", "Messidor-2 - external"):
    r = results[name]
    criteria[name] = bool(r["sensitivity"] > TARGET_SENSITIVITY and r["specificity"] > TARGET_SPECIFICITY)
    print(f"  {'MET    ' if criteria[name] else 'NOT MET'}  {name}: sensitivity {r['sensitivity']:.1%}, specificity {r['specificity']:.1%}")
metrics["success_criteria"] = {"target_sensitivity": TARGET_SENSITIVITY, "target_specificity": TARGET_SPECIFICITY, "met": criteria}
if SMOKE_TEST:
    print("  (smoke test on synthetic images: these numbers mean nothing)")

# Does one threshold transfer between cameras? For each set, the threshold that would give 90% sensitivity
# *on that set*. Chosen with the test labels, so this is a diagnosis, never a result: the closer these
# numbers are to each other, the better a single threshold works everywhere.
transfer = {}
for name, frame in EVAL_SETS.items():
    accepted = frame[frame.gate_pass]
    own = M.threshold_for_sensitivity(accepted.referable, accepted.p_ref, TARGET_SENSITIVITY)
    transfer[name] = {"threshold for 90% sensitivity on this set": own, "specificity there": M.sens_spec(accepted.referable, accepted.p_ref, own)[1]}
print(f"\nThreshold transfer check (the frozen threshold is {DR_THRESHOLD:.3f}):")
display(pd.DataFrame(transfer).T.round(3))
metrics["dr"]["threshold_transfer_diagnostic"] = {k.strip(): v for k, v in transfer.items()}

# %%
# ROC curves with the chosen operating point, and the 5-grade confusion matrix on the internal test set.
fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
for name in ("Internal test (EyePACS + APTOS)", "IDRiD - external, India", "Messidor-2 - external"):
    frame = EVAL_SETS[name][EVAL_SETS[name].gate_pass]
    if frame.referable.nunique() < 2:
        continue
    fpr, tpr, _ = M.roc_curve(frame.referable, frame.p_ref)
    line, = axes[0].plot(fpr, tpr, label=f"{name}  (AUC {results[name]['auc']:.3f})")
    axes[0].plot(1 - results[name]["specificity"], results[name]["sensitivity"], "o", color=line.get_color(), markersize=8)
axes[0].plot([0, 1], [0, 1], ":", color="grey")
axes[0].axhline(TARGET_SENSITIVITY, ls="--", lw=0.8, color="grey")
axes[0].axvline(1 - TARGET_SPECIFICITY, ls="--", lw=0.8, color="grey")
axes[0].set(xlabel="1 - specificity", ylabel="sensitivity", title=f"Referable DR, ensemble: ROC (dots = threshold {DR_THRESHOLD:.2f}; dashed = targets)")
axes[0].legend(loc="lower right", fontsize=8)

frame = EVAL_SETS["Internal test (EyePACS + APTOS)"]
frame = frame[frame.gate_pass]
cm = M.confusion_matrix(frame.grade, frame.grade_pred, 5)
axes[1].imshow(cm / np.maximum(cm.sum(axis=1, keepdims=True), 1), cmap="Blues", vmin=0, vmax=1)
for i in range(5):
    for j in range(5):
        axes[1].text(j, i, cm[i, j], ha="center", va="center", fontsize=9, color="white" if cm[i, j] > 0.5 * max(cm[i].sum(), 1) else "black")
axes[1].set(xticks=range(5), yticks=range(5), xlabel="predicted grade", ylabel="true grade", title="5-grade confusion matrix (internal test)")
axes[1].set_xticklabels(GRADE_NAMES, rotation=30, ha="right", fontsize=8)
axes[1].set_yticklabels(GRADE_NAMES, fontsize=8)
show(fig, "roc_and_confusion")
# Validation rows are included so a different operating point can be studied later without retraining.
index[index.split.isin(["val", "test", "external"])][["source", "image_id", "split", "grade", "referable", "q_prob", "gate_pass", "p_ref", "grade_pred", *MEMBER_COLUMNS]].to_csv(REPORT_DIR / "predictions.csv", index=False)

# %% [markdown]
# ## 9. Grad-CAM: where does the ensemble look?
#
# Every member explains its own prediction; the three heatmaps are averaged. A heatmap that *looks* plausible is
# not evidence. IDRiD provides pixel-level lesion masks drawn by ophthalmologists (microaneurysms, haemorrhages,
# hard and soft exudates) for 81 images the models never trained on, so agreement can be measured:
#
# * **pointing hit** — the hottest point of the heatmap lies on (or within 4% of the image width of) a lesion;
# * **energy on lesions ÷ lesion area** — how many times more heat falls on lesions than a uniform map would put there (1 = chance);
# * **pixel AUC** — the heatmap used as a lesion detector (0.5 = chance).
#
# As a sanity check the same scores are computed for a **baseline copy** of the ensemble that has never seen a
# DR label (ImageNet backbones, random heads). If the trained ensemble does not clearly beat it, the heatmaps are
# showing generic image saliency, not learned evidence.

# %%
def cams_for(models_and_sizes, files, batch=8):
    """Ensemble Grad-CAM (referable-DR output) for cached images. Returns (enhanced images, heatmaps at IMG_SIZE, P(referable))."""
    images, cams, probs = [], [], []
    for i in range(0, len(files), batch):
        cropped = [read_image(f) for f in files[i : i + batch]]
        cam, logits = ensemble_grad_cam(models_and_sizes, cropped, IMG_SIZE, DEVICE)
        images += [model_input(c, IMG_SIZE) for c in cropped]
        cams += list(cam)
        probs += ensemble_probabilities(logits)[0].tolist()
    return images, cams, probs


trained_members = [(m["model"], m["size"]) for m in members]
seg_files = [str(CACHE_DIR / p) for p in seg.cache_path]
seg_masks = [cv2.imread(str(CACHE_DIR / p), cv2.IMREAD_UNCHANGED) for p in seg.mask_path]
seg_images, seg_cams, seg_probs = cams_for(trained_members, seg_files)
torch.manual_seed(SEED)
baseline_members = [(FundusNet(m["arch"], num_grades=5, pretrained=not SMOKE_TEST).to(DEVICE).eval(), m["size"]) for m in members]
_, base_cams, _ = cams_for(baseline_members, seg_files)
del baseline_members

rows = []
for image_id, image, mask, cam, base, prob in zip(seg.image_id, seg_images, seg_masks, seg_cams, base_cams, seg_probs):
    lesion, valid = (mask & D.LESION_ANY) > 0, retina_mask(image) > 0
    trained, baseline = lesion_overlap(cam, lesion, valid), lesion_overlap(base, lesion, valid)
    rows.append({"image_id": image_id, "p_referable": prob, "flagged": prob >= DR_THRESHOLD, **trained, **{f"baseline_{k}": v for k, v in baseline.items() if k != "lesion_area"}})
cam_report = pd.DataFrame(rows)
cam_report.to_csv(REPORT_DIR / "gradcam_vs_idrid_lesions.csv", index=False)

for prefix in ("", "baseline_"):
    cam_report[prefix + "lift"] = cam_report[prefix + "energy_in_mask"] / cam_report["lesion_area"]
summary = pd.DataFrame({
    "trained model": [cam_report.pointing_hit.mean(), cam_report.lift.median(), cam_report.pixel_auc.mean()],
    "baseline (no DR training)": [cam_report.baseline_pointing_hit.mean(), cam_report.baseline_lift.median(), cam_report.baseline_pixel_auc.mean()],
    "chance": [float("nan"), 1.0, 0.5],
}, index=["pointing hit rate", "energy on lesions / lesion area (median)", "pixel AUC (mean)"])
display(summary.round(3))
paired = cam_report.dropna(subset=["pixel_auc", "baseline_pixel_auc"])
wins = int((paired.pixel_auc > paired.baseline_pixel_auc).sum())
auc_ci = M.bootstrap_ci(lambda _, v: float(np.mean(v)), np.zeros(len(paired)), paired.pixel_auc.to_numpy(), n_boot=N_BOOT, seed=SEED) if len(paired) else (float("nan"),) * 2
cam_ok = bool(len(paired) and auc_ci[0] > 0.5 and wins > len(paired) / 2)
print(f"{len(seg)} IDRiD images with lesion masks; the ensemble flags {int(cam_report.flagged.sum())} as referable; {int(cam_report['empty'].sum())} have an empty heatmap (counted as misses).")
print(f"Mean pixel AUC {paired.pixel_auc.mean():.3f}, 95% CI [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]; better than the baseline on {wins} of {len(paired)} images.")
print("Sanity check", "PASSED: heatmaps overlap lesions more than chance and more than networks with no DR training." if cam_ok else "FAILED: heatmaps do not reliably overlap the annotated lesions - do not present them as explanations.")
print("Note: Grad-CAM on these backbones has a coarse grid (about 1/32 of the image), so it marks regions, not individual microaneurysms.")
metrics["gradcam"] = {
    "images": int(len(seg)), "flagged_referable": int(cam_report.flagged.sum()), "empty_heatmaps": int(cam_report["empty"].sum()),
    "trained": summary["trained model"].to_dict(), "baseline": summary["baseline (no DR training)"].to_dict(),
    "pixel_auc_ci": list(auc_ci), "beats_baseline_on": wins, "compared": int(len(paired)), "sanity_check_passed": cam_ok,
}

# %%
order = np.argsort(-cam_report.pixel_auc.fillna(-1).to_numpy())
picks = list(order[:2]) + list(order[len(order) // 2 : len(order) // 2 + 1]) + list(order[-1:])  # best, typical, worst
fig, axes = plt.subplots(len(picks), 3, figsize=(9, 3 * len(picks)), squeeze=False)
for row, i in enumerate(picks):
    outline = cv2.dilate(((seg_masks[i] & D.LESION_ANY) > 0).astype(np.uint8), np.ones((3, 3), np.uint8))
    marked = seg_images[i].copy()
    marked[outline > 0] = (0, 255, 0)
    r = cam_report.iloc[i]
    for ax, img, title in zip(axes[row], [seg_images[i], marked, overlay_heatmap(seg_images[i], seg_cams[i])], [f"{r.image_id}  P(referable) = {r.p_referable:.2f}", "ophthalmologists' lesion masks", f"Grad-CAM (pixel AUC {r.pixel_auc:.2f})"]):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
show(fig, "gradcam_vs_lesion_masks")

# Heatmaps for ordinary test images: the most confident referable cases and the most confident normals.
test = index[(index.split == "test") & has_grade & index.gate_pass].sort_values("p_ref")
picks = pd.concat([test.tail(4), test.head(2)])
images, cams, _ = cams_for(trained_members, list(picks.file))
fig, axes = plt.subplots(2, len(picks), figsize=(2.6 * len(picks), 5.6), squeeze=False)
for col, (r, img, cam) in enumerate(zip(picks.itertuples(), images, cams)):
    axes[0, col].imshow(img)
    axes[0, col].set_title(f"true: {GRADE_NAMES[r.grade]}", fontsize=8)
    axes[1, col].imshow(overlay_heatmap(img, cam))
    axes[1, col].set_title(f"P(referable) = {r.p_ref:.2f}", fontsize=8)
    axes[0, col].axis("off")
    axes[1, col].axis("off")
show(fig, "gradcam_examples")

# %% [markdown]
# ## 10. Field conditions: does the gate protect the classifier?
#
# Good test images are degraded on purpose, at increasing severity. For each level we record how many images
# the gate rejects, and the ensemble's sensitivity and specificity with and without the gate. The gate is doing
# its job when it rejects images *before* the classifier starts to fail on them.

# %%
clean = index[(index.split == "test") & has_grade & index.gate_pass]
clean = clean.sample(min(len(clean), 16 if SMOKE_TEST else 600), random_state=SEED)
clean = clean.assign(q_label=0.0, q_clean=True)
severities = [0.3, 0.7] if SMOKE_TEST else [0.15, 0.3, 0.5, 0.7, 1.0]
base_sens, base_spec = M.sens_spec(clean.referable, clean.p_ref, DR_THRESHOLD)
rows = [{"kind": k, "severity": 0.0, "rejected": 0.0, "sens_no_gate": base_sens, "spec_no_gate": base_spec, "sens_with_gate": base_sens, "spec_with_gate": base_spec} for k in KINDS]
for kind in KINDS:
    for severity in severities:
        q_out, _ = predict(quality_model, quality_ds(clean, synthetic=False, fixed_degradation=(kind, severity)), QUALITY_BATCH * 2, desc=f"{kind} {severity}")
        p = ensemble_scores(clean, fixed_degradation=(kind, severity), desc=f"{kind} {severity}")[0]
        accepted = sigmoid(q_out[:, 0]) < QUALITY_THRESHOLD
        y = clean.referable.to_numpy()
        sens, spec = M.sens_spec(y, p, DR_THRESHOLD)
        gated = M.sens_spec(y[accepted], p[accepted], DR_THRESHOLD) if accepted.any() else (float("nan"), float("nan"))
        rows.append({"kind": kind, "severity": severity, "rejected": 1 - accepted.mean(), "sens_no_gate": sens, "spec_no_gate": spec, "sens_with_gate": gated[0], "spec_with_gate": gated[1]})
robust = pd.DataFrame(rows).sort_values(["kind", "severity"])
robust.to_csv(REPORT_DIR / "robustness.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
for kind, part in robust.groupby("kind"):
    axes[0].plot(part.severity, part.rejected, marker="o", label=kind)
    axes[1].plot(part.severity, part.sens_no_gate, marker="o", label=kind)
    axes[2].plot(part.severity, part.spec_no_gate, marker="o", label=kind)
axes[0].set(title="Rejected by the quality gate", xlabel="severity", ylim=(-0.03, 1.03))
axes[1].set(title="Sensitivity without the gate", xlabel="severity", ylim=(-0.03, 1.03))
axes[2].set(title="Specificity without the gate", xlabel="severity", ylim=(-0.03, 1.03))
axes[1].axhline(TARGET_SENSITIVITY, ls="--", lw=0.8, color="grey")
axes[2].axhline(TARGET_SPECIFICITY, ls="--", lw=0.8, color="grey")
axes[0].legend(fontsize=8)
show(fig, "robustness_field_conditions")
severe = robust[robust.severity >= 0.7]
print(f"On {len(clean)} good test images: the gate rejects {severe.rejected.mean():.1%} of severely degraded copies (severity >= 0.7) and {robust[(robust.severity > 0) & (robust.severity <= 0.3)].rejected.mean():.1%} of mildly degraded ones (<= 0.3).")
metrics["robustness"] = {"images": int(len(clean)), "rows": robust.to_dict("records"), "severe_rejected": float(severe.rejected.mean())}

# %% [markdown]
# ## 11. Export
#
# Writes everything the app needs into `fundalux_outputs/`, reloads it from disk through the real inference
# pipeline as a self-test and zips it into `/kaggle/working/fundalux_outputs.zip`.

# %%
def jsonable(obj):
    """Plain JSON types only; NaN/inf become null so any JSON parser (MATLAB, JavaScript) can read the file."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(obj) else None
    return obj


for i, m in enumerate(members):
    m["weights"] = f"fundalux_dr_{i + 1}_{m['arch']}.pt"
    save_weights(m["model"], OUT_DIR / m["weights"])
save_weights(quality_model, OUT_DIR / "fundalux_quality.pt")
config = {
    "format": 2,
    "image_size": IMG_SIZE,
    "min_input_side": 32 if SMOKE_TEST else 224,
    "grade_names": GRADE_NAMES,
    "referable_min_grade": REFERABLE_MIN_GRADE,
    "dr": {
        "models": [{"arch": m["arch"], "weights": m["weights"], "input_size": m["size"]} for m in members],
        "combine": "mean of the members' probabilities", "num_grades": 5, "threshold": DR_THRESHOLD,
    },
    "quality": {"arch": QUALITY_ARCH, "weights": "fundalux_quality.pt", "input_size": QUALITY_SIZE, "threshold": QUALITY_THRESHOLD, "hint_ranges": HINT_RANGES},
    "trained_with": {"torch": torch.__version__, "run": RUN, "date": time.strftime("%Y-%m-%d"), "hours": (time.time() - RUN_START) / 3600, "smoke_test": SMOKE_TEST},
}
internal = metrics["dr"]["results"]["Internal test (EyePACS + APTOS)"]
metrics["simulation_inputs"] = {"sensitivity": internal["sensitivity"], "specificity": internal["specificity"], "p_ungradable": internal["rejected_by_gate"]}

# Per-site thresholds: one global threshold does not transfer between cameras, so each evaluation dataset gets
# its own, tuned for the same sensitivity. Datasets with a validation split are tuned on it and measured on their
# test split; the external ones are tuned on all their images and measured by cross-fitting. The app offers these
# as sites and lets a new site calibrate its own.
sites = SiteRegistry(OUT_DIR / "sites.json", model_fingerprint(config), builtin_profiles(index[index.split.isin(["val", "test", "external"])], VAL_SENSITIVITY))
sites.save()
site_table = pd.DataFrame([{"site": p.name, "threshold": p.threshold, "images": p.n_images, "expected sensitivity": p.expected_sensitivity, "expected specificity": p.expected_specificity, "estimated by": p.estimate} for p in sites.profiles])
print(f"Per-site thresholds (global threshold {DR_THRESHOLD:.3f}):")
display(site_table.round(3) if len(site_table) else "none - too few images per site")
metrics["sites"] = {p.name: {"threshold": p.threshold, "expected_sensitivity": p.expected_sensitivity, "expected_specificity": p.expected_specificity, "estimate": p.estimate} for p in sites.profiles}
(OUT_DIR / "config.json").write_text(json.dumps(jsonable(config), indent=2))
(OUT_DIR / "metrics.json").write_text(json.dumps(jsonable(metrics), indent=2))

# Self-test: load the files just written, exactly as the app will, and compare with the models in memory.
bundle = Fundalux.load(OUT_DIR)
photo = read_image(index[index.split == "test"].file.iloc[0])
result = bundle.predict(photo, skip_gate=True)
cropped = crop_fundus(photo, IMG_SIZE)[0]
with torch.no_grad():
    logits = [m["model"].eval()(to_tensor(model_input(cropped, m["size"])).unsqueeze(0).to(DEVICE)) for m in members]
expected = float(ensemble_probabilities(logits)[0][0])
print("Self-test:", json.dumps(jsonable(result.summary()))[:500])
assert abs(result.referable_score - expected) < 1e-3, f"exported models disagree with the trained ones: {result.referable_score} vs {expected}"

# Serving copies for ONNX Runtime, so the web app needs no PyTorch: each network with its feature map and logits,
# plus the head parameters for the closed-form heatmaps. Checked against PyTorch before zipping.
if EXPORT_ONNX:
    try:
        if IN_KAGGLE or IN_COLAB:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxscript"], check=False)
        export_bundle(OUT_DIR, members=[(m["arch"], m["model"].cpu().eval(), m["size"]) for m in members], quality=quality_model.cpu().eval(), quality_size=QUALITY_SIZE)
        served = Fundalux.load(OUT_DIR, backend="onnx").predict(photo, skip_gate=True)
        gap = abs(served.referable_score - result.referable_score), float(np.abs(served.heatmap - result.heatmap).max())
        assert gap[0] < 1e-3 and gap[1] < 1e-2, f"ONNX copies disagree with PyTorch: score {gap[0]:.2e}, heatmap {gap[1]:.2e}"
        print(f"ONNX export checked against PyTorch: score difference {gap[0]:.1e}, heatmap difference {gap[1]:.1e}")
    except Exception as exc:
        print(f"ONNX export failed ({type(exc).__name__}: {exc}). The .pt files are complete; run python -m fundalux.export later.")
        shutil.rmtree(OUT_DIR / "onnx", ignore_errors=True)
        config.pop("onnx", None)
        (OUT_DIR / "config.json").write_text(json.dumps(jsonable(config), indent=2))
    finally:
        for m in members:
            m["model"].to(DEVICE)
        quality_model.to(DEVICE)

archive = shutil.make_archive(str(WORK / "fundalux_outputs"), "zip", OUT_DIR)
print(f"\n{archive}  ({os.path.getsize(archive) / 1e6:.0f} MB)   total run time {(time.time() - RUN_START) / 3600:.2f} h")
for f in sorted(OUT_DIR.rglob("*")):
    if f.is_file():
        print(f"  {str(f.relative_to(OUT_DIR)):45s} {f.stat().st_size / 1e6:8.2f} MB")
if IN_KAGGLE:
    from IPython.display import FileLink

    print("\nDownload fundalux_outputs.zip from the Output section of this notebook (right-hand sidebar while the session is open, or the Output tab of the saved version).")
    display(FileLink(os.path.basename(archive)))
elif IN_COLAB:
    from google.colab import files

    files.download(archive)

# %% [markdown]
# ## Done
#
# 1. Download `fundalux_outputs.zip` from this notebook's Output and unzip it into the project's `models/` folder
#    (so that `models/config.json` exists), replacing what is there. The `checkpoints/` and `fundalux/` folders
#    in the Output are not needed.
# 2. Start the app: `streamlit run app/app.py`.
# 3. Read `metrics.json` and the figures before trusting the models: the success criteria printed in section 8
#    are either met or not, per dataset, and the "threshold transfer check" below them shows whether one
#    threshold now works across cameras.
