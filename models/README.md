# models/

The trained bundle. Only the small files are in git; the weights are not (they come to about 480 MB).

Tracked:

```
config.json      members, input sizes, thresholds, training run
metrics.json     evaluation results
sites.json       per-site thresholds
reports/         predictions, robustness, heatmap check
figures/         training curves, ROC, examples
```

Not tracked, so add them after cloning:

```
netraxai_dr_1_efficientnet_b3.pt       one file per member of the DR ensemble
netraxai_dr_2_efficientnet_v2_s.pt
netraxai_dr_3_convnext_tiny.pt
netraxai_quality.pt
onnx/                                  serving copies, needed by the web app and the container
```

**Download:** `fundalux_weights_run2.zip` (445 MB) from https://drive.google.com/file/d/1mTRrxddgEA2I2h_kgvIn5MRn5Mpsi65i/view?usp=drive_link

Unzip it in the project folder (not inside `models/`; the zip already contains the `models/` path), then
check the files:

```bash
unzip fundalux_weights_run2.zip
cd models && shasum -a 256 -c WEIGHTS.sha256      # every line should end in OK
```

The web app and the container need only `onnx/`; the `.pt` files are for the Streamlit app, the PyTorch
backend and re-exporting. To train your own instead, run `Fundalux_Kaggle_Training.ipynb` on Kaggle,
unzip its output here, and run `python -m fundalux.export models` if the output has no `onnx/`.
