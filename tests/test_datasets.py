import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from fundalux import datasets as D
from fundalux.data import DRDataset, QualityDataset
from fundalux.synthetic import make_fake_datasets


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    return make_fake_datasets(root, n_eyepacs_patients=30, n_aptos=20, n_idrid=12, n_idrid_seg=4, n_messidor=16)


@pytest.fixture(scope="module")
def index(fake):
    frames = [fn(fake[name]) for name, fn in D.INDEXERS.items()]
    return D.assign_splits(D.add_eyeq_labels(pd.concat(frames, ignore_index=True), [fake["eyeq_csv"]]), seed=1, val_frac=0.2, test_frac=0.2)


def test_indexers_find_every_image(fake):
    eyepacs = D.index_eyepacs(fake["eyepacs"])
    assert len(eyepacs) == 60 and not eyepacs["path"].str.contains("cropped").any()
    assert eyepacs.loc[eyepacs.image_id == "10_left", "patient"].item() == "eyepacs_10"
    assert len(D.index_aptos(fake["aptos"])) == 20  # three CSVs, trailing blank lines
    idrid = D.index_idrid(fake["idrid"])
    assert len(idrid) == 12 and idrid["image_id"].is_unique
    # Training and testing sets both contain IDRiD_001; each id must point into its own folder.
    assert "a. Training Set" in idrid.loc[idrid.image_id == "train_IDRiD_001", "path"].item()
    assert "b. Testing Set" in idrid.loc[idrid.image_id == "test_IDRiD_001", "path"].item()
    assert not idrid["path"].str.contains("Segmentation").any()
    messidor = D.index_messidor(fake["messidor"])
    assert len(messidor) == 16 and (messidor["grade"] == -1).sum() == 2
    for df in (eyepacs, idrid, messidor):
        assert list(df.columns) == D.COLUMNS and df["grade"].between(-1, 4).all()


def test_labels_match_the_csv(fake):
    labels = pd.read_csv(f"{fake['eyepacs']}/trainLabels.csv").set_index("image")["level"]
    eyepacs = D.index_eyepacs(fake["eyepacs"]).set_index("image_id")
    assert (eyepacs["grade"] == labels.loc[eyepacs.index]).all()


def test_segmentation_index(fake):
    items = D.index_idrid_segmentation(fake["idrid"])
    assert [i["image_id"] for i in items] == ["IDRiD_01", "IDRiD_02", "IDRiD_03", "IDRiD_04"]
    assert all(set(i["masks"]) <= {"HE", "EX"} and i["masks"] for i in items)
    assert all(f"{i['image_id']}_" in p for i in items for p in i["masks"].values())


def test_splits_keep_patients_together_and_external_sets_out(index):
    assert set(index.loc[index.source.isin(["idrid", "messidor"]), "split"]) == {"external"}
    pool = index[index.source.isin(D.TRAIN_SOURCES)]
    assert set(pool["split"]) == {"train", "val", "test"}
    assert (pool.groupby("patient")["split"].nunique() == 1).all()
    assert (index["referable"] == (index["grade"] >= 2)).all()
    again = D.assign_splits(index.drop(columns=["split", "referable"]), seed=1, val_frac=0.2, test_frac=0.2)
    assert (again["split"] == index["split"]).all()
    assert index.loc[index.source == "eyepacs", "eyeq"].notna().any() and index.loc[index.source != "eyepacs", "eyeq"].isna().all()


@pytest.fixture(scope="module")
def cached(index, fake, tmp_path_factory):
    cache = tmp_path_factory.mktemp("cache")
    df = D.build_cache(index, cache, size=64, workers=2)
    seg = D.cache_idrid_masks(D.index_idrid_segmentation(fake["idrid"]), cache, 64)
    return cache, df, seg


def test_cache(cached, index):
    cache, df, seg = cached
    assert len(df) == len(index) and df["fundus_found"].all()
    assert (df["image_id"].to_numpy() == index["image_id"].to_numpy()).all()  # row order survives the worker pool
    img = cv2.imread(str(cache / df["cache_path"].iloc[0]))
    assert img.shape == (64, 64, 3)
    assert {"sharpness", "brightness", "retina_fraction"} <= set(df.columns)
    mask = cv2.imread(str(cache / seg["mask_path"].iloc[0]), cv2.IMREAD_UNCHANGED)
    assert mask.shape == (64, 64) and (mask & D.LESION_ANY).any() and not (mask & ~np.uint8(D.LESION_BITS["HE"] | D.LESION_BITS["EX"])).any()


def test_cache_skips_corrupt_files(index, tmp_path):
    bad = tmp_path / "broken.jpg"
    bad.write_bytes(b"not an image")
    df = index.head(3).copy()
    df.loc[df.index[1], "path"] = str(bad)
    out = D.build_cache(df, tmp_path / "c", size=32, workers=1)
    assert len(out) == 2 and "error" not in out.columns


def test_torch_datasets(cached):
    cache, df, _ = cached
    paths = [str(cache / p) for p in df["cache_path"]]
    grades = df["grade"].clip(lower=0).to_numpy()
    ds = DRDataset(paths, grades, size=64, train=False)
    x, y = ds[0]
    assert x.shape == (3, 64, 64) and x.dtype == torch.uint8 and y == grades[0]
    assert torch.equal(ds[0][0], ds[0][0])  # evaluation is deterministic
    # A model with a smaller input than the cache gets exactly what the app's pipeline will give it.
    from fundalux.models import to_tensor
    from fundalux.preprocess import model_input, read_image

    assert torch.equal(DRDataset(paths, grades, size=48)[1][0], to_tensor(model_input(read_image(paths[1]), 48)))
    train = DRDataset(paths, grades, size=64, train=True, degrade_p=1.0)
    assert not torch.equal(train[0][0], train[0][0])
    degraded = DRDataset(paths, grades, size=64, fixed_degradation=("dark", 0.9))
    assert degraded[0][0].float().mean() < x.float().mean()

    q = QualityDataset(paths, np.zeros(len(paths)), np.ones(len(paths), bool), size=64, input_size=32, severe_p=0.5, mild_p=0.0, seed=3)
    labels = np.array([q[i][1] for i in range(len(q))])
    assert q[0][0].shape == (3, 32, 32) and 0.2 < labels.mean() < 0.8
    assert np.array_equal(labels, [q[i][1] for i in range(len(q))])
    # Images not known to be clean are never turned into synthetic "ungradable" examples.
    q2 = QualityDataset(paths, np.zeros(len(paths)), np.zeros(len(paths), bool), size=64, input_size=32, severe_p=0.5)
    assert sum(q2[i][1] for i in range(len(q2))) == 0
    # The gate and the classifier must see the *same* degraded image in robustness sweeps.
    a = QualityDataset(paths, np.zeros(len(paths)), np.ones(len(paths), bool), size=64, input_size=64, fixed_degradation=("glare", 0.5), seed=7)[2][0]
    b = dg_reference(paths[2], seed=7, index=2)
    assert torch.equal(a, b)


def dg_reference(path, seed, index):
    from fundalux.degrade import degrade
    from fundalux.models import to_tensor
    from fundalux.preprocess import read_image

    return to_tensor(degrade(read_image(path), "glare", 0.5, np.random.default_rng((seed, index))))


def test_find_kaggle_input(tmp_path):
    (tmp_path / "aptos2019" / "train_images").mkdir(parents=True)
    (tmp_path / "datasets" / "someone" / "Messidor2Preprocess").mkdir(parents=True)
    (tmp_path / "other" / "aptos2019-extra").mkdir(parents=True)
    assert D.find_kaggle_input("mariaherrerot/aptos2019", tmp_path) == str(tmp_path / "aptos2019")
    assert D.find_kaggle_input("mariaherrerot/messidor2preprocess", tmp_path) == str(tmp_path / "datasets" / "someone" / "Messidor2Preprocess")
    assert D.find_kaggle_input("tanlikesmath/diabetic-retinopathy-resized", tmp_path) is None
    assert D.find_kaggle_input("a/b", tmp_path / "missing") is None
