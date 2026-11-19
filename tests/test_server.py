"""The web server on a tiny exported bundle: pages, screening, sites, security."""

import importlib
import io
import json
import time

import cv2
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fundalux.export import export_bundle
from fundalux.sites import SiteRegistry, calibrate
from fundalux.synthetic import synthetic_fundus
from test_core import _write_bundle
from test_sites import _predictions, _scores


@pytest.fixture
def server(tmp_path, monkeypatch):
    _write_bundle(tmp_path, quality_bias=-10.0)
    export_bundle(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    (tmp_path / "metrics.json").write_text(json.dumps({
        "dr": {"threshold": 0.5, "validation": {"target_sensitivity": 0.93, "sensitivity": 0.93, "specificity": 0.8},
               "results": {"IDRiD - external, India": {"n": 600, "n_positive": 180, "auc": 0.9, "sensitivity": 0.95, "specificity": 0.7, "rejected_by_gate": 0.01}}},
        "success_criteria": {"target_sensitivity": 0.9, "target_specificity": 0.85},
    }))
    (tmp_path / "reports").mkdir()
    _predictions().to_csv(tmp_path / "reports" / "predictions.csv", index=False)
    SiteRegistry(tmp_path / "sites.json", "", []).save()
    registry = SiteRegistry.load(tmp_path, config)
    registry.put(calibrate("IDRiD (Nanded, India)", *_scores(300), 0.93, source="built-in"))
    monkeypatch.setenv("FUNDALUX_MODELS", str(tmp_path))
    monkeypatch.setenv("FUNDALUX_ADMIN_TOKEN", "secret")
    import server.main as main

    main = importlib.reload(main)  # pick up the environment
    with TestClient(main.app) as client:
        yield client, tmp_path


def _jpeg(img):
    return cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tobytes()


def test_pages_render(server):
    client, _ = server
    for path in ("/", "/how-it-works", "/evidence", "/screen", "/sites", "/camp"):
        res = client.get(path)
        assert res.status_code == 200, path
        assert "Fundalux" in res.text and "<script>" not in res.text  # no inline scripts: the CSP forbids them
    assert "Content-Security-Policy" in client.get("/").headers
    assert client.get("/api/health").json()["backend"] == "onnx"


def test_screen_upload_sample_and_sites(server):
    client, _ = server
    img, _ = synthetic_fundus(np.random.default_rng(0), width=480, height=320, grade=3)
    res = client.post("/api/screen", files={"photo": ("eye.jpg", _jpeg(img), "image/jpeg")})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["gradable"] and 0 <= body["referable_score"] <= 1
    assert body["images"]["heatmap"].startswith("data:image/jpeg;base64,")
    labels = [d["label"] for d in body["decisions"]]
    assert labels[0] == "Global threshold" and "IDRiD (Nanded, India)" in labels
    for d in body["decisions"]:
        assert d["referable"] == (body["referable_score"] >= d["threshold"])
    sample = client.post("/api/screen", data={"sample_name": "idrid_test_IDRiD_029_grade0.jpg", "degrade_kind": "blur", "degrade_severity": "1"})
    assert sample.status_code == 200 and sample.json()["gradable"] is False or sample.json()["reasons"] is not None


def test_screen_rejects_bad_input(server):
    client, _ = server
    assert client.post("/api/screen").status_code == 422
    assert client.post("/api/screen", files={"photo": ("x.jpg", b"not an image", "image/jpeg")}).status_code == 422
    assert client.post("/api/screen", data={"sample_name": "../models/config.json"}).status_code == 422
    assert client.post("/api/screen", data={"sample_name": "idrid_test_IDRiD_029_grade0.jpg", "degrade_kind": "rm -rf"}).status_code == 422
    assert client.get("/figures/../config.json").status_code == 404
    assert client.get("/samples/../../server/main.py").status_code == 404


def test_admin_endpoints_need_the_token(server):
    client, _ = server
    files = [("labels", ("l.csv", b"image,grade\na.jpg,0\n", "text/csv")), ("photos", ("a.jpg", b"x", "image/jpeg"))]
    assert client.post("/api/sites/calibrate", data={"name": "X camp"}, files=files).status_code == 401
    assert client.post("/api/sites/calibrate", data={"name": "X camp"}, files=files, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.delete("/api/sites/IDRiD (Nanded, India)", headers={"Authorization": "Bearer secret"}).status_code == 409  # built-in


def test_calibrate_save_and_remove_a_site(server):
    client, _ = server
    rng = np.random.default_rng(1)
    rows, files = [], []
    for i in range(48):
        grade = 3 if i % 2 else 0
        img, _ = synthetic_fundus(rng, width=240, height=160, grade=grade)
        rows.append(f"p{i}.jpg,{grade}")
        files.append(("photos", (f"p{i}.jpg", _jpeg(img), "image/jpeg")))
    labels = ("labels", ("labels.csv", ("image,grade\n" + "\n".join(rows)).encode(), "text/csv"))
    auth = {"Authorization": "Bearer secret"}
    res = client.post("/api/sites/calibrate", data={"name": "Camp West", "target_sensitivity": "0.9"}, files=[labels, *files], headers=auth)
    assert res.status_code == 200, res.text
    job = res.json()["job"]
    for _ in range(300):
        state = client.get(f"/api/jobs/{job}", headers=auth).json()
        if state["status"] in ("done", "failed"):
            break
        time.sleep(0.2)
    assert state["status"] == "done", state
    assert state["profile"]["n_images"] == 48
    assert client.post(f"/api/jobs/{job}/save", headers=auth).status_code == 200
    assert "Camp West" in [s["name"] for s in client.get("/api/sites").json()]
    assert "Camp West" in "".join(d["label"] for d in client.post("/api/screen", data={"sample_name": "idrid_test_IDRiD_029_grade0.jpg"}).json()["decisions"])
    assert client.delete("/api/sites/Camp West", headers=auth).status_code == 200
    assert "Camp West" not in [s["name"] for s in client.get("/api/sites").json()]


def test_camp_and_report(server):
    client, _ = server
    defaults = client.get("/api/camp/defaults").json()
    res = client.post("/api/camp", json={**defaults, "days": 5})
    assert res.status_code == 200 and {"ai", "manual", "sweep", "best"} <= res.json().keys()
    assert client.post("/api/camp", json={**defaults, "patients_per_day": -3}).status_code == 422
    report = client.get("/api/report").json()
    idrid = report["sets"]["idrid"]
    assert idrid["site"]["name"] == "IDRiD (Nanded, India)" and len(idrid["roc"]) > 10
    assert [g["grade"] for g in idrid["by_grade"]] == [0, 1, 2, 3, 4]
