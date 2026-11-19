"""Clinician review: set building, blinding, storage, analysis and the web endpoints."""

import importlib
import json

import cv2
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fundalux.export import export_bundle
from fundalux.pipeline import Fundalux
from fundalux.review import ResponseStore, ReviewSet, analyse, build_review_set, export_frame, stratified_sample
from fundalux.synthetic import synthetic_fundus
from test_core import _write_bundle


def test_stratified_sample_spreads_grades():
    items = [{"id": i, "reference_grade": g} for i, g in enumerate([0] * 50 + [1] * 3 + [2] * 20 + [4] * 10)]
    chosen = stratified_sample(items, 12, seed=1)
    grades = [c["reference_grade"] for c in chosen]
    assert len(chosen) == 12 and grades.count(1) == 3 and grades.count(0) == 3
    assert stratified_sample(items, 1000) and len(stratified_sample(items, 1000)) == len(items)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("bundle")
    _write_bundle(root, quality_bias=-10.0)
    bundle = Fundalux.load(root)
    rng = np.random.default_rng(0)
    photos, refs = [], {}
    for i in range(8):
        grade = [0, 0, 1, 2, 3, 4, 2, 0][i]
        img, _ = synthetic_fundus(rng, width=240, height=160, grade=grade)
        name = f"patient_{i}_grade{grade}.jpg"
        photos.append((name, cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tobytes()))
        refs[f"patient_{i}_grade{grade}"] = grade
    out = tmp_path_factory.mktemp("review")
    manifest = build_review_set(bundle, photos, out, "Test set", refs, site=None)
    return root, out, manifest


def test_set_hides_names_and_stores_model_output(built):
    _, out, manifest = built
    ids = [it["id"] for it in manifest["items"]]
    assert ids == [f"r{i:03d}" for i in range(1, 9)]
    for it in manifest["items"]:
        assert "grade" not in it["file"] and (out / it["file"]).exists() and (out / it["heatmap"]).exists()
        assert it["model"]["referable_score"] is not None and it["reference_grade"] in range(5)
    rs = ReviewSet.load(out)
    a, b = rs.order_for("Dr A"), rs.order_for("Dr B")
    assert sorted(a) == sorted(ids) and a == rs.order_for(" dr a ") and a != b


def test_store_and_analysis(built, tmp_path):
    _, out, manifest = built
    rs = ReviewSet.load(out)
    store = ResponseStore(tmp_path / "r.sqlite3")
    with pytest.raises(ValueError):
        store.save("Dr A", "r001", True, None, False, None, "", 3.0, "blinded")
    for it in manifest["items"]:
        ref = it["reference_grade"]
        store.save("Dr A", it["id"], True, ref, False, "high", "", 10.0, "blinded")  # perfect grader
        store.save("Dr B", it["id"], it["id"] != "r008", 0 if it["id"] != "r008" else None, it["id"] == "r002", "low", "looks clean" if it["id"] == "r003" else "", 5.0, "teaching")
    store.save("Dr A", "r001", True, manifest["items"][0]["reference_grade"], False, "medium", "changed my mind", 4.0, "blinded")  # overwrite
    df = store.frame()
    assert len(df) == 16 and df[(df.reviewer == "Dr A") & (df.item_id == "r001")].note.item() == "changed my mind"
    assert store.answered("Dr A") == {it["id"] for it in manifest["items"]}

    res = analyse(rs, df, threshold=0.5, threshold_label="global threshold")
    by_name = {r["name"]: r for r in res["reviewers"]}
    assert by_name["Dr A"]["answered"] == 8 and by_name["Dr B"]["ungradable"] == 1 and by_name["Dr B"]["other_referral"] == 1 and by_name["Dr B"]["teaching_mode"] == 8
    a_vs_ref = next(v for v in res["versus_reference"] if v["reviewer"] == "Dr A")
    assert a_vs_ref["doctor"]["sensitivity"] == 1.0 and a_vs_ref["doctor"]["specificity"] == 1.0
    b_vs_ref = next(v for v in res["versus_reference"] if v["reviewer"] == "Dr B")
    assert b_vs_ref["doctor"]["sensitivity"] == 0.0
    doctors = next(p for p in res["pairs"] if {p["a"], p["b"]} == {"Dr A", "Dr B"})
    assert doctors["n"] == 7 and sum(map(sum, doctors["table"])) == 7
    assert sum(res["gradability"].values()) == 16
    assert all(d["kind"] for d in res["disagreements"]) and any(n["note"] == "looks clean" for n in res["notes"])
    exported = export_frame(rs, df, 0.5)
    assert {"original_file", "reference_grade", "model_score", "grade"} <= set(exported.columns) and len(exported) == 16
    assert analyse(rs, df.iloc[0:0], 0.5, "x")["reviewers"] == []


@pytest.fixture
def server(built, tmp_path, monkeypatch):
    root, out, _ = built
    export_bundle(root)
    review_dir = tmp_path / "review"
    import shutil

    shutil.copytree(out, review_dir)
    monkeypatch.setenv("FUNDALUX_MODELS", str(root))
    monkeypatch.setenv("FUNDALUX_REVIEW_DIR", str(review_dir))
    monkeypatch.setenv("FUNDALUX_ADMIN_TOKEN", "admin")
    monkeypatch.setenv("FUNDALUX_REVIEW_TOKEN", "doctors")
    import server.main as main

    main = importlib.reload(main)
    with TestClient(main.app) as client:
        yield client, review_dir


def test_review_flow_over_http(server):
    client, review_dir = server
    doc, admin = {"Authorization": "Bearer doctors"}, {"Authorization": "Bearer admin"}
    assert client.get("/review").status_code == 200 and client.get("/review/results").status_code == 200
    assert client.get("/api/review/next?reviewer=Dr A").status_code == 401
    first = client.get("/api/review/next?reviewer=Dr A", headers=doc).json()
    assert first["answered"] == 0 and first["total"] == 8
    skipped = client.get(f"/api/review/next?reviewer=Dr A&skip={first['item_id']}", headers=doc).json()
    assert skipped["item_id"] != first["item_id"]
    img = client.get(f"/api/review/image/{first['item_id']}", headers=doc)
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    # The model's answer stays hidden until the reviewer has answered.
    assert client.get(f"/api/review/model/{first['item_id']}?reviewer=Dr A", headers=doc).status_code == 409
    bad = {"reviewer": "Dr A", "item_id": first["item_id"], "gradable": True, "grade": 7}
    assert client.post("/api/review/answer", json=bad, headers=doc).status_code == 422
    ok = {"reviewer": "Dr A", "item_id": first["item_id"], "gradable": True, "grade": 2, "confidence": "high", "seconds": 12.5, "mode": "teaching"}
    assert client.post("/api/review/answer", json=ok, headers=doc).status_code == 200
    model = client.get(f"/api/review/model/{first['item_id']}?reviewer=Dr A", headers=doc).json()
    assert 0 <= model["referable_score"] <= 1 and model["heatmap"].startswith("data:image/jpeg")
    assert client.get("/api/review/next?reviewer=Dr A", headers=doc).json()["answered"] == 1
    # Results and export are for the organiser only.
    assert client.get("/api/review/results", headers=doc).status_code == 401
    res = client.get("/api/review/results", headers=admin).json()
    assert res["reviewers"][0]["name"] == "Dr A" and res["items"] == 8 and res["stale_model"] is False
    csv = client.get("/api/review/export.csv", headers=admin)
    assert csv.status_code == 200 and "original_file" in csv.text.splitlines()[0]
    assert client.get(f"/api/review/thumb/{first['item_id']}", headers=admin).status_code == 200
