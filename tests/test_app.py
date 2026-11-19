"""Runs the Streamlit app headless on a tiny fake bundle (no trained models needed)."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from fundalux.sites import SiteRegistry, calibrate
from test_core import _write_bundle
from test_sites import _scores

APP = str(Path(__file__).resolve().parents[1] / "app" / "app.py")


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    _write_bundle(tmp_path, quality_bias=-10.0)
    import json

    config = json.loads((tmp_path / "config.json").read_text())
    registry = SiteRegistry.load(tmp_path, config)
    registry.put(calibrate("Camp North", *_scores(300), 0.93, source="built-in"))
    monkeypatch.setenv("FUNDALUX_MODELS", str(tmp_path))
    return tmp_path


def _run(**query):
    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in query.items():
        at.query_params[k] = v
    return at.run()


def _html(at):
    return "".join(h.proto.body for h in at.get("html"))


def test_screen_tab_screens_a_sample_and_switches_site(bundle):
    at = _run()
    assert not at.exception
    assert at.selectbox(key="site").options == ["Global threshold (no site)", "Camp North"]
    sample = next(s for s in at.selectbox if s.label == "Or open a sample")
    sample.select(sample.options[1]).run()
    assert not at.exception and "nx-verdict" in _html(at) and "Referral at global threshold" in _html(at)
    at.selectbox(key="site").select("Camp North").run()
    assert not at.exception and "Referral at Camp North threshold" in _html(at)


def test_other_tabs_render(bundle):
    for tab, marker in (("Sites", "Per-site thresholds"), ("Model report", "No metrics.json"), ("Camp simulation", "Screening-camp simulation")):
        at = _run(tab=tab)
        assert not at.exception, (tab, [e.message for e in at.exception])
        assert marker in _html(at), tab


def test_missing_models_explains_what_to_do(tmp_path, monkeypatch):
    monkeypatch.setenv("FUNDALUX_MODELS", str(tmp_path / "nothing"))
    at = _run()
    assert not at.exception and any("models/README.md" in w.value for w in at.warning)


def test_save_and_remove_a_calibrated_site(bundle):
    at = _run(tab="Sites")
    profile = calibrate("Camp South", *_scores(300, seed=7), 0.93, source="calibrated in the app")
    at.session_state["pending_site"] = (profile, {"labelled": 300, "matched": 300, "unmatched_labels": 0, "rejected_by_gate": 0})
    at.run()
    next(b for b in at.button if b.label == "Save site").click().run()
    assert not at.exception, [e.message for e in at.exception]
    assert at.session_state["site"] == "Camp South"  # the new site is selected at once
    assert "Camp South" in at.selectbox(key="site").options
    at.button(key="remove-button").click().run()
    assert not at.exception, [e.message for e in at.exception]
    assert at.session_state["site"] == "Global threshold (no site)"
    assert "Camp South" not in at.selectbox(key="site").options
