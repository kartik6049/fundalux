from dataclasses import replace

import pytest

from fundalux import screening_sim as sim


def test_counts_are_consistent():
    r = sim.simulate(sim.Scenario(days=10), seed=1)
    assert r["patients_per_day"] == pytest.approx(80, rel=0.15)
    assert r["detected"] + r["missed"] <= r["referable_cases"]  # the rest were referred as ungradable
    assert 1.0 <= r["imaging_attempts_per_patient"] <= 3.0
    assert 0 < r["camera_utilisation"]
    assert sim.simulate(sim.Scenario(days=10), seed=1) == r  # reproducible


def test_perfect_and_useless_ai():
    base = sim.Scenario(days=10, p_ungradable=0.0, ophthalmologist_hours=8)
    perfect = sim.simulate(replace(base, sensitivity=1.0, specificity=1.0), seed=2)
    assert perfect["missed"] == 0 and perfect["false_referrals"] == 0 and perfect["detected"] == perfect["referable_cases"]
    blind = sim.simulate(replace(base, sensitivity=0.0, specificity=1.0), seed=2)
    assert blind["detected"] == 0 and blind["missed"] == blind["referable_cases"] and blind["reviews_requested_per_day"] == 0


def test_retakes_only_happen_with_the_gate():
    base = sim.Scenario(days=10, p_ungradable=0.4)
    ai, manual = sim.simulate(base, seed=3), sim.simulate(sim.manual_benchmark(base), seed=3)
    assert ai["imaging_attempts_per_patient"] > 1.3 and manual["imaging_attempts_per_patient"] == 1.0
    assert ai["recalls_for_bad_images"] == 0 and manual["ungradable_referrals"] == 0
    # After up to 3 attempts about 0.4^3 = 6.4% of patients are still ungradable.
    assert ai["ungradable_referrals"] / (ai["patients_per_day"] * 10) == pytest.approx(0.064, abs=0.03)


def test_ai_triage_cuts_ophthalmologist_workload():
    base = sim.Scenario(days=20)
    ai, manual = sim.simulate(base, seed=4), sim.simulate(sim.manual_benchmark(base), seed=4)
    assert manual["reviews_requested_per_day"] == pytest.approx(manual["patients_per_day"])
    assert ai["reviews_requested_per_day"] < 0.5 * manual["reviews_requested_per_day"]
    # 2 h/day cannot cover 80 patients at about 4.7 min each: the manual backlog must grow.
    assert manual["review_backlog_end"] > 500 and ai["review_backlog_end"] == 0


def test_more_cameras_shorten_the_queue():
    base = sim.Scenario(days=5, patients_per_day=200, n_cameras=1)
    one, three = sim.simulate(base, seed=5), sim.simulate(replace(base, n_cameras=3), seed=5)
    assert three["time_in_camp_p95_min"] < 0.5 * one["time_in_camp_p95_min"]


def test_sweep_and_recommendation():
    rows = sim.sweep_allocation(sim.Scenario(days=5), cameras=(1, 3), ophthalmologist_hours=(1.0, 6.0), seed=6)
    assert len(rows) == 4
    best = sim.recommend(rows)
    assert best is not None and best["time_in_camp_p95_min"] <= 60 and best["review_backlog_end"] == 0
    assert best["cost_per_day"] == min(r["cost_per_day"] for r in rows if r["time_in_camp_p95_min"] <= 60 and r["review_backlog_end"] == 0)
    assert sim.recommend(rows, max_p95_wait_min=0.0) is None


def test_scenario_from_metrics():
    sc = sim.scenario_from_metrics({"simulation_inputs": {"sensitivity": 0.95, "specificity": None, "p_ungradable": 0.2}}, days=3)
    assert (sc.sensitivity, sc.specificity, sc.p_ungradable, sc.days) == (0.95, 0.85, 0.2, 3)
    assert sim.scenario_from_metrics({}).sensitivity == 0.90
