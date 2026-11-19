"""Discrete-event model of a rural DR screening camp (resource allocation).

Patients arrive, register, are photographed (with retakes when the quality gate
rejects the image) and are then graded either by the AI or, in the benchmark
rejects the image) and are then graded either by the AI or, in the benchmark
scenario, by a remote ophthalmologist who reads every patient. Only AI-referred
patients reach the ophthalmologist in the AI scenario. The ophthalmologist
reads in one session per day, after the camp closes.

This is the Python counterpart of the Simulink/SimEvents model in matlab/.
Times are in minutes. Staffing, timing, prevalence and cost defaults are
placeholders to be replaced with local figures - they are not measurements.

    python -m fundalux.screening_sim --metrics models/metrics.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace

import numpy as np
import simpy

DAY = 24 * 60.0


@dataclass
class Scenario:
    name: str = "AI-assisted"
    use_ai: bool = True
    days: int = 20
    patients_per_day: float = 80.0
    camp_hours: float = 6.0  # arrivals are spread over this window
    n_registration: int = 1
    n_cameras: int = 2  # each camera station has its own technician
    n_ophthalmologists: int = 1
    ophthalmologist_hours: float = 2.0  # remote reading session per ophthalmologist, starting when the camp closes
    register_min: tuple = (1.0, 2.0, 4.0)  # triangular (low, mode, high)
    imaging_min: tuple = (3.0, 5.0, 9.0)  # one attempt, both eyes
    review_min: tuple = (2.0, 4.0, 8.0)  # ophthalmologist, per patient
    prevalence: float = 0.10  # share of attendees with referable DR
    p_ungradable: float = 0.15  # chance that one imaging attempt is ungradable
    max_retakes: int = 2  # the AI quality gate asks for retakes on the spot
    sensitivity: float = 0.90  # of the grader: the AI, or the ophthalmologist in the benchmark
    specificity: float = 0.85
    cost_camera_station_day: float = 1500.0
    cost_registration_day: float = 600.0
    cost_ophthalmologist_hour: float = 1500.0


def simulate(sc: Scenario, seed: int = 0) -> dict:
    """Run one scenario and return per-day averages and totals."""
    rng = np.random.default_rng(seed)
    env = simpy.Environment()
    registration = simpy.Resource(env, sc.n_registration)
    cameras = simpy.Resource(env, sc.n_cameras)
    review_queue = simpy.Store(env)
    n = dict(patients=0, diseased=0, detected=0, missed=0, false_referrals=0, ungradable_referrals=0, recalls=0, reviews=0, attempts=0)
    camp_times, turnaround, overtime = [], [], [0.0] * sc.days
    busy = dict(camera=0.0, review=0.0)

    def tri(spec):
        return float(rng.triangular(*spec))

    def patient(day):
        arrived = env.now
        diseased = rng.random() < sc.prevalence
        n["patients"] += 1
        n["diseased"] += diseased
        with registration.request() as req:
            yield req
            yield env.timeout(tri(sc.register_min))
        with cameras.request() as req:
            yield req
            attempt = 0
            while True:
                attempt += 1
                t = tri(sc.imaging_min)
                busy["camera"] += t
                yield env.timeout(t)
                gradable = rng.random() >= sc.p_ungradable
                # Without the AI nobody knows on site that the photo is unusable.
                if gradable or not sc.use_ai or attempt > sc.max_retakes:
                    break
            n["attempts"] += attempt
        camp_times.append(env.now - arrived)
        overtime[day] = max(overtime[day], env.now - (day * DAY + sc.camp_hours * 60))
        case = dict(diseased=diseased, gradable=gradable, queued=env.now)
        if not sc.use_ai:
            yield review_queue.put(case)  # benchmark: the ophthalmologist reads everyone
        elif not gradable:
            n["ungradable_referrals"] += 1
            yield review_queue.put(case)
        elif rng.random() < (sc.sensitivity if diseased else 1 - sc.specificity):
            n["detected" if diseased else "false_referrals"] += 1
            yield review_queue.put(case)
        elif diseased:
            n["missed"] += 1

    def arrivals(day):
        yield env.timeout(day * DAY - env.now)
        close = day * DAY + sc.camp_hours * 60
        while True:
            yield env.timeout(float(rng.exponential(sc.camp_hours * 60 / sc.patients_per_day)))
            if env.now >= close:
                return
            env.process(patient(day))

    def ophthalmologist():
        # One reading session per camp day, plus one more the day after the last
        # camp so that only a genuine capacity shortfall is left as backlog.
        for day in range(sc.days + 1):
            start = day * DAY + sc.camp_hours * 60
            yield env.timeout(max(0.0, start - env.now))
            end = start + sc.ophthalmologist_hours * 60
            while env.now < end:
                get = review_queue.get()
                yield get | env.timeout(end - env.now)
                if not get.triggered:
                    get.cancel()
                    break
                case = get.value
                t = tri(sc.review_min)
                busy["review"] += t
                yield env.timeout(t)
                n["reviews"] += 1
                turnaround.append(env.now - case["queued"])
                if not sc.use_ai:  # the benchmark reader makes the grading decision
                    if not case["gradable"]:
                        n["recalls"] += 1
                    elif rng.random() < (sc.sensitivity if case["diseased"] else 1 - sc.specificity):
                        n["detected" if case["diseased"] else "false_referrals"] += 1
                    elif case["diseased"]:
                        n["missed"] += 1

    for day in range(sc.days):
        env.process(arrivals(day))
    for _ in range(sc.n_ophthalmologists):
        env.process(ophthalmologist())
    env.run(until=(sc.days + 1) * DAY)

    d = float(sc.days)
    cost_day = sc.n_cameras * sc.cost_camera_station_day + sc.n_registration * sc.cost_registration_day + sc.n_ophthalmologists * sc.ophthalmologist_hours * sc.cost_ophthalmologist_hour
    camp = np.array(camp_times) if camp_times else np.array([np.nan])
    return {
        "scenario": sc.name,
        "patients_per_day": n["patients"] / d,
        "time_in_camp_mean_min": float(camp.mean()),
        "time_in_camp_p95_min": float(np.percentile(camp, 95)),
        "camp_overtime_mean_min": float(np.mean(overtime)),
        "camera_utilisation": busy["camera"] / (sc.n_cameras * sc.camp_hours * 60 * d),
        "imaging_attempts_per_patient": n["attempts"] / max(n["patients"], 1),
        "reviews_requested_per_day": (n["reviews"] + len(review_queue.items)) / d,
        "ophthalmologist_min_per_day": busy["review"] / d,
        "review_backlog_end": len(review_queue.items),
        "referable_cases_in_backlog": int(sum(c["diseased"] for c in review_queue.items)),
        "review_turnaround_mean_h": float(np.mean(turnaround) / 60) if turnaround else float("nan"),
        "referable_cases": int(n["diseased"]),
        "detected": int(n["detected"]),
        "missed": int(n["missed"]),
        "false_referrals": int(n["false_referrals"]),
        "ungradable_referrals": int(n["ungradable_referrals"]),
        "recalls_for_bad_images": int(n["recalls"]),
        "cost_per_day": cost_day,
        "cost_per_patient": cost_day * d / max(n["patients"], 1),
        "cost_per_detected_case": cost_day * d / n["detected"] if n["detected"] else float("nan"),
    }


def manual_benchmark(sc: Scenario, sensitivity: float = 0.90, specificity: float = 0.95) -> Scenario:
    """Same camp without AI: no on-site quality check, the ophthalmologist reads every patient.

    The reader accuracy defaults are assumptions, not measurements.
    """
    return replace(sc, name="Manual grading (benchmark)", use_ai=False, sensitivity=sensitivity, specificity=specificity)


def sweep_allocation(base: Scenario, cameras=(1, 2, 3), ophthalmologist_hours=(1.0, 2.0, 4.0, 6.0), seed: int = 0) -> list[dict]:
    """Simulate every combination of camera stations and daily ophthalmologist hours."""
    rows = []
    for c in cameras:
        for h in ophthalmologist_hours:
            row = simulate(replace(base, n_cameras=c, ophthalmologist_hours=h), seed)
            rows.append({"n_cameras": c, "ophthalmologist_hours": h, **row})
    return rows


def recommend(rows: list[dict], max_p95_wait_min: float = 60.0, max_backlog: int = 0) -> dict | None:
    """Cheapest allocation that keeps patient waits and the review backlog within limits."""
    ok = [r for r in rows if r["time_in_camp_p95_min"] <= max_p95_wait_min and r["review_backlog_end"] <= max_backlog]
    return min(ok, key=lambda r: r["cost_per_day"]) if ok else None


def scenario_from_metrics(metrics: dict, **overrides) -> Scenario:
    """Build the AI scenario from the numbers measured by the training notebook."""
    sim = metrics.get("simulation_inputs", {})
    known = {k: sim[k] for k in ("sensitivity", "specificity", "p_ungradable") if sim.get(k) is not None}
    return Scenario(**{**known, **overrides})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--metrics", help="metrics.json written by the training notebook (for sensitivity/specificity)")
    parser.add_argument("--patients", type=float, default=80.0, help="patients per camp day")
    parser.add_argument("--days", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    metrics = json.load(open(args.metrics)) if args.metrics else {}
    ai = scenario_from_metrics(metrics, patients_per_day=args.patients, days=args.days)
    print("Scenario:", json.dumps(asdict(ai), indent=1))
    for sc in (ai, manual_benchmark(ai)):
        print(f"\n== {sc.name} ==")
        for k, v in simulate(sc, args.seed).items():
            print(f"  {k:32s} {v:.2f}" if isinstance(v, float) else f"  {k:32s} {v}")
    rows = sweep_allocation(ai, seed=args.seed)
    print("\n== Allocation sweep (AI-assisted) ==")
    print(f"  {'cameras':>7} {'oph h':>6} {'p95 wait':>9} {'backlog':>8} {'cost/day':>9}")
    for r in rows:
        print(f"  {r['n_cameras']:>7} {r['ophthalmologist_hours']:>6.1f} {r['time_in_camp_p95_min']:>9.1f} {r['review_backlog_end']:>8} {r['cost_per_day']:>9.0f}")
    best = recommend(rows)
    print("\nCheapest allocation meeting the limits:", f"{best['n_cameras']} camera(s), {best['ophthalmologist_hours']} ophthalmologist h/day" if best else "none - add resources")


if __name__ == "__main__":
    main()
