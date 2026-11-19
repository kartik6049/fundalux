"""Fundalux screening app.

    streamlit run app/app.py

Needs the trained models in models/ (see models/README.md). Set the
FUNDALUX_MODELS environment variable to load them from another folder.
Run it from the project folder so .streamlit/config.toml (theme, fonts) applies.
"""

from __future__ import annotations

import io
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ui  # noqa: E402
from fundalux import GRADE_NAMES  # noqa: E402
from fundalux import screening_sim as sim  # noqa: E402
from fundalux.calibrate import calibrate_site, read_labels, score_images  # noqa: E402
from fundalux.degrade import KINDS, degrade  # noqa: E402
from fundalux.pipeline import DISCLAIMER, Fundalux  # noqa: E402
from fundalux.preprocess import decode_image, read_image  # noqa: E402
from fundalux.sites import CalibrationError, SiteRegistry  # noqa: E402

MODEL_DIR = Path(os.environ.get("FUNDALUX_MODELS", ROOT / "models"))
SAMPLE_DIR = ROOT / "samples"
IMAGE_TYPES = ["jpg", "jpeg", "png", "tif", "tiff", "bmp"]
GLOBAL = "Global threshold (no site)"
SAMPLE_LABELS = {
    "idrid_test_IDRiD_029_grade0.jpg": "IDRiD 029 · graded No DR",
    "idrid_test_IDRiD_008_grade2.jpg": "IDRiD 008 · graded Moderate NPDR",
    "idrid_test_IDRiD_001_grade4.jpg": "IDRiD 001 · graded Proliferative DR",
}

st.set_page_config(page_title="Fundalux", page_icon=":material/visibility:", layout="wide")
st.html(ui.GLOBAL_CSS)


@st.cache_resource(show_spinner="Loading the models ...")
def load_bundle(model_dir: str) -> Fundalux:
    return Fundalux.load(model_dir)


@st.cache_data
def load_metrics(model_dir: str) -> dict:
    path = Path(model_dir) / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else {}


def load_sites(bundle: Fundalux) -> SiteRegistry:
    return SiteRegistry.load(MODEL_DIR, bundle.config)


@st.cache_data(show_spinner=False, max_entries=24)
def analyse(_bundle: Fundalux, model_dir: str, source: str, payload, kind: str, severity: float, skip_gate: bool):
    """Screen one photograph at the bundle's global threshold. Cached: changing the site re-decides without re-scoring."""
    image = decode_image(payload) if source == "upload" else read_image(payload)
    if kind != "none":
        image = degrade(image, kind, severity, np.random.default_rng(0))
    return _bundle.predict(image, skip_gate=skip_gate)


# --------------------------------------------------------------------------- masthead


def masthead(bundle: Fundalux | None, metrics: dict, registry: SiteRegistry | None):
    """Wordmark and model facts on the left, the site picker on the right. Returns the chosen site profile or None."""
    left, right = st.columns([3, 1.15], vertical_alignment="bottom")
    meta = []
    if bundle is not None:
        trained = bundle.config.get("trained_with", {})
        meta.append(("Models", f"{len(bundle.dr_members)}-network ensemble" if len(bundle.dr_members) > 1 else bundle.dr_members[0][0]))
        if trained.get("date"):
            meta.append(("Trained", trained["date"]))
        meta.append(("Global threshold", f"{bundle.dr_threshold:.3f}"))
    with left:
        st.html(ui.masthead(meta))
    site = None
    if registry is not None:
        with right.container(key="site-picker"):
            names = [GLOBAL] + [p.name for p in registry.usable()]
            choice = st.selectbox("Screening site", names, key="site", help="A site's threshold is calibrated on graded photographs from its own camera. See the Sites tab.")
            site = registry.get(choice) if choice != GLOBAL else None
    if metrics.get("smoke_test"):
        st.warning("These are smoke-test weights trained for seconds on synthetic images. Predictions are meaningless.")
    return site


# --------------------------------------------------------------------------- screening


def screening_tab(bundle: Fundalux, site) -> None:
    intake, report = st.columns([1, 2.05], gap="large")
    with intake, st.container(key="intake"):
        st.html(ui.step(1, "Photograph"))
        uploaded = st.file_uploader("Colour fundus photograph", type=IMAGE_TYPES, label_visibility="collapsed")
        samples = sorted(p for p in SAMPLE_DIR.glob("*") if p.suffix.lower().lstrip(".") in IMAGE_TYPES) if SAMPLE_DIR.exists() else []
        sample = None
        if samples:
            names = [p.name for p in samples]
            sample = st.selectbox("Or open a sample", ["-"] + names, format_func=lambda n: "Choose a sample ..." if n == "-" else SAMPLE_LABELS.get(n, n))
        st.html(ui.step(2, "Field conditions"))
        kind = st.selectbox("Simulate a camera problem", ["none", *KINDS], format_func=lambda k: "None - use the photograph as taken" if k == "none" else k.capitalize())
        severity = st.slider("Severity", 0.0, 1.0, 0.8, 0.05, disabled=kind == "none")
        skip_gate = st.toggle("Grade even if the image is rejected", help="Shows what the classifier would have guessed on an unreadable photo - the reason the quality gate exists.")

    if uploaded is not None:
        source, payload = "upload", uploaded.getvalue()
    elif sample and sample != "-":
        source, payload = "sample", str(SAMPLE_DIR / sample)
    else:
        with report:
            st.html(ui.empty_state())
        return

    with report:
        with st.spinner("Screening - three networks and their heatmaps, about 15-20 s on this laptop ..."):
            base = analyse(bundle, str(MODEL_DIR), source, payload, kind, severity, skip_gate)
        result = base.with_threshold(site.threshold, site.name) if site else base
        st.html(screening_report(bundle, result, site))
    st.html(ui.note(DISCLAIMER))


def screening_report(bundle: Fundalux, result, site) -> str:
    where = f"{site.name} threshold" if site else "global threshold"
    parts = []
    if not result.gradable:
        parts.append(ui.verdict("retake", "Not graded", "Retake the photograph", result.advice, result.reasons))
    elif result.referable:
        urgent = result.grade is not None and result.grade >= 3
        parts.append(ui.verdict("refer", "Refer" + (" · urgent" if urgent else ""), "Referable diabetic retinopathy suspected", result.advice))
    else:
        parts.append(ui.verdict("clear", "No referral", "No referable diabetic retinopathy", result.advice))

    if result.referable_score is not None:
        members = result.member_scores if result.member_scores and len(result.member_scores) > 1 else None
        caption = "Dot: the ensemble's score. Ticks above the track: each network's own score. Hatched: the referral zone."
        if not result.gradable:
            caption = "Shown only because grading of rejected images is switched on. Do not act on this score."
        parts.append(ui.gauge("Referable-DR score", result.referable_score, result.threshold, f"Referral at {where}", members, caption))
    if result.ungradable_score is not None:
        parts.append(ui.gauge("Ungradable score", result.ungradable_score, bundle.quality_threshold, "Rejected at", None, "From the image-quality gate, which sees the photograph before any enhancement."))

    figs = [(result.cropped, "Cropped", "to the retina; what the quality gate sees.")]
    if result.enhanced is not None:
        figs.append((result.enhanced, "Enhanced", "illumination corrected and contrast-equalised; what the classifier sees."))
        low = result.referable_score is not None and result.referable_score < result.threshold
        figs.append((ui.heat_overlay(result.enhanced, result.heatmap), "Grad-CAM", "regions that raised the score most; outline at half the peak."
                     + (" At a score this low they are the least normal-looking areas, not lesions." if low else "")))
    parts.append(ui.figures(figs))

    if result.grade_probs is not None:
        parts.append(ui.kicker("Estimated grade (secondary to the referral decision)") + ui.grade_bars(GRADE_NAMES, result.grade_probs))
    return '<div class="nx-reveal">' + "".join(parts) + "</div>"


# --------------------------------------------------------------------------- sites


def sites_tab(bundle: Fundalux, registry: SiteRegistry, metrics: dict) -> None:
    target = metrics.get("dr", {}).get("validation", {}).get("target_sensitivity", 0.93)
    crit = metrics.get("success_criteria", {})
    t_sens, t_spec = crit.get("target_sensitivity", 0.90), crit.get("target_specificity", 0.85)
    info, form = st.columns([2.3, 1], gap="large")
    with info:
        st.html(
            ui.heading("Per-site thresholds")
            + ui.lede(
                "The networks rank photographs well at every site, but their scores sit on a different scale for each camera, "
                "so one threshold refers too many patients at some sites. A site threshold is tuned on graded photographs from "
                f"that site for the same {target:.0%} sensitivity. The expected figures below were measured on images the threshold "
                "was not tuned on."
            )
        )
        rows, selected = [], None
        show_source = any(p.source != "built-in" for p in registry.profiles)
        for i, p in enumerate(registry.profiles):
            ok = None if p.stale else bool(p.expected_sensitivity > t_sens and p.expected_specificity > t_spec)
            rows.append([
                p.name + (" (stale)" if p.stale else ""),
                ui.num(p.threshold), f"{p.n_images:,} ({p.n_referable:,})",
                ui.pct(p.expected_sensitivity), ui.pct(p.expected_specificity),
                ui.verdict_cell(ok, "meets", "below"),
            ] + ([p.source or "–"] if show_source else []))
            if st.session_state.get("site") == p.name:
                selected = i
        if rows:
            st.html(ui.table([("Site", False), ("Threshold", True), ("Images (referable)", True), ("Exp. sensitivity", True), ("Exp. specificity", True), (f"Target {t_sens:.0%} / {t_spec:.0%}", False)] + ([("Source", False)] if show_source else []), rows, selected))
            st.html(ui.note(f"Global threshold {bundle.dr_threshold:.3f}, tuned on the pooled validation split. Built-in sites are the evaluation datasets; their profiles ship with the model."))
            for p in registry.profiles:
                for w in p.warnings:
                    st.html(ui.note(f"{p.name}: {w}"))
        else:
            st.html(ui.note("No site profiles yet. Run python scripts/build_site_profiles.py, or calibrate one on the right."))

        own = [p.name for p in registry.profiles if p.source != "built-in"]
        if own:
            st.html(ui.heading("Remove a site", 3))
            c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
            gone = c1.selectbox("Site to remove", own, key="remove-site")
            c2.button("Remove", key="remove-button", on_click=remove_site, args=(registry, gone))

    with form, st.container(key="calibrate"):
        st.html(ui.heading("Calibrate a new site", 3) + ui.note(
            f"Needs graded photographs from the site's own camera: at least 20 referable and 20 non-referable that pass the quality gate, "
            f"ideally 50 or more of each. Scoring takes several seconds a photograph here; for hundreds use python -m fundalux.calibrate."
        ))
        name = st.text_input("Site name", placeholder="e.g. PHC Nanded, camera 2")
        labels_file = st.file_uploader("Labels (CSV)", type=["csv"], help="Columns: image (file name) and grade (0-4) or referable (0/1).")
        photos = st.file_uploader("Photographs", type=IMAGE_TYPES, accept_multiple_files=True)
        sens = st.slider("Target sensitivity", 0.85, 0.98, float(target), 0.01)
        if st.button("Score and calibrate", type="primary", disabled=not (name and labels_file and photos)):
            run_calibration(bundle, name.strip(), labels_file, photos, sens)

        pending = st.session_state.get("pending_site")
        if pending:
            profile, summary = pending
            st.html(
                ui.headline_figures([
                    (ui.num(profile.threshold), "threshold", f"fitted on {profile.n_images} gradable photos ({profile.n_referable} referable)"),
                    (ui.pct(profile.expected_specificity, 0), "specificity", f"expected, with {ui.pct(profile.expected_sensitivity, 0)} sensitivity"),
                ])
                + ui.note(f"{summary['matched']} of {summary['labelled']} labelled photos found; {summary['rejected_by_gate']} rejected by the quality gate. {profile.estimate}.")
                + "".join(ui.note(w) for w in profile.warnings)
            )
            c1, c2 = st.columns(2)
            c1.button("Save site", type="primary", on_click=save_site, args=(registry, profile))
            c2.button("Discard", on_click=st.session_state.pop, args=("pending_site", None))


# Callbacks run before the page is redrawn, the only moment the site picker's value may be changed.
def save_site(registry: SiteRegistry, profile) -> None:
    registry.put(profile)
    st.session_state.pop("pending_site", None)
    st.session_state["site"] = profile.name


def remove_site(registry: SiteRegistry, name: str) -> None:
    registry.remove(name)
    if st.session_state.get("site") == name:
        st.session_state["site"] = GLOBAL


def run_calibration(bundle: Fundalux, name: str, labels_file, photos, target: float) -> None:
    try:
        labels = read_labels(pd.read_csv(io.BytesIO(labels_file.getvalue())))
    except (CalibrationError, ValueError) as exc:
        st.error(f"Could not read the labels: {exc}")
        return
    wanted = set(labels["key"])
    chosen = [(f.name, f.getvalue()) for f in photos if Path(f.name).stem.lower() in wanted]
    if not chosen:
        st.error("None of the photographs match a file name in the labels.")
        return
    bar = st.progress(0.0, text="Scoring ...")
    scored = score_images(bundle, chosen, progress=lambda done, total: bar.progress(done / total, text=f"Scored {done} of {total}"))
    bar.empty()
    try:
        st.session_state["pending_site"] = calibrate_site(name, labels, scored, target, source="calibrated in the app")
    except CalibrationError as exc:
        st.error(f"Not calibrated: {exc}")


# --------------------------------------------------------------------------- model report


def report_tab(metrics: dict, registry: SiteRegistry | None) -> None:
    if not metrics:
        st.html(ui.note("No metrics.json next to the models."))
        return
    crit = metrics.get("success_criteria", {})
    t_sens, t_spec = crit.get("target_sensitivity", 0.90), crit.get("target_specificity", 0.85)
    dr = metrics.get("dr", {})
    st.html(ui.heading("Held-out results") + ui.lede(
        f"Referable DR on photographs the quality gate accepted, at the global threshold ({dr.get('threshold', float('nan')):.3f}). "
        f"Plan target: sensitivity above {t_sens:.0%} and specificity above {t_spec:.0%}."
    ))
    rows = []
    for name, r in dr.get("results", {}).items():
        met = crit.get("met", {}).get(name)
        rows.append([name, f"{r['n']:,}", ui.pct(r.get("rejected_by_gate")), ui.num(r["auc"]), ui.pct(r["sensitivity"]), ui.pct(r["specificity"]), ui.verdict_cell(met, "met", "not met") if met is not None else "–"])
    st.html(ui.table([("Held-out set", False), ("Graded", True), ("Rejected", True), ("AUC", True), ("Sensitivity", True), ("Specificity", True), ("Target", False)], rows))

    if registry and registry.usable():
        st.html(ui.heading("With per-site thresholds", 3))
        global_spec = {"EyePACS": "EyePACS test", "APTOS": "APTOS test", "IDRiD": "IDRiD - external, India", "Messidor-2": "Messidor-2 - external"}
        rows = []
        for p in registry.usable():
            key = next((v for k, v in global_spec.items() if p.name.startswith(k)), None)
            g = dr.get("results", {}).get(key) if key else None
            ok = bool(p.expected_sensitivity > t_sens and p.expected_specificity > t_spec)
            rows.append([p.name, ui.pct(g["specificity"]) if g else "–", ui.pct(p.expected_sensitivity), ui.pct(p.expected_specificity), ui.verdict_cell(ok, "met", "not met")])
        st.html(ui.table([("Site", False), ("Spec. at global threshold", True), ("Sensitivity, site threshold", True), ("Specificity, site threshold", True), ("Target", False)], rows))
        st.html(ui.note("Site-threshold figures are measured on images the threshold was not tuned on (a separate test split, or cross-fitting for the external sets)."))

    member_auc = dr.get("member_auc")
    if member_auc:
        st.html(ui.heading("The ensemble and its members", 3))
        archs = list(next(iter(member_auc.values())).keys())
        rows = [[name, *[ui.num(v) for v in aucs.values()]] for name, aucs in member_auc.items()]
        st.html(ui.table([("AUC on", False), *[(a.replace("_", " "), True) for a in archs]], rows))

    cam = metrics.get("gradcam")
    if cam:
        st.html(ui.heading("Do the heatmaps point at lesions?", 3) + ui.lede(
            f"On {cam['images']} IDRiD photographs with ophthalmologists' lesion masks, compared with networks that never saw a DR label. "
            f"Sanity check {'passed' if cam.get('sanity_check_passed') else 'FAILED'}: better on {cam['beats_baseline_on']} of {cam['compared']}."
        ))
        labels = {"pointing hit rate": "Hottest point on a lesion", "energy on lesions / lesion area (median)": "Heat on lesions vs chance (median)", "pixel AUC (mean)": "Pixel AUC (mean)"}
        rows = [[labels.get(k, k), ui.num(v), ui.num(cam["baseline"].get(k))] for k, v in cam["trained"].items()]
        st.html(ui.table([("Measure", False), ("Trained ensemble", True), ("No DR training", True)], rows))

    figures = sorted((MODEL_DIR / "figures").glob("*.png"))
    if figures:
        st.html(ui.heading("Figures from training", 3))
        cols = st.columns(2, gap="medium")
        for i, fig in enumerate(figures):
            cols[i % 2].image(str(fig), caption=fig.stem.replace("_", " ").capitalize(), width="stretch")


# --------------------------------------------------------------------------- camp simulation


@st.cache_data(show_spinner="Simulating the camp ...", max_entries=32)
def run_camp(scenario: sim.Scenario):
    """Both scenarios and the allocation sweep; cached because every tab re-runs on each interaction."""
    rows = sim.sweep_allocation(scenario, seed=0)
    return sim.simulate(scenario, seed=0), sim.simulate(sim.manual_benchmark(scenario), seed=0), rows, sim.recommend(rows)


def simulation_tab(metrics: dict, site) -> None:
    st.html(ui.heading("Screening-camp simulation") + ui.lede(
        "Registration, photography with on-the-spot retakes, AI grading and remote review of referrals, compared with a camp where "
        "the ophthalmologist reads every patient. Timing, prevalence and cost values are placeholders for local figures."
    ))
    base = sim.scenario_from_metrics(metrics)
    if site is not None:
        base = replace(base, sensitivity=site.expected_sensitivity, specificity=site.expected_specificity)
    key = site.name if site else "global"  # new defaults when the site changes
    with st.container(key="camp-controls"):
        c = st.columns(4, gap="medium")
        patients = c[0].slider("Patients per day", 20, 400, int(base.patients_per_day), 10)
        cameras = c[1].slider("Camera stations", 1, 6, base.n_cameras)
        hours = c[2].slider("Ophthalmologist hours per day", 0.5, 8.0, base.ophthalmologist_hours, 0.5)
        days = c[3].slider("Days simulated", 5, 60, base.days, 5)
        c = st.columns(4, gap="medium")
        prevalence = c[0].slider("Referable DR prevalence", 0.01, 0.40, base.prevalence, 0.01)
        p_bad = c[1].slider("Ungradable photos per attempt", 0.0, 0.6, float(min(base.p_ungradable, 0.6)), 0.01)
        sens = c[2].slider("AI sensitivity", 0.5, 1.0, float(round(base.sensitivity, 2)), 0.01, key=f"sens-{key}")
        spec = c[3].slider("AI specificity", 0.5, 1.0, float(round(base.specificity, 2)), 0.01, key=f"spec-{key}")
        st.html(ui.note(f"AI accuracy defaults: {'the ' + site.name + ' site profile' if site else 'the internal test set at the global threshold'}."))
    ai = replace(base, patients_per_day=patients, n_cameras=cameras, ophthalmologist_hours=hours, days=days, prevalence=prevalence, p_ungradable=p_bad, sensitivity=sens, specificity=spec)
    a, m, sweep, best = run_camp(ai)

    st.html(ui.headline_figures([
        (f"{a['reviews_requested_per_day']:.0f}", f"vs {m['reviews_requested_per_day']:.0f}", "patients a day sent to the ophthalmologist, AI-assisted vs reading everyone"),
        (f"{a['review_backlog_end']:,}", f"vs {m['review_backlog_end']:,}", f"patients still waiting for review after {days} days"),
        (f"{a['detected']}", f"of {a['referable_cases']}", "referable patients detected and referred in the AI-assisted camp"),
        (f"{a['time_in_camp_p95_min']:.0f}", "min", "95th-percentile time a patient spends in the camp"),
    ]))
    labels = {
        "patients_per_day": ("Patients per day", 1), "time_in_camp_mean_min": ("Mean time in camp, min", 1), "time_in_camp_p95_min": ("95th-pct time in camp, min", 1),
        "imaging_attempts_per_patient": ("Photo attempts per patient", 2), "reviews_requested_per_day": ("Reviews requested per day", 1),
        "ophthalmologist_min_per_day": ("Ophthalmologist min per day", 0), "review_backlog_end": ("Backlog at the end", 0),
        "review_turnaround_mean_h": ("Mean review turnaround, h", 1), "detected": ("Referable cases detected", 0), "missed": ("Referable cases missed", 0),
        "false_referrals": ("False referrals", 0), "recalls_for_bad_images": ("Recalls for unreadable photos", 0), "cost_per_patient": ("Cost per patient", 1),
        "cost_per_detected_case": ("Cost per detected case", 0),
    }
    fmt = lambda v, d: "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:,.{d}f}"
    rows = [[label, fmt(a[k], d), fmt(m[k], d)] for k, (label, d) in labels.items()]
    st.html(ui.heading("AI-assisted camp and the manual benchmark", 3) + ui.table([("Measure", False), ("AI-assisted", True), ("Manual grading", True)], rows))

    grid = pd.DataFrame(sweep)
    hours_cols = sorted(grid.ophthalmologist_hours.unique())
    wait = grid.pivot(index="n_cameras", columns="ophthalmologist_hours", values="time_in_camp_p95_min")
    backlog = grid.pivot(index="n_cameras", columns="ophthalmologist_hours", values="review_backlog_end")
    w_lo, w_hi, b_hi = float(wait.min().min()), float(wait.max().max()), float(backlog.max().max())
    rows = [
        [f"{n} camera{'s' if n > 1 else ''}", *[ui.heat_cell(wait.loc[n, h], w_lo, w_hi, f"{wait.loc[n, h]:.0f} min · {backlog.loc[n, h]:.0f}") for h in hours_cols]]
        for n in wait.index
    ]
    st.html(
        ui.heading("Resource allocation", 3)
        + ui.note("Each cell: 95th-percentile time in camp · review backlog at the end. Shading follows waiting time, powder (short) to peach (long).")
        + ui.table([("", False), *[(f"{h:g} h ophthalmologist", True) for h in hours_cols]], rows, cls="nx-heat")
    )
    st.html(ui.note(
        f"Cheapest allocation with a 95th-percentile camp time within 60 min and no backlog: {best['n_cameras']} camera station(s) and "
        f"{best['ophthalmologist_hours']:g} ophthalmologist hour(s) a day, {best['cost_per_day']:,.0f} per day." if best else
        "No allocation in the grid keeps the camp time within 60 min with no backlog - reduce the load or add resources."
    ))


# --------------------------------------------------------------------------- page


def main() -> None:
    metrics = load_metrics(str(MODEL_DIR))
    bundle, registry, problem = None, None, None
    try:
        bundle = load_bundle(str(MODEL_DIR))
        registry = load_sites(bundle)
    except FileNotFoundError as exc:
        problem = str(exc)
    site = masthead(bundle, metrics, registry)
    if problem:
        st.warning(f"Models not found. {problem}")

    # Lazy tabs: only the open one runs (screening and the camp sweep take seconds); the URL keeps the tab.
    screen, sites, report, camp = st.tabs(["Screen", "Sites", "Model report", "Camp simulation"], key="tab", bind="query-params")
    if screen.open and bundle is not None:
        with screen:
            screening_tab(bundle, site)
    if sites.open and bundle is not None:
        with sites:
            sites_tab(bundle, registry, metrics)
    if report.open:
        with report:
            report_tab(metrics, registry)
    if camp.open:
        with camp:
            simulation_tab(metrics, site)


main()
