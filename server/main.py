"""Fundalux web application: clinician-facing pages and a JSON API.

    uvicorn server.main:app --host 0.0.0.0 --port 8000

Environment:
    FUNDALUX_MODELS         model bundle folder (default: models)
    FUNDALUX_ADMIN_TOKEN    enables calibrating and removing sites; send as "Authorization: Bearer <token>"
    FUNDALUX_MAX_UPLOAD_MB  per photograph (default 20)
    FUNDALUX_CONCURRENCY    photographs screened at the same time (default 2)

Photographs are processed in memory and never written to disk or logged.
"""

from __future__ import annotations

import base64
import hmac
import io
import json
import os
import shutil
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import cv2
import jinja2
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from fundalux import GRADE_NAMES
from fundalux import screening_sim as sim
from fundalux.calibrate import calibrate_site, read_labels, score_images
from fundalux.degrade import KINDS, degrade
from fundalux.pipeline import DISCLAIMER, Fundalux, referral_advice
from fundalux.preprocess import decode_image
from fundalux.sites import CalibrationError, SiteRegistry, model_fingerprint

from .report import build_report, clean

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("FUNDALUX_MODELS", ROOT / "models"))
SAMPLE_DIR = ROOT / "samples"
MAX_UPLOAD = int(float(os.environ.get("FUNDALUX_MAX_UPLOAD_MB", "20")) * 1024 * 1024)
MAX_CALIBRATION_PHOTOS = 1000
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
SAMPLES = {
    "idrid_test_IDRiD_029_grade0.jpg": "IDRiD 029 · graded No DR",
    "idrid_test_IDRiD_008_grade2.jpg": "IDRiD 008 · graded Moderate NPDR",
    "idrid_test_IDRiD_001_grade4.jpg": "IDRiD 001 · graded Proliferative DR",
}


class State:
    """Process-wide state, loaded once at startup."""

    bundle: Fundalux | None = None
    error: str | None = None
    report: dict = {}
    registry: SiteRegistry | None = None
    registry_lock = threading.Lock()
    screening = threading.BoundedSemaphore(int(os.environ.get("FUNDALUX_CONCURRENCY", "2")))
    jobs: dict[str, dict] = {}
    calibrating = threading.Lock()


S = State()


def load_state() -> None:
    try:
        S.bundle = Fundalux.load(MODEL_DIR)
        S.registry = SiteRegistry.load(MODEL_DIR, S.bundle.config)
        S.report = build_report(MODEL_DIR)
        S.error = None
    except FileNotFoundError as exc:
        S.bundle, S.registry, S.report, S.error = None, None, {}, str(exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_state()
    yield


app = FastAPI(title="Fundalux", lifespan=lifespan, docs_url="/api/docs", redoc_url=None)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=HERE / "templates")


def _missing(v) -> bool:
    return v is None or isinstance(v, jinja2.Undefined)


def _pct(v, digits: int = 1) -> str:
    return "–" if _missing(v) else f"{v * 100:.{digits}f}%"


def _num(v, digits: int = 3) -> str:
    return "–" if _missing(v) else f"{v:.{digits}f}"


def _ci(pair, digits: int = 1) -> str:
    return "" if _missing(pair) or not pair or None in pair else f"{pair[0] * 100:.{digits}f}–{pair[1] * 100:.{digits}f}"


templates.env.filters.update(pct=_pct, num=_num, ci=_ci)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Content-Security-Policy", "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; font-src 'self'")
    return response


# --------------------------------------------------------------------------- helpers


def require_bundle() -> Fundalux:
    if S.bundle is None:
        raise HTTPException(503, S.error or "Models are not loaded.")
    return S.bundle


def require_admin(authorization: str | None = Header(default=None)) -> None:
    token = os.environ.get("FUNDALUX_ADMIN_TOKEN", "")
    if not token:
        raise HTTPException(403, "Site calibration is switched off on this server. Set FUNDALUX_ADMIN_TOKEN to enable it.")
    given = (authorization or "").removeprefix("Bearer ").strip()
    if not hmac.compare_digest(given.encode(), token.encode()):
        raise HTTPException(401, "Wrong or missing admin token.")


async def read_upload(file: UploadFile, limit: int = MAX_UPLOAD) -> bytes:
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"{file.filename} is larger than {limit // (1024 * 1024)} MB.")
    return data


def data_uri(img: np.ndarray, quality: int = 88) -> str:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def site_summary(p) -> dict:
    d = asdict(p)
    d["warnings"] = p.warnings
    return clean(d)


def decisions(result, bundle: Fundalux) -> list[dict]:
    """The same score decided at the global threshold and at every calibrated site."""
    options = [{"site": None, "label": "Global threshold", "threshold": bundle.dr_threshold, "expected_sensitivity": None, "expected_specificity": None}]
    for p in (S.registry.usable() if S.registry else []):
        options.append({"site": p.name, "label": p.name, "threshold": p.threshold, "expected_sensitivity": p.expected_sensitivity, "expected_specificity": p.expected_specificity})
    for o in options:
        decided = result.with_threshold(o["threshold"], o["site"])
        o["referable"] = decided.referable
        o["advice"] = decided.advice if result.gradable else referral_advice(False, None, None)
    return options


# --------------------------------------------------------------------------- pages


def page(request: Request, name: str, **context):
    hero_file = HERE / "static" / "img" / "hero.json"
    hero = json.loads(hero_file.read_text()) if hero_file.exists() else None
    figures = {p.name for p in (MODEL_DIR / "figures").glob("*.png")}
    return templates.TemplateResponse(request, name, {"r": S.report, "error": S.error, "disclaimer": DISCLAIMER, "grade_names": GRADE_NAMES, "page": name.split(".")[0], "hero": hero, "figures": figures, **context})


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return page(request, "home.html")


@app.get("/how-it-works", response_class=HTMLResponse)
def guide(request: Request):
    return page(request, "guide.html")


@app.get("/evidence", response_class=HTMLResponse)
def evidence(request: Request):
    return page(request, "evidence.html")


@app.get("/screen", response_class=HTMLResponse)
def screen_page(request: Request):
    return page(request, "screen.html", samples=SAMPLES, kinds=KINDS)


@app.get("/sites", response_class=HTMLResponse)
def sites_page(request: Request):
    return page(request, "sites.html", admin_enabled=bool(os.environ.get("FUNDALUX_ADMIN_TOKEN")))


@app.get("/camp", response_class=HTMLResponse)
def camp_page(request: Request):
    return page(request, "camp.html")


@app.get("/figures/{name}")
def figure(name: str):
    path = (MODEL_DIR / "figures" / name).resolve()
    if path.parent != (MODEL_DIR / "figures").resolve() or path.suffix != ".png" or not path.exists():
        raise HTTPException(404)
    return FileResponse(path, headers={"Cache-Control": "public, max-age=3600"})


@app.get("/samples/{name}")
def sample(name: str):
    if name not in SAMPLES:
        raise HTTPException(404)
    return FileResponse(SAMPLE_DIR / name, headers={"Cache-Control": "public, max-age=86400"})


# --------------------------------------------------------------------------- API


@app.get("/api/health")
def health():
    return {"ok": S.bundle is not None, "backend": S.bundle.backend if S.bundle else None, "error": S.error}


@app.get("/api/report")
def report():
    return S.report


@app.post("/api/screen")
async def screen(
    photo: UploadFile | None = File(default=None),
    sample_name: str | None = Form(default=None),
    degrade_kind: str = Form(default="none"),
    degrade_severity: float = Form(default=0.0, ge=0.0, le=1.0),
    grade_rejected: bool = Form(default=False),
):
    bundle = require_bundle()
    if photo is not None and photo.filename:
        raw = await read_upload(photo)
    elif sample_name in SAMPLES:
        raw = (SAMPLE_DIR / sample_name).read_bytes()
    else:
        raise HTTPException(422, "Send a photograph or choose a sample.")
    if degrade_kind not in ("none", *KINDS):
        raise HTTPException(422, f"Unknown field condition {degrade_kind!r}.")
    # The models are CPU-bound: run them on a worker thread so the server keeps answering.
    result, elapsed = await run_in_threadpool(_screen_sync, bundle, raw, degrade_kind, degrade_severity, grade_rejected)
    return JSONResponse(clean(_screen_response(result, bundle, elapsed)))


def _screen_sync(bundle: Fundalux, raw: bytes, kind: str, severity: float, grade_rejected: bool):
    try:
        image = decode_image(raw)
    except Exception:
        image = None
    if image is None or image.ndim != 3:
        raise HTTPException(422, "This file is not a colour image Fundalux can read.")
    if kind != "none" and severity > 0:
        image = degrade(image, kind, severity, np.random.default_rng(0))
    if not S.screening.acquire(timeout=90):
        raise HTTPException(503, "The server is busy screening other photographs. Try again in a minute.")
    try:
        started = time.perf_counter()
        return bundle.predict(image, skip_gate=grade_rejected), time.perf_counter() - started
    finally:
        S.screening.release()


def _screen_response(result, bundle: Fundalux, elapsed: float) -> dict:
    return {
        "gradable": result.gradable,
        "reasons": result.reasons,
        "ungradable_score": result.ungradable_score,
        "quality_threshold": bundle.quality_threshold,
        "referable_score": result.referable_score,
        "member_scores": result.member_scores,
        "grade": result.grade,
        "grade_name": result.grade_name,
        "grade_probs": result.grade_probs,
        "decisions": decisions(result, bundle),
        "images": {
            "cropped": data_uri(result.cropped),
            "enhanced": data_uri(result.enhanced) if result.enhanced is not None else None,
            "heatmap": data_uri(result.overlay) if result.overlay is not None else None,
        },
        "seconds": elapsed,
        "model": bundle.config.get("trained_with", {}).get("run"),
        "disclaimer": DISCLAIMER,
    }


@app.get("/api/sites")
def sites():
    return [site_summary(p) for p in (S.registry.profiles if S.registry else [])]


def _run_calibration(job_id: str, name: str, labels: pd.DataFrame, photos: list[tuple[str, bytes]], target: float) -> None:
    job = S.jobs[job_id]
    try:
        with S.calibrating:
            job["status"] = "scoring"

            def progress(done, total):
                job["done"] = done

            scored = score_images(S.bundle, photos, progress=progress)
            profile, summary = calibrate_site(name, labels, scored, target, source="calibrated on the web")
            job.update(status="done", profile=profile, summary=summary)
    except CalibrationError as exc:
        job.update(status="failed", error=str(exc))
    except Exception as exc:  # report, never crash the worker thread
        job.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        photos.clear()  # drop the photographs as soon as they are scored


@app.post("/api/sites/calibrate", dependencies=[Depends(require_admin)])
async def calibrate(
    name: str = Form(..., min_length=2, max_length=80),
    target_sensitivity: float = Form(default=0.93, ge=0.8, le=0.99),
    labels: UploadFile = File(...),
    photos: list[UploadFile] = File(...),
):
    require_bundle()
    if S.calibrating.locked():
        raise HTTPException(409, "Another calibration is running. Wait for it to finish.")
    try:
        table = read_labels(pd.read_csv(io.BytesIO(await read_upload(labels, 5 * 1024 * 1024))))
    except (CalibrationError, ValueError, pd.errors.ParserError) as exc:
        raise HTTPException(422, f"Could not read the labels: {exc}")
    if len(photos) > MAX_CALIBRATION_PHOTOS:
        raise HTTPException(413, f"At most {MAX_CALIBRATION_PHOTOS} photographs per calibration.")
    wanted = set(table["key"])
    chosen = []
    for f in photos:
        if Path(f.filename or "").suffix.lower() in IMAGE_SUFFIXES and Path(f.filename).stem.lower() in wanted:
            chosen.append((Path(f.filename).name, await read_upload(f)))
    if not chosen:
        raise HTTPException(422, "None of the photographs match a file name in the labels.")
    job_id = uuid.uuid4().hex
    S.jobs[job_id] = {"status": "queued", "done": 0, "total": len(chosen), "name": name.strip(), "created": time.time()}
    threading.Thread(target=_run_calibration, args=(job_id, name.strip(), table, chosen, target_sensitivity), daemon=True).start()
    return {"job": job_id, "total": len(chosen), "labelled": len(table)}


@app.get("/api/jobs/{job_id}", dependencies=[Depends(require_admin)])
def job(job_id: str):
    j = S.jobs.get(job_id)
    if j is None:
        raise HTTPException(404, "No such calibration.")
    out = {k: v for k, v in j.items() if k not in ("profile",)}
    if j.get("profile") is not None:
        out["profile"] = site_summary(j["profile"])
    return clean(out)


@app.post("/api/jobs/{job_id}/save", dependencies=[Depends(require_admin)])
def save_job(job_id: str):
    j = S.jobs.get(job_id)
    if j is None or j.get("status") != "done":
        raise HTTPException(409, "This calibration has not finished.")
    with S.registry_lock:
        S.registry.put(j["profile"])
        S.report = build_report(MODEL_DIR)
    S.jobs.pop(job_id, None)
    return {"saved": j["profile"].name}


@app.delete("/api/sites/{name}", dependencies=[Depends(require_admin)])
def remove_site(name: str):
    p = S.registry.get(name) if S.registry else None
    if p is None:
        raise HTTPException(404, "No such site.")
    if p.source == "built-in":
        raise HTTPException(409, "Built-in sites come with the model and cannot be removed here.")
    with S.registry_lock:
        S.registry.remove(name)
        S.report = build_report(MODEL_DIR)
    return {"removed": name}


class CampRequest(BaseModel):
    patients_per_day: float = Field(80, ge=10, le=500)
    n_cameras: int = Field(2, ge=1, le=8)
    ophthalmologist_hours: float = Field(2.0, ge=0.5, le=10)
    days: int = Field(20, ge=5, le=60)
    prevalence: float = Field(0.10, ge=0.01, le=0.5)
    p_ungradable: float = Field(0.15, ge=0.0, le=0.7)
    sensitivity: float = Field(0.92, ge=0.5, le=1.0)
    specificity: float = Field(0.85, ge=0.5, le=1.0)


@lru_cache(maxsize=64)
def _camp(params: tuple) -> dict:
    scenario = sim.Scenario(**dict(params))
    sweep = sim.sweep_allocation(scenario, seed=0)
    return clean({"ai": sim.simulate(scenario, seed=0), "manual": sim.simulate(sim.manual_benchmark(scenario), seed=0), "sweep": sweep, "best": sim.recommend(sweep)})


@app.post("/api/camp")
def camp(req: CampRequest):
    return _camp(tuple(sorted(req.model_dump().items())))


@app.get("/api/camp/defaults")
def camp_defaults():
    metrics_inputs = {"simulation_inputs": {}}
    if S.report:
        internal = S.report.get("sets", {}).get("internal", {}).get("global", {})
        metrics_inputs["simulation_inputs"] = {"sensitivity": internal.get("sensitivity"), "specificity": internal.get("specificity")}
    base = sim.scenario_from_metrics(metrics_inputs)
    return clean({k: getattr(base, k) for k in CampRequest.model_fields})


# --------------------------------------------------------------------------- clinician review

from fundalux import review as R  # noqa: E402

REVIEW_DIR = Path(os.environ.get("FUNDALUX_REVIEW_DIR", ROOT / "review"))


class Review:
    set: R.ReviewSet | None = None
    store: R.ResponseStore | None = None


def load_review() -> None:
    Review.set = R.ReviewSet.load(REVIEW_DIR)
    Review.store = R.ResponseStore(REVIEW_DIR / R.DATABASE) if Review.set else None


def require_reviewer(authorization: str | None = Header(default=None)) -> None:
    """Doctors use the review code; the admin token works too."""
    codes = [c for c in (os.environ.get("FUNDALUX_REVIEW_TOKEN", ""), os.environ.get("FUNDALUX_ADMIN_TOKEN", "")) if c]
    if not codes:
        raise HTTPException(403, "Clinician review is switched off on this server. Set FUNDALUX_REVIEW_TOKEN to enable it.")
    given = (authorization or "").removeprefix("Bearer ").strip().encode()
    if not any(hmac.compare_digest(given, c.encode()) for c in codes):
        raise HTTPException(401, "Wrong or missing review code.")


def require_set() -> R.ReviewSet:
    if Review.set is None:
        load_review()
    if Review.set is None:
        raise HTTPException(404, "No review set on this server yet.")
    return Review.set


def _reviewer(name: str) -> str:
    name = (name or "").strip()
    if not 1 <= len(name) <= 60:
        raise HTTPException(422, "Give a reviewer name or initials (up to 60 characters).")
    return name


def _set_threshold(site_name: str | None) -> tuple[float, str]:
    bundle = require_bundle()
    p = S.registry.get(site_name) if (S.registry and site_name) else None
    if p is not None and not p.stale:
        return p.threshold, f"{p.name} threshold"
    return bundle.dr_threshold, "global threshold"


@app.get("/review", response_class=HTMLResponse)
def review_page(request: Request):
    if Review.set is None:
        load_review()
    return page(request, "review.html", review=Review.set.manifest if Review.set else None,
                review_enabled=bool(os.environ.get("FUNDALUX_REVIEW_TOKEN") or os.environ.get("FUNDALUX_ADMIN_TOKEN")))


@app.get("/review/results", response_class=HTMLResponse)
def review_results_page(request: Request):
    if Review.set is None:
        load_review()
    return page(request, "review_results.html", review=Review.set.manifest if Review.set else None)


@app.get("/api/review/next", dependencies=[Depends(require_reviewer)])
def review_next(reviewer: str, skip: str = ""):
    """The reviewer's next unanswered photograph; skipped ones come back once the rest are done."""
    rs, name = require_set(), _reviewer(reviewer)
    done, skipped = Review.store.answered(name), set(filter(None, skip.split(",")))
    order = rs.order_for(name)
    remaining = [i for i in order if i not in done]
    pick = next((i for i in remaining if i not in skipped), remaining[0] if remaining else None)
    return {"item_id": pick, "answered": len(done & set(order)), "total": len(order)}


@app.get("/api/review/image/{item_id}", dependencies=[Depends(require_reviewer)])
def review_image(item_id: str):
    it = require_set().item(item_id)
    if it is None:
        raise HTTPException(404)
    return FileResponse(REVIEW_DIR / it["file"], headers={"Cache-Control": "private, max-age=3600"})


class Answer(BaseModel):
    reviewer: str = Field(min_length=1, max_length=60)
    item_id: str = Field(max_length=20)
    gradable: bool
    grade: int | None = Field(default=None, ge=0, le=4)
    other_referral: bool = False
    confidence: str | None = Field(default=None, pattern="^(low|medium|high)$")
    note: str = Field(default="", max_length=2000)
    seconds: float | None = Field(default=None, ge=0, le=86400)
    mode: str = Field(default="blinded", pattern="^(blinded|teaching)$")


@app.post("/api/review/answer", dependencies=[Depends(require_reviewer)])
def review_answer(a: Answer):
    if require_set().item(a.item_id) is None:
        raise HTTPException(404, "No such photograph in the review set.")
    try:
        Review.store.save(_reviewer(a.reviewer), a.item_id, a.gradable, a.grade, a.other_referral, a.confidence, a.note, a.seconds, a.mode)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    return {"saved": True}


@app.get("/api/review/model/{item_id}", dependencies=[Depends(require_reviewer)])
def review_model(item_id: str, reviewer: str):
    """Teaching mode only: the model's answer, revealed after the reviewer has committed theirs."""
    rs, name = require_set(), _reviewer(reviewer)
    it = rs.item(item_id)
    if it is None:
        raise HTTPException(404)
    if item_id not in Review.store.answered(name):
        raise HTTPException(409, "Grade the photograph first; the model's answer is shown only afterwards.")
    threshold, label = _set_threshold(rs.manifest.get("site"))
    m = it["model"]
    heat = None
    if it.get("heatmap"):
        heat = "data:image/jpeg;base64," + base64.b64encode((REVIEW_DIR / it["heatmap"]).read_bytes()).decode()
    referable = bool(m["gradable"] and m["referable_score"] is not None and m["referable_score"] >= threshold)
    return clean({**m, "grade_name": GRADE_NAMES[m["grade"]] if m.get("grade") is not None else None, "threshold": threshold,
                  "threshold_label": label, "referable": referable, "heatmap": heat})


@app.get("/api/review/results", dependencies=[Depends(require_admin)])
def review_results(site: str | None = None):
    rs = require_set()
    threshold, label = _set_threshold(site if site is not None else rs.manifest.get("site"))
    result = R.analyse(rs, Review.store.frame(), threshold, label)
    result["set"] = {k: rs.manifest.get(k) for k in ("name", "created", "site", "model_run")}
    result["stale_model"] = bool(S.bundle and rs.manifest.get("model_fingerprint") not in (None, model_fingerprint(S.bundle.config)))
    return clean(result)


@app.get("/api/review/export.csv", dependencies=[Depends(require_admin)])
def review_export(site: str | None = None):
    rs = require_set()
    threshold, _ = _set_threshold(site if site is not None else rs.manifest.get("site"))
    frame = R.export_frame(rs, Review.store.frame(), threshold)
    return Response(frame.to_csv(index=False), media_type="text/csv", headers={"Content-Disposition": 'attachment; filename="fundalux_review.csv"'})


@app.get("/api/review/thumb/{item_id}", dependencies=[Depends(require_admin)])
def review_thumb(item_id: str):
    return review_image(item_id)


def _build_set(job_id: str, name: str, photos: list[tuple[str, bytes]], references: dict, site: str | None) -> None:
    job = S.jobs[job_id]
    try:
        with S.calibrating:
            job["status"] = "scoring"
            tmp = REVIEW_DIR.with_name(REVIEW_DIR.name + ".new")
            shutil.rmtree(tmp, ignore_errors=True)
            R.build_review_set(S.bundle, photos, tmp, name, references, site, progress=lambda d, t: job.update(done=d))
            old = REVIEW_DIR.with_name(REVIEW_DIR.name + ".old")
            shutil.rmtree(old, ignore_errors=True)
            if REVIEW_DIR.exists():
                REVIEW_DIR.rename(old)  # answers to the previous set are kept in review.old/
            tmp.rename(REVIEW_DIR)
            load_review()
            job.update(status="done", items=len(Review.set.items))
    except Exception as exc:
        job.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        photos.clear()


@app.post("/api/review/set", dependencies=[Depends(require_admin)])
async def review_new_set(
    name: str = Form(..., min_length=2, max_length=80),
    site: str = Form(default=""),
    photos: list[UploadFile] = File(...),
    labels: UploadFile | None = File(default=None),
):
    require_bundle()
    if S.calibrating.locked():
        raise HTTPException(409, "A calibration or another set is being processed. Wait for it to finish.")
    if len(photos) > 300:
        raise HTTPException(413, "At most 300 photographs in a review set.")
    references = {}
    if labels is not None and labels.filename:
        try:
            table = pd.read_csv(io.BytesIO(await read_upload(labels, 5 * 1024 * 1024)))
            cols = {c.lower().strip(): c for c in table.columns}
            image_col = next(cols[c] for c in ("image", "image_id", "id_code", "file", "filename") if c in cols)
            grade_col = next(cols[c] for c in ("grade", "diagnosis", "level") if c in cols)
            references = {Path(str(k)).stem.lower(): int(v) for k, v in zip(table[image_col], table[grade_col]) if pd.notna(v)}
        except (StopIteration, ValueError, pd.errors.ParserError):
            raise HTTPException(422, "The labels CSV needs an image column and a grade column (0-4).")
    chosen = [(Path(f.filename).name, await read_upload(f)) for f in photos if Path(f.filename or "").suffix.lower() in IMAGE_SUFFIXES]
    if not chosen:
        raise HTTPException(422, "No photographs received.")
    job_id = uuid.uuid4().hex
    S.jobs[job_id] = {"status": "queued", "done": 0, "total": len(chosen), "kind": "review-set", "created": time.time()}
    threading.Thread(target=_build_set, args=(job_id, name.strip(), chosen, references, site.strip() or None), daemon=True).start()
    return {"job": job_id, "total": len(chosen)}
