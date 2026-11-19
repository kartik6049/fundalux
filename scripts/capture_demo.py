"""Screenshot every part of the running app into demo/.

    python scripts/capture_demo.py [--url http://localhost:8000]

Needs Playwright with Chrome (pip install playwright) and a running server. The review pages use the
codes in .env; the script opens a review session but never submits an answer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "demo"
SITE = "IDRiD (Nanded, India)"  # the samples are IDRiD photographs, so show that site's threshold
SAMPLES = {"pdr": "idrid_test_IDRiD_001_grade4.jpg", "moderate": "idrid_test_IDRiD_008_grade2.jpg", "normal": "idrid_test_IDRiD_029_grade0.jpg"}


def env_codes() -> dict:
    path = ROOT / ".env"
    if not path.exists():
        return {}
    return dict(line.strip().split("=", 1) for line in path.read_text().splitlines() if "=" in line and not line.startswith("#"))


def shoot(page, name, full=True):
    page.wait_for_timeout(600)
    page.screenshot(path=OUT / f"{name}.jpg", full_page=full, type="jpeg", quality=80)
    print("saved", name)


def screen(page, sample, kind="none", severity=None, grade_rejected=False):
    """Screen a bundled sample, optionally with a simulated camera defect, and wait for the report."""
    if kind != "none":
        page.select_option("#kind", kind)
        page.wait_for_timeout(300)
    if severity is not None:
        page.fill("#severity", str(severity))
    if grade_rejected:
        page.check("#grade-rejected")
    with page.expect_response(lambda r: r.url.endswith("/api/screen"), timeout=180_000):
        page.click(f"[data-sample='{sample}']")
    page.wait_for_selector("#print:not([disabled])", timeout=60_000)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    url = ap.parse_args().url
    codes = env_codes()
    OUT.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        ctx = browser.new_context(viewport={"width": 1360, "height": 860}, device_scale_factor=1.25)
        page = ctx.new_page()

        for name, path in (("01_overview", "/"), ("02_how_it_works", "/how-it-works"), ("03_evidence", "/evidence")):
            page.goto(url + path, wait_until="networkidle")
            shoot(page, name)
        tabs = page.locator("[id^='tab-']")
        if tabs.count() > 1:
            tabs.nth(tabs.count() - 1).click()
            shoot(page, "04_evidence_external_test_set")

        page.goto(url + "/screen", wait_until="networkidle")
        screen(page, SAMPLES["pdr"])
        # The site menu is filled from each screening's decisions; the choice is remembered for later screenings.
        page.select_option("#site", SITE)
        shoot(page, "05_screen_referable_pdr")
        screen(page, SAMPLES["normal"])
        shoot(page, "06_screen_no_dr")
        page.goto(url + "/screen", wait_until="networkidle")
        screen(page, SAMPLES["moderate"], kind="blur", severity=0.8)
        shoot(page, "07_screen_refused_blurred_photo")

        page.goto(url + "/sites", wait_until="networkidle")
        shoot(page, "08_site_calibration")

        page.goto(url + "/camp", wait_until="networkidle")
        page.wait_for_selector("#camp-results table, #camp-results svg", timeout=120_000)
        shoot(page, "09_screening_camp_model")

        if codes.get("FUNDALUX_REVIEW_TOKEN"):
            page.goto(url + "/review", wait_until="networkidle")
            shoot(page, "10_review_start", full=False)
            page.fill("#reviewer", "Demo screenshots")
            page.fill("#code", codes["FUNDALUX_REVIEW_TOKEN"])
            page.click("#start-form button[type=submit]")
            page.wait_for_selector("#photo[src^='blob:']", timeout=60_000)
            shoot(page, "11_review_blinded_grading", full=False)
        if codes.get("FUNDALUX_ADMIN_TOKEN"):
            page.goto(url + "/review/results", wait_until="networkidle")
            page.fill("#token", codes["FUNDALUX_ADMIN_TOKEN"])
            page.click("#auth button[type=submit]")
            page.wait_for_selector("#set-info:not(:empty)", timeout=60_000)
            shoot(page, "12_review_results_organiser")

        phone = browser.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=2, is_mobile=True, has_touch=True)
        mp = phone.new_page()
        mp.goto(url + "/", wait_until="networkidle")
        shoot(mp, "13_phone_overview", full=False)
        mp.goto(url + "/screen", wait_until="networkidle")
        screen(mp, SAMPLES["pdr"])
        shoot(mp, "14_phone_screening_report")

        browser.close()


if __name__ == "__main__":
    main()
