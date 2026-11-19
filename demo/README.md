# Demo screenshots

Fundalux running with the run 2 models (three-network ensemble), captured from the Docker container on
19 Nov 2026. The sample photographs are IDRiD test images, which the networks never trained on.
Retake them with `python scripts/capture_demo.py` while the app is running.

| Screenshot | What it shows |
|---|---|
| [01_overview](01_overview.jpg) | Home page: what Fundalux does, results at a glance, intended use, limitations |
| [02_how_it_works](02_how_it_works.jpg) | Clinician's guide: grading scale, every processing step, thresholds, reading a heatmap, model card |
| [03_evidence](03_evidence.jpg) | Evidence page: ROC curves with operating points, results by grade, confusion matrices, heatmap validation, robustness |
| [04_evidence_external_test_set](04_evidence_external_test_set.jpg) | Evidence for a hospital the networks never saw (Messidor-2, France) |
| [05_screen_referable_pdr](05_screen_referable_pdr.jpg) | Proliferative DR: referred urgently at the IDRiD site threshold, with the heatmap over the lesions |
| [06_screen_no_dr](06_screen_no_dr.jpg) | A healthy eye: no referral, routine re-screening |
| [07_screen_refused_blurred_photo](07_screen_refused_blurred_photo.jpg) | A photograph blurred on purpose: refused by the quality gate, with the reason and retake advice |
| [08_site_calibration](08_site_calibration.jpg) | Per-camera thresholds and calibrating a new site |
| [09_screening_camp_model](09_screening_camp_model.jpg) | Simulated screening camp with and without AI triage, and the staffing recommendation |
| [10_review_start](10_review_start.jpg) | Clinician review: starting a blinded session |
| [11_review_blinded_grading](11_review_blinded_grading.jpg) | Grading a photograph without seeing the system's answer |
| [12_review_results_organiser](12_review_results_organiser.jpg) | The organiser's results page (no answers yet) and creating a review set |
| [13_phone_overview](13_phone_overview.jpg) | Home page on a phone |
| [14_phone_screening_report](14_phone_screening_report.jpg) | A screening report on a phone |
