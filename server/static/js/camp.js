// Screening-camp simulation: controls -> /api/camp -> headline figures, comparison and allocation grid.
"use strict";
(() => {
  const { esc, json } = window.NX;
  const $ = (id) => document.getElementById(id);
  const KEYS = ["patients_per_day", "n_cameras", "ophthalmologist_hours", "days", "prevalence", "p_ungradable", "sensitivity", "specificity"];
  const fmt = (v, d = 0) => (v == null ? "–" : Number(v).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d }));
  let timer = null, pending = null;

  const ROWS = [
    ["patients_per_day", "Patients per day", 1], ["time_in_camp_mean_min", "Mean time in camp, min", 1], ["time_in_camp_p95_min", "95th-percentile time in camp, min", 1],
    ["imaging_attempts_per_patient", "Photo attempts per patient", 2], ["reviews_requested_per_day", "Reviews requested per day", 1], ["ophthalmologist_min_per_day", "Ophthalmologist minutes per day", 0],
    ["review_backlog_end", "Patients still waiting for review", 0], ["review_turnaround_mean_h", "Mean review turnaround, h", 1], ["detected", "Referable patients detected", 0],
    ["missed", "Referable patients missed", 0], ["false_referrals", "False referrals", 0], ["recalls_for_bad_images", "Recalls for unreadable photos", 0],
    ["cost_per_patient", "Cost per patient", 1], ["cost_per_detected_case", "Cost per detected case", 0],
  ];

  function shade(value, lo, hi) {
    const f = hi <= lo ? 0 : Math.min(Math.max((value - lo) / (hi - lo), 0), 1);
    const a = [214, 229, 250], b = [251, 227, 217];
    return `rgb(${a.map((x, i) => Math.round(x * (1 - f) + b[i] * f)).join(",")})`;
  }

  function render(res, params) {
    const a = res.ai, m = res.manual;
    const hours = [...new Set(res.sweep.map((r) => r.ophthalmologist_hours))].sort((x, y) => x - y);
    const cams = [...new Set(res.sweep.map((r) => r.n_cameras))].sort((x, y) => x - y);
    const waits = res.sweep.map((r) => r.time_in_camp_p95_min), lo = Math.min(...waits), hi = Math.max(...waits);
    const cell = (c, h) => res.sweep.find((r) => r.n_cameras === c && r.ophthalmologist_hours === h);
    $("camp-results").innerHTML = `<div class="reveal">
      <div class="big-figures">
        <div><b>${fmt(a.reviews_requested_per_day)}</b><small>vs ${fmt(m.reviews_requested_per_day)}</small><p>patients a day sent to the ophthalmologist, with AI triage vs reading everyone</p></div>
        <div><b>${fmt(a.review_backlog_end)}</b><small>vs ${fmt(m.review_backlog_end)}</small><p>patients still waiting for review after ${params.days} days</p></div>
        <div><b>${fmt(a.detected)}</b><small>of ${fmt(a.referable_cases)}</small><p>referable patients detected and referred in the AI-assisted camp</p></div>
        <div><b>${fmt(a.time_in_camp_p95_min)}</b><small>min</small><p>95th-percentile time a patient spends in the camp</p></div>
      </div>
      <div class="cols-2" style="grid-template-columns:minmax(0,.9fr) minmax(0,1.1fr);margin-top:2rem">
        <div><h3 style="margin-bottom:.6rem">AI-assisted camp and the benchmark</h3><div class="scroll"><table class="table">
          <thead><tr><th>Measure</th><th class="r">AI-assisted</th><th class="r">Reading everyone</th></tr></thead>
          <tbody>${ROWS.map(([k, label, d]) => `<tr><td>${label}</td><td class="r">${fmt(a[k], d)}</td><td class="r">${fmt(m[k], d)}</td></tr>`).join("")}</tbody>
        </table></div></div>
        <div><h3 style="margin-bottom:.6rem">Resource allocation</h3>
          <p class="note" style="margin-top:0">Rows: camera stations. Columns: ophthalmologist reading hours a day. Each cell: 95th-percentile time in camp, and patients still waiting for review at the end. Shading follows waiting time, powder (short) to peach (long).</p>
          <div class="scroll"><table class="table">
            <thead><tr><th>Cameras</th>${hours.map((h) => `<th class="r">${h} h / day</th>`).join("")}</tr></thead>
            <tbody>${cams.map((c) => `<tr><td style="white-space:nowrap">${c}</td>${hours.map((h) => { const r = cell(c, h); return `<td class="heatcell" style="background:${shade(r.time_in_camp_p95_min, lo, hi)};white-space:nowrap">${fmt(r.time_in_camp_p95_min)} min<span class="ci">${fmt(r.review_backlog_end)} waiting</span></td>`; }).join("")}</tr>`).join("")}</tbody>
          </table></div>
          <p class="note">${res.best ? `Cheapest allocation keeping the 95th-percentile camp time within 60 minutes with no backlog: <b>${res.best.n_cameras} camera station(s), ${res.best.ophthalmologist_hours} ophthalmologist hour(s) a day</b>, ${fmt(res.best.cost_per_day)} a day.` : "No allocation in the grid keeps the camp time within 60 minutes with no backlog: reduce the load or add resources."}</p>
        </div>
      </div></div>`;
  }

  async function run() {
    const params = Object.fromEntries(KEYS.map((k) => [k, Number($(k).value)]));
    pending?.abort();
    pending = new AbortController();
    $("camp-results").style.opacity = ".55";
    try {
      const res = await json("/api/camp", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(params), signal: pending.signal });
      render(res, params);
    } catch (err) {
      if (err.name !== "AbortError") $("camp-results").innerHTML = `<p class="error" role="alert">${esc(err.message)}</p>`;
    } finally { $("camp-results").style.opacity = ""; }
  }

  const show = (k) => { const el = $(k); el.parentElement.querySelector("output").textContent = Number(el.value).toString(); };
  const schedule = () => { clearTimeout(timer); timer = setTimeout(run, 350); };

  document.addEventListener("DOMContentLoaded", async () => {
    const defaults = await json("/api/camp/defaults");
    KEYS.forEach((k) => {
      const el = $(k);
      el.value = Math.round(defaults[k] / Number(el.step)) * Number(el.step);
      show(k);
      el.addEventListener("input", () => { show(k); schedule(); });
    });
    $("accuracy").addEventListener("change", (e) => {
      const [s, p] = e.target.value ? e.target.value.split("|").map(Number) : [defaults.sensitivity, defaults.specificity];
      $("sensitivity").value = s.toFixed(2); $("specificity").value = p.toFixed(2);
      show("sensitivity"); show("specificity"); schedule();
    });
    run();
  });
})();
