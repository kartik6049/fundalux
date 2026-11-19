// Screening tool: send a photograph, render the report, switch sites without re-scoring.
"use strict";
(() => {
  const { esc, pct, num, json } = window.NX;
  const $ = (id) => document.getElementById(id);
  const GRADES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"];
  const ICONS = {
    refer: '<circle cx="10" cy="10" r="8.5"/><path d="M7 13l6-6M8 7h5v5"/>',
    clear: '<circle cx="10" cy="10" r="8.5"/><path d="M6.5 10.2l2.4 2.4 4.6-5"/>',
    retake: '<rect x="2.5" y="5.5" width="15" height="10.5" rx="1.5"/><path d="M7 5.5l1.2-2h3.6l1.2 2"/><circle cx="10" cy="10.7" r="2.8"/><path d="M3 17L17 3"/>',
  };
  const icon = (name, colour) => `<svg width="26" height="26" viewBox="0 0 20 20" fill="none" stroke="${colour}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[name]}</svg>`;

  let source = null;   // {file} or {sample, label}
  let last = null;     // last API response
  let pending = null;  // AbortController of the request in flight

  const savedSite = () => { try { return localStorage.getItem("nx-site") || ""; } catch { return ""; } };
  const saveSite = (v) => { try { localStorage.setItem("nx-site", v); } catch { /* private mode */ } };

  function gauge(label, value, threshold, thresholdLabel, members, caption) {
    const v = Math.min(Math.max(value, 0), 1), t = Math.min(Math.max(threshold, 0), 1);
    const ticks = Object.entries(members || {}).map(([n, p]) => `<span class="member" style="left:${(p * 100).toFixed(2)}%" title="${esc(n)} ${num(p)}"></span>`).join("");
    return `<div class="gauge">
      <div class="gauge-head"><p class="kicker" style="margin:0">${esc(label)}</p><span class="value">${num(v)}</span></div>
      <div class="track" role="img" aria-label="${esc(label)} ${num(v)}, threshold ${num(t, 3)}">
        <span class="zone" style="left:${(t * 100).toFixed(2)}%"></span>${ticks}
        <span class="thr" style="left:${(t * 100).toFixed(2)}%"></span>
        <span class="dot${value >= threshold ? " over" : ""}" style="left:${(v * 100).toFixed(2)}%"></span>
      </div>
      <div class="scale"><span>0</span><span>${esc(thresholdLabel)} <b>${num(t, 3)}</b></span><span>1</span></div>
      ${caption ? `<p class="note" style="margin:.4rem 0 0">${esc(caption)}</p>` : ""}
    </div>`;
  }

  function render() {
    if (!last) return;
    const r = last;
    const siteName = $("site").value || null;
    const d = r.decisions.find((x) => x.site === siteName) || r.decisions[0];
    const where = d.site ? `${d.site} threshold` : "global threshold";
    let verdict;
    if (!r.gradable) {
      verdict = `<div class="verdict v-retake" role="status">${icon("retake", "#1D2140")}<p class="kicker">Not graded</p><h2>Retake the photograph</h2><p>${esc(d.advice)}</p><ul>${r.reasons.map((x) => `<li>${esc(x)}</li>`).join("")}</ul></div>`;
    } else if (d.referable) {
      const urgent = r.grade != null && r.grade >= 3;
      verdict = `<div class="verdict v-refer" role="status">${icon("refer", "#9A4426")}<p class="kicker">Refer${urgent ? " · urgent" : ""}</p><h2>Referable diabetic retinopathy suspected</h2><p>${esc(d.advice)}</p></div>`;
    } else {
      verdict = `<div class="verdict v-clear" role="status">${icon("clear", "#2B5C99")}<p class="kicker">No referral</p><h2>No referable diabetic retinopathy</h2><p>${esc(d.advice)}</p></div>`;
    }
    const parts = [verdict];
    if (r.referable_score != null) {
      const members = r.member_scores && Object.keys(r.member_scores).length > 1 ? r.member_scores : null;
      let caption = "Dot: the ensemble's score. Ticks above the track: each network's own score. Hatched: the referral zone.";
      if (!r.gradable) caption = "Shown only because grading of refused photographs is switched on. Do not act on this score.";
      else if (d.site && d.expected_specificity != null) caption += ` At this site the threshold is expected to give ${pct(d.expected_sensitivity, 0)} sensitivity and ${pct(d.expected_specificity, 0)} specificity.`;
      parts.push(gauge("Referable-DR score", r.referable_score, d.threshold, `Referral at ${where}`, members, caption));
    }
    if (r.ungradable_score != null) parts.push(gauge("Ungradable score", r.ungradable_score, r.quality_threshold, "Refused at", null, "From the quality gate, which sees the photograph before any enhancement."));
    const low = r.referable_score != null && r.referable_score < d.threshold;
    const figs = [[r.images.cropped, "Cropped", "to the retina; what the quality gate sees."]];
    if (r.images.enhanced) figs.push([r.images.enhanced, "Enhanced", "illumination evened and contrast equalised; what the networks read."]);
    if (r.images.heatmap) figs.push([r.images.heatmap, "Heatmap", "regions that raised the score most; outline at half the peak." + (low ? " At a score this low these are the least normal-looking areas, not lesions." : "")]);
    parts.push(`<div class="figs">${figs.map(([src, t, c]) => `<figure><img src="${src}" alt="${esc(t)}"><figcaption><b>${esc(t)}</b> ${esc(c)}</figcaption></figure>`).join("")}</div>`);
    if (r.grade_probs) {
      const top = r.grade_probs.indexOf(Math.max(...r.grade_probs));
      parts.push(`<p class="kicker" style="margin-top:1.4rem">Estimated grade (secondary to the referral decision)</p><div class="grades">${r.grade_probs.map((p, i) => `<span class="${i === top ? "top" : ""}">${GRADES[i]}</span><i class="${i === top ? "top" : ""}" style="width:${Math.max(p * 100, 0.4).toFixed(1)}%"></i><em>${(p * 100).toFixed(0)}%</em>`).join("")}</div>`);
    }
    parts.push(`<p class="note" style="margin-top:1.2rem">Screened in ${num(r.seconds, 1)} s · model ${esc(r.model || "")}</p>`);
    $("report").innerHTML = `<div class="reveal">${parts.join("")}</div>`;
    $("print-meta").textContent = `${new Date().toLocaleString()} · ${source?.label || source?.file?.name || ""} · ${d.label}`;
  }

  function fillSites(decisions) {
    const sel = $("site"), want = sel.value || savedSite();
    sel.innerHTML = decisions.map((d) => `<option value="${esc(d.site || "")}">${esc(d.label)}</option>`).join("");
    if ([...sel.options].some((o) => o.value === want)) sel.value = want;
  }

  async function screen() {
    if (!source) return;
    pending?.abort();
    pending = new AbortController();
    const form = new FormData();
    if (source.file) form.append("photo", source.file);
    else form.append("sample_name", source.sample);
    form.append("degrade_kind", $("kind").value);
    form.append("degrade_severity", $("severity").value);
    form.append("grade_rejected", $("grade-rejected").checked ? "true" : "false");
    $("report").innerHTML = '<div class="loading"><span class="spinner"></span>Screening: quality check, three networks and their heatmaps ...</div>';
    $("print").disabled = true;
    try {
      last = await json("/api/screen", { method: "POST", body: form, signal: pending.signal });
      fillSites(last.decisions);
      render();
      $("print").disabled = false;
    } catch (err) {
      if (err.name === "AbortError") return;
      $("report").innerHTML = `<p class="error" role="alert">${esc(err.message)}</p>`;
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    const drop = $("drop"), input = $("photo");
    const choose = (file) => {
      if (!file) return;
      source = { file };
      document.querySelectorAll("[data-sample]").forEach((b) => b.setAttribute("aria-pressed", "false"));
      drop.querySelector("strong").textContent = file.name;
      screen();
    };
    input.addEventListener("change", () => choose(input.files[0]));
    drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
    drop.tabIndex = 0;
    ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("is-over"); }));
    ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("is-over"); }));
    drop.addEventListener("drop", (e) => choose(e.dataTransfer.files[0]));
    document.querySelectorAll("[data-sample]").forEach((b) => b.addEventListener("click", () => {
      source = { sample: b.dataset.sample, label: b.getAttribute("aria-label") };
      document.querySelectorAll("[data-sample]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      drop.querySelector("strong").textContent = "Drop a photograph here";
      screen();
    }));
    const kind = $("kind"), sev = $("severity");
    kind.addEventListener("change", () => { sev.disabled = kind.value === "none"; screen(); });
    sev.addEventListener("input", () => { $("severity-out").textContent = Number(sev.value).toFixed(2); });
    sev.addEventListener("change", screen);
    $("grade-rejected").addEventListener("change", screen);
    $("site").addEventListener("change", () => { saveSite($("site").value); render(); });
    $("print").addEventListener("click", () => window.print());
  });
})();
