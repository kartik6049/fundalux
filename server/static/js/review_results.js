// Organiser view: agreement tables, disagreement gallery, CSV export, new review sets.
"use strict";
(() => {
  const { esc, pct, num, json } = window.NX;
  const $ = (id) => document.getElementById(id);
  const token = () => $("token").value.trim();
  const auth = () => ({ Authorization: `Bearer ${token()}` });
  const kappa = (k) => (k == null ? "–" : num(k, 2));
  const urls = [];

  const table = (head, rows) => `<div class="scroll"><table class="table"><thead><tr>${head.map(([h, r]) => `<th class="${r ? "r" : ""}">${esc(h)}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr>${row.map((c, i) => `<td class="${head[i][1] ? "r" : ""}">${c}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;

  async function thumbs() {
    for (const img of document.querySelectorAll("img[data-item]")) {
      const res = await fetch(`/api/review/thumb/${encodeURIComponent(img.dataset.item)}`, { headers: auth() });
      if (res.ok) { const u = URL.createObjectURL(await res.blob()); urls.push(u); img.src = u; }
    }
  }

  function render(d) {
    $("set-info").textContent = `Set “${d.set.name || "unnamed"}”, created ${d.set.created || "–"}, ${d.items} photographs · Fundalux at the ${d.threshold_label} (${num(d.threshold, 3)})${d.stale_model ? " · warning: the set was scored by a different model than the one now loaded" : ""}`;
    const parts = [];
    if (!d.reviewers.length) {
      parts.push('<p class="callout calm" style="margin-top:1.4rem">No answers yet. Share the <a href="/review">review page</a> and the review code with clinicians.</p>');
    } else {
      parts.push('<h2 style="margin:2rem 0 .8rem">Reviewers</h2>' + table(
        [["Reviewer"], ["Graded", 1], ["Ungradable", 1], ["Referred for other reasons", 1], ["Median seconds", 1], ["In teaching mode", 1]],
        d.reviewers.map((r) => [esc(r.name), `${r.answered} of ${r.of}`, r.ungradable, r.other_referral, r.median_seconds == null ? "–" : num(r.median_seconds, 0), r.teaching_mode])));
      parts.push('<h2 style="margin:2.4rem 0 .4rem">Agreement on referral</h2><p class="note">Referable = ICDR moderate NPDR or worse. Only photographs both raters called gradable. Kappa: 1 perfect, 0 chance.</p>' + table(
        [["Rater A"], ["Rater B"], ["Photographs", 1], ["Agree", 1], ["Kappa", 1], ["Both refer", 1], ["Only A refers", 1], ["Only B refers", 1]],
        d.pairs.map((p) => [esc(p.a), esc(p.b), p.n, pct(p.agreement, 0), kappa(p.kappa), p.table[1][1], p.table[1][0], p.table[0][1]])));
      if (d.grade_kappa.length) parts.push('<h3 style="margin:1.8rem 0 .4rem">Agreement on the five-step grade</h3>' + table(
        [["Rater A"], ["Rater B"], ["Photographs", 1], ["Quadratic-weighted kappa", 1]], d.grade_kappa.map((k) => [esc(k.a), esc(k.b), k.n, kappa(k.kappa)])));
      if (d.versus_reference.length) parts.push('<h2 style="margin:2.4rem 0 .4rem">Against the reference grade</h2><p class="note">Each reviewer and Fundalux on exactly the same photographs: sensitivity · specificity for referral, and quadratic-weighted kappa for the grade. Small sets give wide uncertainty: one case moves these figures by several points.</p>' + table(
        [["Reviewer"], ["Photographs (referable)", 1], ["Reviewer", 1], ["Fundalux", 1], ["Grade kappa: reviewer · Fundalux", 1]],
        d.versus_reference.map((v) => [esc(v.reviewer), `${v.doctor.n} (${v.doctor.referable})`, `${pct(v.doctor.sensitivity, 0)} · ${pct(v.doctor.specificity, 0)}`, `${pct(v.model.sensitivity, 0)} · ${pct(v.model.specificity, 0)}`, `${kappa(v.doctor_grade_kappa)} · ${kappa(v.model_grade_kappa)}`])));
      const g = d.gradability;
      parts.push(`<h2 style="margin:2.4rem 0 .4rem">Image quality</h2><p class="note">Does the quality gate agree with clinicians about what can be graded? (All answers pooled.)</p>${table(
        [[""], ["Fundalux: gradable", 1], ["Fundalux: refused", 1]],
        [["Clinician: gradable", g.both_gradable, g.gate_only_refused], ["Clinician: ungradable", g.doctor_only_ungradable, g.both_ungradable]])}`);
      parts.push(`<h2 style="margin:2.4rem 0 .4rem">Disagreements</h2><p class="note">Photographs where a reviewer and Fundalux disagreed on referral; cases Fundalux would have missed come first.</p>${
        d.disagreements.length ? `<div class="gallery">${d.disagreements.map((x) => `<figure><img data-item="${esc(x.item_id)}" alt="Photograph ${esc(x.item_id)}"><figcaption><span class="kind">${esc(x.kind)}</span>${esc(x.item_id)} · ${esc(x.reviewer)}: ${esc(x.doctor_grade)}<br>Fundalux: score ${num(x.model_score)}, ${esc(x.model_grade || "–")}${x.reference_grade ? `<br>Reference: ${esc(x.reference_grade)}` : ""}${x.note ? `<br><em>“${esc(x.note)}”</em>` : ""}</figcaption></figure>`).join("")}</div>` : '<p>None.</p>'}`);
      if (d.notes.length) parts.push('<h2 style="margin:2.4rem 0 .4rem">Notes from reviewers</h2>' + table([["Photograph"], ["Reviewer"], ["Note"]], d.notes.map((n) => [esc(n.item_id), esc(n.reviewer), esc(n.note)])));
    }
    urls.splice(0).forEach((u) => URL.revokeObjectURL(u));
    $("results-body").innerHTML = parts.join("");
    thumbs();
  }

  let first = true;
  async function load() {
    // The first view uses the set's own site; after that, whatever the menu says ("" = global).
    const query = first ? "" : `?site=${encodeURIComponent($("site").value)}`;
    const d = await json(`/api/review/results${query}`, { headers: auth() });
    if (first && d.set.site && [...$("site").options].some((o) => o.value === d.set.site)) $("site").value = d.set.site;
    first = false;
    render(d);
  }

  async function followSet(jobId) {
    const box = $("set-job");
    for (;;) {
      const j = await json(`/api/jobs/${jobId}`, { headers: auth() });
      if (j.status === "failed") { box.innerHTML = `<p class="error">${esc(j.error)}</p>`; return; }
      if (j.status === "done") { box.innerHTML = `<p class="pass">Review set created with ${j.items} photographs.</p>`; load(); return; }
      box.innerHTML = `<div class="loading" style="padding:.3rem 0"><span class="spinner"></span>Scoring ${j.done} of ${j.total} photographs ...</div>`;
      await new Promise((r) => setTimeout(r, 1500));
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    try { $("token").value = sessionStorage.getItem("nx-token") || ""; } catch { /* storage blocked */ }
    $("auth").addEventListener("submit", async (e) => {
      e.preventDefault();
      try {
        try { sessionStorage.setItem("nx-token", token()); } catch { /* storage blocked */ }
        $("results").hidden = false; $("new-set").hidden = false; $("auth").hidden = true;
        await load();
      } catch (err) {
        $("results").hidden = true; $("auth").hidden = false;
        if (/No review set/.test(err.message)) { $("new-set").hidden = false; $("auth-error").textContent = "There is no review set yet: create one below."; }
        else { $("new-set").hidden = true; $("auth-error").textContent = err.message; }
      }
    });
    $("site").addEventListener("change", () => load().catch((err) => { $("results-body").innerHTML = `<p class="error">${esc(err.message)}</p>`; }));
    $("export").addEventListener("click", async () => {
      const res = await fetch(`/api/review/export.csv?site=${encodeURIComponent($("site").value)}`, { headers: auth() });
      const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(await res.blob()), download: "fundalux_review.csv" });
      a.click();
    });
    $("set-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      const photos = [...$("set-photos").files], box = $("set-job");
      if (!$("set-name").value.trim() || !photos.length) { box.innerHTML = '<p class="error">Give the set a name and choose photographs.</p>'; return; }
      const form = new FormData();
      form.append("name", $("set-name").value.trim());
      form.append("site", $("set-site").value);
      photos.forEach((f) => form.append("photos", f));
      if ($("set-labels").files[0]) form.append("labels", $("set-labels").files[0]);
      box.innerHTML = `<div class="loading" style="padding:.3rem 0"><span class="spinner"></span>Uploading ${photos.length} photographs ...</div>`;
      try { const r = await json("/api/review/set", { method: "POST", body: form, headers: auth() }); await followSet(r.job); }
      catch (err) { box.innerHTML = `<p class="error">${esc(err.message)}</p>`; }
    });
  });
})();
