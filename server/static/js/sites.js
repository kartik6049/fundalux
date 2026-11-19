// Site calibration: upload graded photographs, follow the job, save or discard; remove own sites.
"use strict";
(() => {
  const { esc, pct, num, json } = window.NX;
  const $ = (id) => document.getElementById(id);
  const token = () => $("token").value.trim();
  const auth = () => ({ Authorization: `Bearer ${token()}` });
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  async function follow(jobId) {
    const box = $("job");
    for (;;) {
      const j = await json(`/api/jobs/${jobId}`, { headers: auth() });
      if (j.status === "failed") { box.innerHTML = `<p class="error" role="alert">Not calibrated: ${esc(j.error)}</p>`; return; }
      if (j.status === "done") {
        const p = j.profile, s = j.summary;
        box.innerHTML = `<div class="callout calm">
          <p class="kicker" style="margin:0 0 .4rem">${esc(p.name)}</p>
          <p style="margin:0"><b style="font:500 1.8rem var(--serif)">${num(p.threshold, 3)}</b> threshold, fitted on ${p.n_images} gradable photographs (${p.n_referable} referable).</p>
          <p style="margin:.4rem 0 0">Expected: <b>${pct(p.expected_sensitivity)}</b> sensitivity, <b>${pct(p.expected_specificity)}</b> specificity.</p>
          <p class="note" style="margin:.5rem 0 0">${s.matched} of ${s.labelled} labelled photographs found; ${s.rejected_by_gate} refused by the quality gate. ${esc(p.estimate)}.</p>
          ${(p.warnings || []).map((w) => `<p class="fail" style="margin:.4rem 0 0">${esc(w)}</p>`).join("")}
          <div style="display:flex;gap:.6rem;margin-top:1rem"><button type="button" class="btn" id="save">Save site</button><button type="button" class="btn btn-quiet" id="discard">Discard</button></div>
        </div>`;
        $("save").addEventListener("click", async () => {
          try { await json(`/api/jobs/${jobId}/save`, { method: "POST", headers: auth() }); location.reload(); }
          catch (err) { box.insertAdjacentHTML("beforeend", `<p class="error">${esc(err.message)}</p>`); }
        });
        $("discard").addEventListener("click", () => { box.innerHTML = ""; });
        return;
      }
      box.innerHTML = `<div class="loading" style="padding:.4rem 0"><span class="spinner"></span>Scoring ${j.done} of ${j.total} photographs ...</div>`;
      await sleep(1500);
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    try { $("token").value = sessionStorage.getItem("nx-token") || ""; } catch { /* storage blocked */ }
    const showAdmin = () => document.querySelectorAll(".admin-only").forEach((b) => { b.hidden = !token(); });
    $("token").addEventListener("input", () => { try { sessionStorage.setItem("nx-token", token()); } catch { /* storage blocked */ } showAdmin(); });
    showAdmin();
    $("target").addEventListener("input", (e) => { $("target-out").textContent = Number(e.target.value).toFixed(2); });

    $("calibrate").addEventListener("submit", async (e) => {
      e.preventDefault();
      const box = $("job"), labels = $("labels").files[0], photos = [...$("photos").files];
      if (!token() || !$("name").value.trim() || !labels || !photos.length) { box.innerHTML = '<p class="error">Fill in the token, a site name, the labels file and the photographs.</p>'; return; }
      const form = new FormData();
      form.append("name", $("name").value.trim());
      form.append("target_sensitivity", $("target").value);
      form.append("labels", labels);
      photos.forEach((f) => form.append("photos", f));
      box.innerHTML = `<div class="loading" style="padding:.4rem 0"><span class="spinner"></span>Uploading ${photos.length} photographs ...</div>`;
      try {
        const res = await json("/api/sites/calibrate", { method: "POST", body: form, headers: auth() });
        await follow(res.job);
      } catch (err) { box.innerHTML = `<p class="error" role="alert">${esc(err.message)}</p>`; }
    });

    document.querySelectorAll("[data-remove]").forEach((b) => b.addEventListener("click", async () => {
      const name = b.dataset.remove;
      b.disabled = true;
      try { await json(`/api/sites/${encodeURIComponent(name)}`, { method: "DELETE", headers: auth() }); location.reload(); }
      catch (err) { b.disabled = false; b.insertAdjacentHTML("afterend", `<span class="error" style="display:block;font-size:.8rem">${esc(err.message)}</span>`); }
    }));
  });
})();
