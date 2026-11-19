// Blinded grading: fetch the next photograph, collect the grade, save, repeat.
"use strict";
(() => {
  const { esc, pct, num, json } = window.NX;
  const $ = (id) => document.getElementById(id);
  const GRADES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"];
  let who = "", code = "", mode = "blinded", current = null, shownAt = 0, objectUrl = null, busy = false;
  const skipped = new Set();

  const auth = () => ({ Authorization: `Bearer ${code}` });
  const store = (k, v) => { try { sessionStorage.setItem(k, v); } catch { /* storage blocked */ } };
  const recall = (k) => { try { return sessionStorage.getItem(k) || ""; } catch { return ""; } };

  async function loadImage(id) {
    const res = await fetch(`/api/review/image/${encodeURIComponent(id)}`, { headers: auth() });
    if (!res.ok) throw new Error(`Could not load the photograph (${res.status}).`);
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(await res.blob());
    $("photo").src = objectUrl;
  }

  function resetForm() {
    $("g-yes").checked = true;
    document.querySelectorAll('input[name="grade"]').forEach((r) => { r.checked = false; });
    $("other").checked = false;
    $("c-medium").checked = true;
    $("note").value = "";
    $("grades").classList.remove("is-disabled");
    $("reveal").innerHTML = "";
    $("save").disabled = false;
    $("status").textContent = "";
    setZoom(false);
  }

  async function next() {
    const n = await json(`/api/review/next?reviewer=${encodeURIComponent(who)}&skip=${encodeURIComponent([...skipped].join(","))}`, { headers: auth() });
    $("progress").style.width = `${(n.answered / n.total) * 100}%`;
    if (!n.item_id) {
      $("grader").hidden = true;
      $("done").hidden = false;
      $("done-text").textContent = `You have graded all ${n.total} photographs. You can close this page; your answers are saved.`;
      return;
    }
    current = n.item_id;
    $("counter").textContent = `${n.answered + 1} of ${n.total}`;
    resetForm();
    await loadImage(current);
    shownAt = performance.now();
    $("stage").focus({ preventScroll: true });  // keyboard grading starts at once, not on the hidden Start button
  }

  function setZoom(on, x = 50, y = 50) {
    const stage = $("stage"), img = $("photo");
    stage.classList.toggle("zoomed", on);
    img.style.transformOrigin = `${x}% ${y}%`;
    img.style.transform = on ? "scale(2.4)" : "";
  }

  async function reveal() {
    const m = await json(`/api/review/model/${encodeURIComponent(current)}?reviewer=${encodeURIComponent(who)}`, { headers: auth() });
    const verdict = !m.gradable ? "Refused as ungradable" : m.referable ? "Refer" : "No referral";
    $("reveal").innerHTML = `<div class="callout calm">
      <p class="kicker" style="margin:0 0 .3rem">Fundalux said</p>
      <p style="margin:0"><strong>${esc(verdict)}</strong>${m.gradable ? ` · score ${num(m.referable_score)} (${esc(m.threshold_label)} ${num(m.threshold, 3)}) · estimated ${esc(m.grade_name || "–")}` : ""}</p>
      ${m.heatmap ? `<img src="${m.heatmap}" alt="Fundalux heatmap" style="margin-top:.7rem;border-radius:6px;max-width:260px">` : ""}
      <p style="margin:.8rem 0 0"><button type="button" class="btn" id="continue">Next photograph</button></p></div>`;
    $("continue").addEventListener("click", () => next().catch(fail));
    $("continue").focus();
  }

  function fail(err) { $("status").textContent = err.message; $("save").disabled = false; busy = false; }

  async function save(skip = false) {
    if (busy || !current) return;
    const gradable = $("g-yes").checked;
    const grade = document.querySelector('input[name="grade"]:checked');
    if (!skip && gradable && !grade) { $("status").textContent = "Choose a grade, or mark the photograph ungradable."; return; }
    busy = true;
    $("save").disabled = true;
    try {
      if (skip) skipped.add(current);
      else {
        skipped.delete(current);
        await json("/api/review/answer", {
          method: "POST", headers: { ...auth(), "Content-Type": "application/json" },
          body: JSON.stringify({
            reviewer: who, item_id: current, gradable, grade: gradable ? Number(grade.value) : null, other_referral: $("other").checked,
            confidence: document.querySelector('input[name="confidence"]:checked')?.value || null, note: $("note").value.trim(),
            seconds: Math.round((performance.now() - shownAt) / 100) / 10, mode,
          }),
        });
      }
      if (mode === "teaching" && !skip) await reveal();
      else await next();
    } catch (err) { fail(err); return; }
    busy = false;
  }

  document.addEventListener("DOMContentLoaded", () => {
    if (!$("start-form")) return;
    $("stage").tabIndex = 0;
    $("stage").setAttribute("aria-label", "Photograph; press Z to zoom");
    $("reviewer").value = recall("nx-reviewer");
    $("code").value = recall("nx-review-code");
    $("start-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      who = $("reviewer").value.trim(); code = $("code").value.trim();
      mode = document.querySelector('input[name="mode"]:checked').value;
      if (!who || !code) { $("start-error").textContent = "Enter your name and the review code."; return; }
      try {
        store("nx-reviewer", who); store("nx-review-code", code);
        $("start").hidden = true; $("grader").hidden = false;
        await next();
        $("grader").scrollIntoView({ behavior: "smooth", block: "start" });
      } catch (err) {
        $("start").hidden = false; $("grader").hidden = true;
        $("start-error").textContent = err.message;
      }
    });
    document.querySelectorAll('input[name="gradable"]').forEach((r) => r.addEventListener("change", () => {
      $("grades").classList.toggle("is-disabled", !$("g-yes").checked);
    }));
    $("grade-form").addEventListener("submit", (e) => { e.preventDefault(); save(); });
    $("skip").addEventListener("click", () => save(true));
    $("stage").addEventListener("click", (e) => {
      const r = e.currentTarget.getBoundingClientRect();
      setZoom(!$("stage").classList.contains("zoomed"), ((e.clientX - r.left) / r.width) * 100, ((e.clientY - r.top) / r.height) * 100);
    });
    document.addEventListener("keydown", (e) => {
      if ($("grader").hidden || e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT" && e.target.type === "text") return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (/^[0-4]$/.test(e.key)) { $("g-yes").checked = true; $("grades").classList.remove("is-disabled"); $(`grade-${e.key}`).checked = true; }
      else if (e.key === "u" || e.key === "U") { $("g-no").checked = true; $("grades").classList.add("is-disabled"); }
      else if (e.key === "r" || e.key === "R") { $("other").checked = !$("other").checked; }
      else if (e.key === "z" || e.key === "Z") { setZoom(!$("stage").classList.contains("zoomed")); }
      else if (e.key === "Enter" && !(e.target.tagName === "BUTTON" && $("grade-form").contains(e.target))) { e.preventDefault(); if ($("continue")) $("continue").click(); else save(); }
    });
  });
})();
