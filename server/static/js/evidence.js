// Test-set tabs with arrow-key navigation (WAI-ARIA tabs pattern).
"use strict";
document.addEventListener("DOMContentLoaded", () => {
  const tabs = [...document.querySelectorAll('[role="tab"]')];
  const select = (tab) => {
    tabs.forEach((t) => {
      const on = t === tab;
      t.setAttribute("aria-selected", String(on));
      t.tabIndex = on ? 0 : -1;
      document.getElementById(t.getAttribute("aria-controls")).hidden = !on;
    });
  };
  tabs.forEach((t, i) => {
    t.tabIndex = i === 0 ? 0 : -1;
    t.addEventListener("click", () => select(t));
    t.addEventListener("keydown", (e) => {
      const d = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
      if (d) { const next = tabs[(i + d + tabs.length) % tabs.length]; select(next); next.focus(); }
    });
  });
});
