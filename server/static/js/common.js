// Shared behaviour: mobile menu, header rule on scroll, print buttons, guide contents highlighting.
"use strict";

window.NX = {
  esc(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  },
  pct(v, digits = 1) { return v == null ? "–" : `${(v * 100).toFixed(digits)}%`; },
  num(v, digits = 2) { return v == null ? "–" : Number(v).toFixed(digits); },
  async json(url, options = {}) {
    const res = await fetch(url, options);
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.detail ? (typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)) : `${res.status} ${res.statusText}`);
    return body;
  },
};

document.addEventListener("DOMContentLoaded", () => {
  const header = document.querySelector(".site-header");
  const onScroll = () => header?.classList.toggle("is-scrolled", window.scrollY > 8);
  onScroll();
  window.addEventListener("scroll", onScroll, { passive: true });

  const toggle = document.querySelector(".menu-toggle");
  const nav = document.getElementById("nav");
  toggle?.addEventListener("click", () => {
    const open = nav.classList.toggle("is-open");
    toggle.setAttribute("aria-expanded", String(open));
  });

  document.querySelectorAll("[data-print]").forEach((b) => b.addEventListener("click", () => window.print()));

  // Highlight the guide section being read.
  const links = [...document.querySelectorAll(".toc a")];
  if (links.length && "IntersectionObserver" in window) {
    const byId = new Map(links.map((a) => [a.getAttribute("href").slice(1), a]));
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          links.forEach((a) => a.classList.remove("is-active"));
          byId.get(e.target.id)?.classList.add("is-active");
        }
      });
    }, { rootMargin: "-20% 0px -70% 0px" });
    byId.forEach((_, id) => { const el = document.getElementById(id); if (el) io.observe(el); });
  }
});
