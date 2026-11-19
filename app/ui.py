"""Visual language of the Fundalux app: tokens, global styles and HTML components.

Direction: an ophthalmic screening report. Lavender paper, indigo ink, powder
blue for the instrument (scales, heatmaps, calm results) and warm sunset peach
reserved for what needs attention (referrals). Newsreader sets the headings and
verdicts like a printed report; IBM Plex Sans carries the interface and every
number, with tabular figures.

Every function returns an HTML string for st.html; dynamic text is escaped here.
"""

from __future__ import annotations

import base64
import html

import cv2
import numpy as np

TOKENS = {
    "paper": "#E9E8F8",  # soft lavender white
    "raised": "#F6F5FC",
    "sunk": "#E1E0F3",
    "ink": "#1D2140",
    "muted": "#545A7A",
    "line": "#CFCDEA",
    "powder": "#8DB9F0",  # powder blue
    "powder-deep": "#2B5C99",
    "powder-wash": "#D6E5FA",
    "peach": "#F5B69E",  # warm sunset peach
    "peach-deep": "#9A4426",
    "peach-wash": "#FBE3D9",
}

_ROOT = "\n".join(f"  --nx-{k}: {v};" for k, v in TOKENS.items())

GLOBAL_CSS = f"""
<style>
:root {{
{_ROOT}
  --nx-serif: "Newsreader", Georgia, serif;
  --nx-sans: "IBM Plex Sans", system-ui, sans-serif;
  --nx-ease: cubic-bezier(.2, .7, .2, 1);
}}

/* Paper with a faint ophthalmoscope reticle behind the masthead. */
.stApp {{
  background-color: var(--nx-paper);
  background-image:
    radial-gradient(circle at 92% -4%, transparent 0 118px, rgba(141,185,240,.20) 119px 120px, transparent 121px 178px,
                    rgba(141,185,240,.16) 179px 180px, transparent 181px 248px, rgba(141,185,240,.12) 249px 250px,
                    transparent 251px 330px, rgba(141,185,240,.09) 331px 332px, transparent 333px),
    radial-gradient(ellipse 60% 40% at 0% 100%, rgba(245,182,158,.16), transparent 70%);
  background-attachment: fixed;
}}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stMainBlockContainer"] {{ max-width: 1280px; padding-top: 2.2rem; }}
.stApp, .stApp p, .stApp li, .stApp label {{ font-variant-numeric: tabular-nums; }}

/* Tabs: a typographic index rather than buttons. */
[data-testid="stTabs"] [role="tablist"] {{ gap: 2rem; border-bottom: 1px solid var(--nx-line); }}
[data-testid="stTabs"] button[role="tab"] {{ padding: .6rem 0; background: transparent; }}
[data-testid="stTabs"] button[role="tab"] p {{ font-size: .95rem; font-weight: 500; color: var(--nx-muted); transition: color .15s; }}
[data-testid="stTabs"] button[role="tab"]:hover p, [data-testid="stTabs"] button[aria-selected="true"] p {{ color: var(--nx-ink); }}
[data-baseweb="tab-highlight"] {{ background: var(--nx-powder-deep); height: 2px; }}
[data-baseweb="tab-border"] {{ display: none; }}

/* Keyed containers become panels: raised paper, one hairline, no stacked cards. */
.st-key-intake, .st-key-calibrate, .st-key-camp-controls {{
  background: var(--nx-raised); border: 1px solid var(--nx-line); border-radius: 8px; padding: 1.1rem 1.2rem 1.2rem;
}}
.st-key-site-picker [data-testid="stSelectbox"] label p {{ font-size: .72rem; letter-spacing: .08em; text-transform: uppercase; color: var(--nx-muted); }}

/* Typography */
.nx-kicker {{ font: 500 .72rem/1.2 var(--nx-sans); letter-spacing: .09em; text-transform: uppercase; color: var(--nx-muted); margin: 0 0 .35rem; }}
.nx-step {{ display: flex; gap: .55rem; align-items: baseline; margin: .2rem 0 .5rem; }}
.nx-step b {{ font: 500 1.05rem/1 var(--nx-serif); color: var(--nx-powder-deep); }}
.nx-step span {{ font: 500 .95rem/1.3 var(--nx-sans); color: var(--nx-ink); }}
.nx-lede {{ font: 400 1.02rem/1.55 var(--nx-sans); color: var(--nx-muted); max-width: 62ch; margin: 0; }}
.nx-note {{ font: 400 .82rem/1.5 var(--nx-sans); color: var(--nx-muted); max-width: 70ch; }}
.nx-h2 {{ font: 500 1.6rem/1.2 var(--nx-serif); color: var(--nx-ink); margin: 1.6rem 0 .3rem; }}
.nx-h3 {{ font: 500 1.15rem/1.3 var(--nx-serif); color: var(--nx-ink); margin: 1.2rem 0 .4rem; }}

/* Masthead */
.nx-mast {{ display: flex; gap: 1.1rem; align-items: center; }}
.nx-mast img {{ flex: none; }}
.nx-word {{ font: 500 2.6rem/1 var(--nx-serif); color: var(--nx-ink); letter-spacing: -.01em; margin: 0; }}
.nx-word em {{ font-style: normal; color: var(--nx-powder-deep); }}
.nx-meta {{ display: flex; flex-wrap: wrap; gap: .25rem 1.1rem; margin-top: .55rem; font: 400 .8rem/1.4 var(--nx-sans); color: var(--nx-muted); }}
.nx-meta b {{ font-weight: 500; color: var(--nx-ink); }}

/* Verdict slip */
.nx-verdict {{ display: grid; grid-template-columns: auto 1fr; gap: .3rem 1rem; padding: 1.1rem 1.25rem; border-radius: 8px; border-left: 4px solid; }}
.nx-verdict .nx-kicker {{ grid-column: 2; margin: 0; }}
.nx-verdict > img {{ grid-row: 1 / span 3; margin-top: .15rem; }}
.nx-verdict h3 {{ grid-column: 2; font: 500 1.75rem/1.2 var(--nx-serif); margin: 0; color: var(--nx-ink); }}
.nx-verdict p {{ grid-column: 2; margin: .15rem 0 0; font: 400 .95rem/1.5 var(--nx-sans); color: var(--nx-ink); }}
.nx-refer {{ background: var(--nx-peach-wash); border-color: var(--nx-peach-deep); }}
.nx-refer .nx-kicker {{ color: var(--nx-peach-deep); }}
.nx-clear {{ background: var(--nx-powder-wash); border-color: var(--nx-powder-deep); }}
.nx-clear .nx-kicker {{ color: var(--nx-powder-deep); }}
.nx-retake {{ background: var(--nx-raised); border: 1px dashed var(--nx-muted); border-left: 4px solid var(--nx-ink); }}
.nx-reasons {{ grid-column: 2; margin: .4rem 0 0; padding-left: 1.1rem; font: 400 .9rem/1.5 var(--nx-sans); color: var(--nx-ink); }}

/* Score instrument */
.nx-gauge {{ margin: 1.3rem 0 .2rem; }}
.nx-gauge-head {{ display: flex; justify-content: space-between; align-items: baseline; gap: 1rem; }}
.nx-gauge-head .nx-value {{ font: 600 1.6rem/1 var(--nx-sans); color: var(--nx-ink); font-variant-numeric: tabular-nums; }}
.nx-track {{ position: relative; height: 10px; margin: .55rem 0 .35rem; background: var(--nx-sunk); border-radius: 5px; }}
.nx-zone {{ position: absolute; top: 0; bottom: 0; right: 0; background: repeating-linear-gradient(135deg, rgba(245,182,158,.55) 0 4px, rgba(245,182,158,.25) 4px 8px); border-radius: 0 5px 5px 0; }}
.nx-thr {{ position: absolute; top: -5px; bottom: -5px; width: 2px; margin-left: -1px; background: var(--nx-ink); }}
.nx-dot {{ position: absolute; top: 50%; width: 16px; height: 16px; margin: -8px 0 0 -8px; border-radius: 50%; background: var(--nx-raised); border: 3px solid var(--nx-powder-deep); animation: nx-slide .7s var(--nx-ease) both; }}
.nx-dot.is-over {{ border-color: var(--nx-peach-deep); }}
.nx-member {{ position: absolute; top: -9px; width: 2px; height: 6px; margin-left: -1px; background: var(--nx-powder-deep); opacity: .55; }}
.nx-scale {{ display: flex; justify-content: space-between; font: 400 .75rem/1.3 var(--nx-sans); color: var(--nx-muted); }}
.nx-scale b {{ font-weight: 500; color: var(--nx-ink); }}

/* Grade distribution */
.nx-grades {{ display: grid; grid-template-columns: max-content 1fr 3.2rem; gap: .35rem .8rem; align-items: center; margin-top: .4rem; max-width: 34rem; }}
.nx-grades span {{ font: 400 .85rem/1.2 var(--nx-sans); color: var(--nx-muted); }}
.nx-grades .is-top {{ color: var(--nx-ink); font-weight: 500; }}
.nx-grades i {{ display: block; height: 6px; background: var(--nx-powder); border-radius: 3px; transform-origin: left; animation: nx-grow .6s var(--nx-ease) both; }}
.nx-grades i.is-top {{ background: var(--nx-powder-deep); }}
.nx-grades em {{ font: 500 .85rem/1 var(--nx-sans); font-style: normal; text-align: right; color: var(--nx-ink); }}

/* Figures */
.nx-figs {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .9rem; margin: 1.3rem 0 .4rem; }}
.nx-figs figure {{ margin: 0; }}
.nx-figs img {{ width: 100%; display: block; border-radius: 6px; background: #000; }}
.nx-figs figcaption {{ margin-top: .45rem; font: 400 .78rem/1.4 var(--nx-sans); color: var(--nx-muted); }}
.nx-figs figcaption b {{ font-weight: 500; color: var(--nx-ink); }}

/* Tables */
.nx-table {{ width: 100%; border-collapse: collapse; font: 400 .88rem/1.4 var(--nx-sans); font-variant-numeric: tabular-nums; }}
.nx-table th {{ text-align: left; font: 500 .7rem/1.2 var(--nx-sans); letter-spacing: .08em; text-transform: uppercase; color: var(--nx-muted); padding: .5rem .6rem; border-bottom: 1px solid var(--nx-ink); }}
.nx-table td {{ padding: .55rem .6rem; border-bottom: 1px solid var(--nx-line); color: var(--nx-ink); vertical-align: top; }}
.nx-table .num {{ text-align: right; white-space: nowrap; }}
.nx-table tbody tr {{ transition: background .15s; }}
.nx-table tbody tr:hover {{ background: rgba(141,185,240,.12); }}
.nx-table .is-selected td {{ background: rgba(141,185,240,.18); }}
.nx-scroll {{ overflow-x: auto; }}
.nx-pass, .nx-fail {{ font-weight: 500; white-space: nowrap; }}
.nx-pass {{ color: var(--nx-powder-deep); }}
.nx-fail {{ color: var(--nx-peach-deep); }}
.nx-heat td.num {{ text-align: center; }}

/* Headline numbers */
.nx-figures {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 1.4rem; margin: .8rem 0 1rem; }}
.nx-figure .nx-big {{ font: 500 2.4rem/1 var(--nx-serif); color: var(--nx-ink); font-variant-numeric: lining-nums tabular-nums; }}
.nx-figure .nx-big small {{ font: 400 .95rem/1 var(--nx-sans); color: var(--nx-muted); margin-left: .3rem; }}
.nx-figure p {{ margin: .35rem 0 0; font: 400 .82rem/1.45 var(--nx-sans); color: var(--nx-muted); }}

/* Empty state: an un-photographed fundus */
.nx-empty {{ display: grid; justify-items: start; gap: .9rem; padding: 2.4rem 0 1rem; }}
.nx-empty h3 {{ font: 500 1.5rem/1.2 var(--nx-serif); margin: 0; color: var(--nx-ink); }}

.nx-reveal > * {{ animation: nx-rise .4s var(--nx-ease) both; }}
.nx-reveal > *:nth-child(2) {{ animation-delay: .05s; }}
.nx-reveal > *:nth-child(3) {{ animation-delay: .1s; }}
.nx-reveal > *:nth-child(4) {{ animation-delay: .15s; }}
.nx-reveal > *:nth-child(n+5) {{ animation-delay: .2s; }}
@keyframes nx-rise {{ from {{ opacity: 0; transform: translateY(6px); }} to {{ opacity: 1; transform: none; }} }}
@keyframes nx-grow {{ from {{ transform: scaleX(0); }} to {{ transform: scaleX(1); }} }}
@keyframes nx-slide {{ from {{ left: 0; opacity: 0; }} }}
@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{ animation: none !important; transition: none !important; }}
}}
@media (max-width: 640px) {{
  .nx-word {{ font-size: 2rem; }}
  .nx-verdict h3 {{ font-size: 1.4rem; }}
  .nx-figs {{ grid-template-columns: 1fr; }}
  [data-testid="stTabs"] [role="tablist"] {{ gap: 1.1rem; overflow-x: auto; }}
}}
</style>
"""

e = html.escape


def pct(value, digits: int = 1) -> str:
    return "–" if value is None or (isinstance(value, float) and not np.isfinite(value)) else f"{value * 100:.{digits}f}%"


def num(value, digits: int = 3) -> str:
    return "–" if value is None or (isinstance(value, float) and not np.isfinite(value)) else f"{value:.{digits}f}"


# Icons: 20 px line drawings, colour from currentColor. No emoji.
_ICONS = {
    "refer": '<circle cx="10" cy="10" r="8.5"/><path d="M7 13l6-6M8 7h5v5"/>',
    "clear": '<circle cx="10" cy="10" r="8.5"/><path d="M6.5 10.2l2.4 2.4 4.6-5"/>',
    "retake": '<rect x="2.5" y="5.5" width="15" height="10.5" rx="1.5"/><path d="M7 5.5l1.2-2h3.6l1.2 2"/><circle cx="10" cy="10.7" r="2.8"/><path d="M3 17L17 3"/>',
}


def _svg(markup: str, width: int, height: int, alt: str = "") -> str:
    """st.html strips inline <svg>, so drawings are embedded as images."""
    data = base64.b64encode(markup.encode()).decode()
    return f'<img src="data:image/svg+xml;base64,{data}" width="{width}" height="{height}" alt="{e(alt)}"{"" if alt else ' aria-hidden="true"'}>'


def icon(name: str, color: str, size: int = 26) -> str:
    return _svg(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 20 20" fill="none" stroke="{color}" stroke-width="1.5" '
        f'stroke-linecap="round" stroke-linejoin="round">{_ICONS[name]}</svg>',
        size, size,
    )


def mark(size: int = 46) -> str:
    """The Fundalux mark: a fundus inside an instrument reticle, optic disc in peach."""
    t = TOKENS
    return _svg(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 48 48">'
        f'<circle cx="24" cy="24" r="22" fill="none" stroke="{t["powder-deep"]}" stroke-width="1.5"/>'
        f'<circle cx="24" cy="24" r="15" fill="{t["powder-wash"]}" stroke="{t["powder"]}" stroke-width="1.5"/>'
        f'<path d="M24 0v6M24 42v6M0 24h6M42 24h6" stroke="{t["powder-deep"]}" stroke-width="1.5"/>'
        f'<path d="M14.5 21c4-3 8-3.5 13-2M15 27.5c4.5 2 9 2 14 0" fill="none" stroke="{t["muted"]}" stroke-width="1" stroke-linecap="round"/>'
        f'<circle cx="30" cy="23" r="4" fill="{t["peach"]}" stroke="{t["peach-deep"]}" stroke-width="1"/></svg>',
        size, size,
    )


def masthead(meta: list[tuple[str, str]]) -> str:
    items = "".join(f"<span>{e(k)} <b>{e(v)}</b></span>" for k, v in meta)
    return (
        f'<div class="nx-mast">{mark()}<div><p class="nx-word">Funda<em>lux</em></p>'
        f'<p class="nx-lede">Screening for referable diabetic retinopathy that shows where it looked and refuses photographs it cannot read.</p>'
        f'<div class="nx-meta">{items}</div></div></div>'
    )


def step(n: int, text: str) -> str:
    return f'<div class="nx-step"><b>{n}</b><span>{e(text)}</span></div>'


def kicker(text: str) -> str:
    return f'<p class="nx-kicker">{e(text)}</p>'


def heading(text: str, level: int = 2) -> str:
    return f'<p class="nx-h{level}">{e(text)}</p>'


def note(text: str) -> str:
    return f'<p class="nx-note">{e(text)}</p>'


def lede(text: str) -> str:
    return f'<p class="nx-lede">{e(text)}</p>'


def verdict(state: str, label: str, title: str, body: str, reasons: list[str] | None = None) -> str:
    color = {"refer": TOKENS["peach-deep"], "clear": TOKENS["powder-deep"], "retake": TOKENS["ink"]}[state]
    items = "".join(f"<li>{e(r)}</li>" for r in reasons or [])
    return (
        f'<div class="nx-verdict nx-{state}" role="status">{icon(state, color)}'
        f"{kicker(label)}<h3>{e(title)}</h3><p>{e(body)}</p>"
        + (f'<ul class="nx-reasons">{items}</ul>' if items else "")
        + "</div>"
    )


def gauge(label: str, value: float, threshold: float, threshold_label: str, members: dict[str, float] | None = None, caption: str = "") -> str:
    """A 0-1 score on a track; the hatched zone right of the threshold is where a referral (or rejection) happens."""
    v, t = float(np.clip(value, 0, 1)), float(np.clip(threshold, 0, 1))
    ticks = "".join(f'<span class="nx-member" style="left:{p * 100:.2f}%" title="{e(name)} {p:.2f}"></span>' for name, p in (members or {}).items())
    over = " is-over" if value >= threshold else ""
    return (
        f'<div class="nx-gauge"><div class="nx-gauge-head">{kicker(label)}<span class="nx-value">{v:.2f}</span></div>'
        f'<div class="nx-track" role="img" aria-label="{e(label)} {v:.2f}, threshold {t:.2f}">'
        f'<span class="nx-zone" style="left:{t * 100:.2f}%"></span>'
        f'{ticks}<span class="nx-thr" style="left:{t * 100:.2f}%"></span><span class="nx-dot{over}" style="left:{v * 100:.2f}%"></span></div>'
        f'<div class="nx-scale"><span>0</span><span>{e(threshold_label)} <b>{t:.3f}</b></span><span>1</span></div>'
        + (f'<p class="nx-note">{e(caption)}</p>' if caption else "")
        + "</div>"
    )


def grade_bars(names: list[str], probs: list[float]) -> str:
    top = int(np.argmax(probs))
    rows = "".join(
        f'<span class="{"is-top" if i == top else ""}">{e(n)}</span>'
        f'<i class="{"is-top" if i == top else ""}" style="width:{max(p, .004) * 100:.1f}%"></i><em>{p * 100:.0f}%</em>'
        for i, (n, p) in enumerate(zip(names, probs))
    )
    return f'<div class="nx-grades">{rows}</div>'


def _data_uri(img: np.ndarray, quality: int = 88) -> str:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def figures(items: list[tuple[np.ndarray, str, str]]) -> str:
    """[(RGB image, bold title, caption)] as a responsive strip."""
    cells = "".join(
        f'<figure><img src="{_data_uri(img)}" alt="{e(title)}"><figcaption><b>{e(title)}</b> {e(text)}</figcaption></figure>' for img, title, text in items
    )
    return f'<div class="nx-figs">{cells}</div>'


def heat_overlay(img: np.ndarray, cam: np.ndarray) -> np.ndarray:
    from fundalux.heatmap import palette_overlay

    return palette_overlay(img, cam)


def table(columns: list[tuple[str, bool]], rows: list[list[str]], selected: int | None = None, cls: str = "") -> str:
    """columns: [(label, numeric)]. Cell strings are escaped unless they are already markup (start with '<')."""
    head = "".join(f'<th class="{"num" if n else ""}">{e(c)}</th>' for c, n in columns)
    body = ""
    for i, row in enumerate(rows):
        cells = "".join(f'<td class="{"num" if n else ""}">{v if str(v).startswith("<") else e(str(v))}</td>' for v, (_, n) in zip(row, columns))
        body += f'<tr class="{"is-selected" if i == selected else ""}">{cells}</tr>'
    return f'<div class="nx-scroll"><table class="nx-table {cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def verdict_cell(ok: bool | None, yes: str = "meets target", no: str = "below target") -> str:
    if ok is None:
        return "–"
    return f'<span class="nx-{"pass" if ok else "fail"}">{e(yes if ok else no)}</span>'


def heat_cell(value: float, low: float, high: float, text: str) -> str:
    """A table cell shaded from powder (good) to peach (bad)."""
    f = 0.0 if high <= low else float(np.clip((value - low) / (high - low), 0, 1))
    a = np.array([214, 229, 250]) * (1 - f) + np.array([251, 227, 217]) * f
    return f'<span style="display:block;margin:-.55rem -.6rem;padding:.55rem .6rem;background:rgb({a[0]:.0f},{a[1]:.0f},{a[2]:.0f})">{e(text)}</span>'


def headline_figures(items: list[tuple[str, str, str]]) -> str:
    """[(big value, unit, explanation)]"""
    cells = "".join(f'<div class="nx-figure"><div class="nx-big">{e(v)}<small>{e(u)}</small></div><p>{e(t)}</p></div>' for v, u, t in items)
    return f'<div class="nx-figures">{cells}</div>'


def empty_state() -> str:
    t = TOKENS
    fundus = _svg(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="132" height="132" viewBox="0 0 132 132">'
        f'<circle cx="66" cy="66" r="62" fill="none" stroke="{t["line"]}" stroke-dasharray="3 5"/>'
        f'<circle cx="66" cy="66" r="48" fill="{t["raised"]}" stroke="{t["powder"]}" stroke-width="1.5"/>'
        f'<path d="M40 58c12-9 24-11 38-6M41 76c14 6 27 6 42 0M52 48c6 8 8 20 6 34" fill="none" stroke="{t["line"]}" stroke-width="1.5" stroke-linecap="round"/>'
        f'<circle cx="84" cy="62" r="9" fill="{t["peach-wash"]}" stroke="{t["peach"]}" stroke-width="1.5"/></svg>',
        132, 132,
    )
    return (
        f'<div class="nx-empty">{fundus}<h3>Awaiting a photograph</h3>'
        f'<p class="nx-lede">Upload a colour fundus photograph, or open one of the three IDRiD samples. '
        f"The report appears here: a referral decision, the image-quality check and the regions the models relied on.</p></div>"
    )
