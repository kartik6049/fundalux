"""Build docs/Fundalux_System_Documentation.pdf.

    python scripts/build_docs_pdf.py

Three parts in one document: the software requirements specification (SRS),
the model documentation, and the function and API reference. Every number is
read from the model bundle (models/), signatures and docstrings from the code,
and API routes from the server, so re-running this after retraining or a code
change keeps the document true. Test names in the traceability table are
checked against tests/; a missing one stops the build.
"""

from __future__ import annotations

import ast
import datetime as dt
import html
import importlib
import inspect
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reportlab.graphics.shapes import Circle, Drawing, Line, Polygon, Rect, String  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    BaseDocTemplate, CondPageBreak, Frame, Image, KeepTogether, NextPageTemplate, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents  # noqa: E402

OUT = ROOT / "docs" / "Fundalux_System_Documentation.pdf"
MODELS = ROOT / "models"
SCREENS = ROOT / "docs" / "screens"
FIGURES = MODELS / "figures"

# ----------------------------------------------------------------------------- design tokens (same as the web app)
INK, MUTED, LINE = colors.HexColor("#1D2140"), colors.HexColor("#545A7A"), colors.HexColor("#CFCDEA")
PAPER, RAISED, SUNK = colors.HexColor("#E9E8F8"), colors.HexColor("#F6F5FC"), colors.HexColor("#E1E0F3")
POWDER, POWDER_DEEP, POWDER_WASH = colors.HexColor("#8DB9F0"), colors.HexColor("#2B5C99"), colors.HexColor("#D6E5FA")
PEACH, PEACH_DEEP, PEACH_WASH = colors.HexColor("#F5B69E"), colors.HexColor("#9A4426"), colors.HexColor("#FBE3D9")

for name in ("Newsreader-Regular", "Newsreader-Medium", "IBMPlexSans-Regular", "IBMPlexSans-Medium", "IBMPlexSans-SemiBold", "IBMPlexMono-Regular"):
    pdfmetrics.registerFont(TTFont(name, str(ROOT / "docs" / "fonts" / f"{name}.ttf")))
# <b> inside a paragraph looks up the family by the paragraph's font name, so register it under that name.
pdfmetrics.registerFontFamily("IBMPlexSans-Regular", normal="IBMPlexSans-Regular", bold="IBMPlexSans-SemiBold", italic="IBMPlexSans-Regular", boldItalic="IBMPlexSans-SemiBold")

SERIF, SERIF_M, SANS, SANS_M, SANS_B, MONO = "Newsreader-Regular", "Newsreader-Medium", "IBMPlexSans-Regular", "IBMPlexSans-Medium", "IBMPlexSans-SemiBold", "IBMPlexMono-Regular"

S = {
    "body": ParagraphStyle("body", fontName=SANS, fontSize=9.4, leading=14, textColor=INK, spaceAfter=5),
    "small": ParagraphStyle("small", fontName=SANS, fontSize=8, leading=11.5, textColor=MUTED, spaceAfter=4),
    "cell": ParagraphStyle("cell", fontName=SANS, fontSize=8, leading=10.6, textColor=INK),
    "cellr": ParagraphStyle("cellr", fontName=SANS, fontSize=8, leading=10.6, textColor=INK, alignment=2),
    "head": ParagraphStyle("head", fontName=SANS_M, fontSize=7, leading=9, textColor=MUTED),
    "h1": ParagraphStyle("h1", fontName=SERIF_M, fontSize=24, leading=28, textColor=INK, spaceBefore=0, spaceAfter=10),
    "h2": ParagraphStyle("h2", fontName=SERIF_M, fontSize=15.5, leading=19, textColor=INK, spaceBefore=14, spaceAfter=6, keepWithNext=1),
    "h3": ParagraphStyle("h3", fontName=SANS_B, fontSize=10, leading=13, textColor=POWDER_DEEP, spaceBefore=10, spaceAfter=4, keepWithNext=1),
    "kicker": ParagraphStyle("kicker", fontName=SANS_M, fontSize=7.5, leading=10, textColor=MUTED, spaceAfter=4),
    "bullet": ParagraphStyle("bullet", fontName=SANS, fontSize=9.4, leading=13.6, textColor=INK, leftIndent=12, bulletIndent=2, spaceAfter=2.5, bulletFontName=SANS),
    "sig": ParagraphStyle("sig", fontName=MONO, fontSize=7.6, leading=10.2, textColor=INK, backColor=RAISED, borderColor=LINE, borderWidth=0.5, borderPadding=(4, 5, 4, 5), spaceBefore=6, spaceAfter=5),
    "doc": ParagraphStyle("doc", fontName=SANS, fontSize=8.6, leading=12.2, textColor=INK, leftIndent=6, spaceAfter=3),
    "callout": ParagraphStyle("callout", fontName=SANS, fontSize=9, leading=13, textColor=INK, backColor=PEACH_WASH, borderColor=PEACH_DEEP, borderWidth=0, borderPadding=(7, 9, 7, 9), spaceBefore=6, spaceAfter=10),
    "calm": ParagraphStyle("calm", fontName=SANS, fontSize=9, leading=13, textColor=INK, backColor=POWDER_WASH, borderPadding=(7, 9, 7, 9), spaceBefore=6, spaceAfter=10),
    "caption": ParagraphStyle("caption", fontName=SANS, fontSize=7.6, leading=10.5, textColor=MUTED, spaceBefore=3, spaceAfter=10),
    "toc1": ParagraphStyle("toc1", fontName=SANS_M, fontSize=10, leading=15, textColor=INK, leftIndent=0),
    "toc2": ParagraphStyle("toc2", fontName=SANS, fontSize=8.8, leading=12.5, textColor=MUTED, leftIndent=14),
}
e = html.escape
WIDTH = A4[0] - 40 * mm


def pct(v, d=1):
    return "–" if v is None else f"{v * 100:.{d}f}%"


def num(v, d=3):
    return "–" if v is None else f"{v:.{d}f}"


def P(text, style="body"):
    return Paragraph(text, S[style])


def code(text, size=7.2):
    return f"<font name='{MONO}' size='{size}'>{e(text)}</font>"


def bullets(items):
    return [Paragraph(t, S["bullet"], bulletText="•") for t in items]


def table(head, rows, widths, right=(), zebra=False, head_style="head"):
    """head: list of labels; rows: list of lists of str (markup allowed); widths in mm."""
    data = [[Paragraph(e(h).upper(), S[head_style]) for h in head]]
    for row in rows:
        data.append([c if not isinstance(c, str) else Paragraph(c, S["cellr" if i in right else "cell"]) for i, c in enumerate(row)])
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1, hAlign="LEFT")
    style = [
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK), ("LINEBELOW", (0, 1), (-1, -1), 0.35, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 3.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.6),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    if zebra:
        style += [("BACKGROUND", (0, i), (-1, i), RAISED) for i in range(2, len(data), 2)]
    t.setStyle(TableStyle(style))
    return t


def image(path, width_mm, caption=None):
    from reportlab.lib.utils import ImageReader

    if not Path(path).exists():
        return []
    w, h = ImageReader(str(path)).getSize()
    img = Image(str(path), width=width_mm * mm, height=width_mm * mm * h / w)
    img.hAlign = "LEFT"
    out = [img]
    if caption:
        out.append(P(caption, "caption"))
    return [KeepTogether(out)]


# ----------------------------------------------------------------------------- document with a table of contents


class Doc(BaseDocTemplate):
    def __init__(self, path, **kw):
        super().__init__(str(path), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=22 * mm, bottomMargin=20 * mm,
                         title="Fundalux — System Documentation", author="Fundalux project", subject="SRS, model documentation and function reference", **kw)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")
        self.addPageTemplates([PageTemplate("cover", [frame], onPage=cover_page), PageTemplate("body", [frame], onPage=body_page)])

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and getattr(flowable, "toc_level", None) is not None:
            text = re.sub("<[^>]+>", "", flowable.getPlainText())
            key = f"h{id(flowable)}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=flowable.toc_level, closed=flowable.toc_level > 0)
            self.notify("TOCEntry", (flowable.toc_level, text, self.page, key))


def H(text, level):
    p = Paragraph(text, S["h1" if level == 0 else "h2"])
    p.toc_level = level
    return p


def draw_mark(c, x, y, r):
    """The Fundalux mark: a fundus inside an instrument reticle, optic disc in peach."""
    c.saveState()
    c.setStrokeColor(POWDER_DEEP); c.setLineWidth(r * 0.06)
    c.circle(x, y, r, stroke=1, fill=0)
    c.setFillColor(POWDER_WASH); c.setStrokeColor(POWDER); c.circle(x, y, r * 0.68, stroke=1, fill=1)
    c.setStrokeColor(POWDER_DEEP)
    for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        c.line(x + dx * r * 0.9, y + dy * r * 0.9, x + dx * r * 1.18, y + dy * r * 1.18)
    c.setFillColor(PEACH); c.setStrokeColor(PEACH_DEEP); c.setLineWidth(r * 0.04)
    c.circle(x + r * 0.27, y + r * 0.04, r * 0.18, stroke=1, fill=1)
    c.restoreState()


def cover_page(c, doc):
    w, h = A4
    c.saveState()
    c.setFillColor(PAPER); c.rect(0, 0, w, h, stroke=0, fill=1)
    c.setStrokeColor(POWDER); c.setLineWidth(0.6)
    for i, r in enumerate((120, 175, 240, 320)):
        c.setStrokeAlpha(0.35 - i * 0.07); c.circle(w - 30 * mm, h - 10 * mm, r, stroke=1, fill=0)
    c.setStrokeAlpha(1)
    draw_mark(c, 32 * mm, h - 52 * mm, 11 * mm)
    c.setFillColor(INK); c.setFont(SERIF_M, 46); c.drawString(20 * mm, h - 92 * mm, "Funda")
    c.setFillColor(POWDER_DEEP); c.drawString(20 * mm + pdfmetrics.stringWidth("Funda", SERIF_M, 46), h - 92 * mm, "lux")
    c.setFillColor(INK); c.setFont(SERIF, 21)
    c.drawString(20 * mm, h - 108 * mm, "System documentation")
    c.setFont(SANS, 10.5); c.setFillColor(MUTED)
    for i, line in enumerate(("Software requirements specification · Model documentation · Function and API reference",
                              "Explainable screening for referable diabetic retinopathy")):
        c.drawString(20 * mm, h - 122 * mm - i * 15, line)
    c.setStrokeColor(INK); c.setLineWidth(0.8); c.line(20 * mm, 70 * mm, w - 20 * mm, 70 * mm)
    meta = doc.meta
    rows = [("Document version", "1.0"), ("Date", meta["date"]), ("Software", f"fundalux {meta['version']}"), ("Model", meta["run"]), ("Model trained", meta["trained"])]
    for i, (k, v) in enumerate(rows):
        c.setFont(SANS_M, 8); c.setFillColor(MUTED); c.drawString(20 * mm, 62 * mm - i * 12, k.upper())
        c.setFont(SANS, 9); c.setFillColor(INK); c.drawString(62 * mm, 62 * mm - i * 12, v)
    c.setFillColor(PEACH_WASH); c.rect(20 * mm, 14 * mm, w - 40 * mm, 13 * mm, stroke=0, fill=1)
    c.setFillColor(PEACH_DEEP); c.rect(20 * mm, 14 * mm, 1.2 * mm, 13 * mm, stroke=0, fill=1)
    c.setFont(SANS, 8); c.setFillColor(INK)
    c.drawString(25 * mm, 21.5 * mm, "Research prototype. Not a medical device, no regulatory clearance, not prospectively validated.")
    c.drawString(25 * mm, 17 * mm, "A qualified clinician must confirm every result.")
    c.restoreState()


def body_page(c, doc):
    w, h = A4
    c.saveState()
    c.setFont(SANS_M, 7.2); c.setFillColor(MUTED)
    c.drawString(20 * mm, h - 13 * mm, "FUNDALUX · SYSTEM DOCUMENTATION")
    c.drawRightString(w - 20 * mm, h - 13 * mm, getattr(doc, "part", "").upper())
    c.setStrokeColor(LINE); c.setLineWidth(0.5); c.line(20 * mm, h - 15.5 * mm, w - 20 * mm, h - 15.5 * mm)
    c.line(20 * mm, 13 * mm, w - 20 * mm, 13 * mm)
    c.setFont(SANS, 7.2)
    c.drawString(20 * mm, 9 * mm, "Research prototype · not a medical device · version 1.0, " + doc.meta["date"])
    c.drawRightString(w - 20 * mm, 9 * mm, str(doc.page))
    c.restoreState()


class PartMarker(Spacer):
    """Zero-height flowable that sets the running header from the next page on (headers are drawn as a page begins)."""

    def __init__(self, name):
        super().__init__(0, 0)
        self.name = name

    def draw(self):
        self.canv._doctemplate.part = self.name


# ----------------------------------------------------------------------------- architecture diagram


def architecture():
    d = Drawing(WIDTH, 92 * mm)

    def box(x, y, w, h, title, lines, fill=RAISED, stroke=LINE, title_color=INK):
        d.add(Rect(x, y, w, h, rx=4, ry=4, fillColor=fill, strokeColor=stroke, strokeWidth=0.7))
        d.add(String(x + 6, y + h - 13, title, fontName=SANS_B, fontSize=8, fillColor=title_color))
        for i, line in enumerate(lines):
            d.add(String(x + 6, y + h - 25 - i * 10, line, fontName=SANS, fontSize=6.8, fillColor=MUTED))

    def arrow(x1, y1, x2, y2, color=POWDER_DEEP):
        d.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=1))
        import math

        a = math.atan2(y2 - y1, x2 - x1)
        p = [x2, y2, x2 - 6 * math.cos(a - 0.4), y2 - 6 * math.sin(a - 0.4), x2 - 6 * math.cos(a + 0.4), y2 - 6 * math.sin(a + 0.4)]
        d.add(Polygon(p, fillColor=color, strokeColor=color))

    W = WIDTH
    box(0, 205, 120, 52, "Browser", ["Pages: overview, guide,", "evidence, screen, sites,", "camp, review"])
    box(0, 128, 120, 58, "Clinician / organiser", ["Review code: grading", "Admin token: results,", "calibration, new sets"])
    box(150, 150, 165, 107, "FastAPI server  (server/)", ["main.py  pages + JSON API", "report.py  evidence from the bundle", "Jinja templates, plain JS, CSS", "Security headers, CSP, upload limits", "Screening on a worker thread", "Admin / review tokens (HMAC compare)"], fill=POWDER_WASH, stroke=POWDER, title_color=POWDER_DEEP)
    box(345, 150, W - 345, 107, "Pipeline  (fundalux/)", ["1 crop to retina", "2 quality gate (rules + network)", "3 illumination + CLAHE", "4 three ONNX networks, averaged", "5 threshold: global or site", "6 closed-form Grad-CAM"], fill=PEACH_WASH, stroke=PEACH, title_color=PEACH_DEEP)
    box(150, 40, 165, 88, "Model bundle  (models/)", ["onnx/  3 networks + gate + heads.npz", "config.json  thresholds, inputs", "metrics.json  evaluation", "sites.json  per-site thresholds", "reports/  predictions, robustness"])
    box(345, 40, W - 345, 88, "Review store  (review/)", ["set.json  photos under neutral ids,", "reference grades, model output", "responses.sqlite3  answers", "(volume when containerised)"])
    box(0, 40, 120, 70, "Training (Kaggle GPU)", ["Fundalux_Kaggle_Training", ".ipynb writes the bundle,", "checks ONNX = PyTorch"])
    arrow(120, 231, 150, 231)
    arrow(120, 157, 150, 180)
    arrow(315, 203, 345, 203)
    arrow(232, 150, 232, 128)
    arrow(410, 150, 410, 128, PEACH_DEEP)
    arrow(120, 75, 150, 80)
    d.add(String(0, 12, "Arrows show requests and data flow. Only models/sites.json and review/ are written while the server runs.", fontName=SANS, fontSize=6.8, fillColor=MUTED))
    return d


# ----------------------------------------------------------------------------- content


def load():
    from fundalux import __version__
    from server.report import build_report

    config = json.loads((MODELS / "config.json").read_text())
    report = build_report(MODELS)
    return config, report, __version__


def front_matter(meta, story):
    story += [PartMarker("About this document"), NextPageTemplate("body"), PageBreak()]
    story += [P("ABOUT THIS DOCUMENT", "kicker"), Paragraph("Contents", S["h1"])]
    toc = TableOfContents()
    toc.levelStyles = [S["toc1"], S["toc2"]]
    toc.dotsMinLevel = 0
    story += [toc, PageBreak()]
    story += [P("ABOUT THIS DOCUMENT", "kicker"), Paragraph("Document information", S["h1"])]
    story += [P("This document describes Fundalux completely: what it must do (the software requirements specification), how its models were built and how well they perform (the model documentation), and what every function and API endpoint does (the reference). It is generated from the code and the model files by <font name='%s'>scripts/build_docs_pdf.py</font>, so figures and signatures match the system it describes." % MONO)]
    story.append(table(["Item", "Value"], [
        ["Product", "Fundalux: explainable screening for referable diabetic retinopathy (formerly NetraXAI)"],
        ["Status", "Research prototype; not a medical device"],
        ["Software version", f"fundalux {meta['version']}"],
        ["Model", f"{meta['run']} (trained {meta['trained']})"],
        ["Audience", "Clinicians evaluating the system, developers maintaining it, and anyone hosting it"],
        ["Generated", meta["date"]],
    ], [40, 130]))
    story += [Spacer(1, 8), P("How to read it", "h3")]
    story += bullets([
        "<b>Part I</b> is a short overview with the architecture.",
        "<b>Part II, the SRS</b>, lists every requirement with an identifier, its priority, how it is verified, and its current status.",
        "<b>Part III, the model documentation</b>, is written for clinicians and reviewers: data, training, thresholds, results, explainability, limitations.",
        "<b>Part IV, the reference</b>, documents the HTTP API, configuration, command-line tools and every public Python function.",
    ])
    story += [Spacer(1, 6), P("Revision history", "h3")]
    story.append(table(["Version", "Date", "Change"], [["1.0", meta["date"], "First complete edition: SRS, model documentation, function and API reference"]], [22, 30, 118]))


def part_overview(config, report, story):
    story += [PartMarker("Part I · Overview"), PageBreak(), P("PART I", "kicker"), H("System overview", 0)]
    story += [P("Fundalux reads a colour fundus photograph and does one of two things. If the photograph cannot be graded, it refuses it and says why. Otherwise it reports whether <b>referable diabetic retinopathy</b> (ICDR moderate non-proliferative retinopathy or worse) is suspected, an estimated grade, and a heatmap of the regions that raised the score. A threshold calibrated to the camera turns the score into a referral decision.")]
    story += [P("Every result is meant for a clinician to confirm. The system is designed to sit in front of a grader or ophthalmologist in a screening programme and decide who is seen first, not to replace them.", "calm")]
    story += [H("Architecture", 1), architecture(), P("Figure 1. Components and data flow.", "caption")]
    story += [H("Repository layout", 1)]
    story.append(table(["Path", "Contents"], [
        ["fundalux/", "Core Python package shared by training and serving: preprocessing, quality gate, inference backends, pipeline, site calibration, clinician review, metrics, camp simulation"],
        ["server/", "FastAPI web application: pages, JSON API, evidence report, templates, CSS design system, JavaScript"],
        ["models/", "The trained bundle: weights, ONNX serving copies, config, metrics, site thresholds, reports, figures"],
        ["review/", "The clinician review set (photographs under neutral ids, reference grades, model output); answers in responses.sqlite3"],
        ["Fundalux_Kaggle_Training.ipynb", "Trains and evaluates every model on a free Kaggle GPU; generated from scripts/notebook_src.py"],
        ["tests/", "79 automated tests: preprocessing, metrics, ONNX equivalence, sites, review, server, app, notebook consistency"],
        ["Dockerfile", "Container: CPU only, no PyTorch, non-root, health check"],
        ["app/", "Earlier Streamlit interface, kept as a local alternative"],
    ], [52, 118]))
    m = report["model"]
    story += [H("The pipeline in brief", 1)]
    story.append(table(["Step", "What happens", "Module"], [
        ["1 Locate", f"Find the fundus disc, cut a square around it (extended where the camera cut it off), resize to {m['image_size']} px, black out the rest", "preprocess"],
        ["2 Gate", f"Rule checks (size, retina found, colour), then the gate network (EfficientNet-B0, {m['quality_input']} px); refuse at P(ungradable) ≥ {num(m['quality_threshold'])}", "quality, pipeline"],
        ["3 Enhance", "Divide lightness by its smoothed illumination field, then CLAHE on lightness", "preprocess"],
        ["4 Score", "Three networks read the enhanced image at their own resolution; probabilities are averaged", "inference, pipeline"],
        ["5 Decide", f"Refer when the score reaches the site's threshold, or the global threshold {num(m['global_threshold'])}", "sites, pipeline"],
        ["6 Explain", "Grad-CAM per network from its feature map (closed form), averaged and overlaid in blue", "inference, heatmap"],
    ], [20, 118, 32]))
    story += [H("Ways to run it", 1)]
    story.append(table(["Mode", "Command", "Use"], [
        ["Development", code("uvicorn server.main:app --port 8000"), "Local work; PyTorch or ONNX backend"],
        ["Container", code("docker run --env-file .env -p 127.0.0.1:8000:8000 -v fundalux-review:/app/review fundalux:latest"), "Hosting; ONNX only, non-root, health check"],
        ["Shared from a laptop", code("cloudflared tunnel --url http://localhost:8000"), "Free public link for a clinician review or a demo"],
        ["Command line", code("python -m fundalux.cli photo.jpg"), "One photograph, JSON result"],
    ], [28, 92, 50]))


# ----------------------------------------------------------------------------- SRS


TESTS: dict[str, str] = {}


def collect_tests():
    for f in sorted((ROOT / "tests").glob("test_*.py")):
        for node in ast.parse(f.read_text()).body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                TESTS[node.name] = f.name


def t(*names):
    """Test references for the traceability column; the build stops if a test does not exist."""
    missing = [n for n in names if n not in TESTS]
    if missing:
        raise SystemExit(f"Traceability names tests that do not exist: {missing}")
    return "<br/>".join(f"<font name='{MONO}' size='6.6'>{wrap_name(n, 6.6, 40 * mm - 20)}</font>" for n in names)


def wrap_name(name, size, width):
    """Break an identifier after underscores so it fits the width; continuation lines are indented."""
    lines, line = [], ""
    for part in re.findall(r"[^_]+_?|_", name):
        if line and pdfmetrics.stringWidth(line + part, MONO, size) > width:
            lines.append(line)
            line = ""
        line += part
    lines.append(line)
    return "<br/>&nbsp;&nbsp;".join(lines)


def requirement_table(rows):
    return table(["ID", "Requirement", "Priority", "Verified by", "Status"], rows, [16, 82, 15, 40, 17])


MET = "<font color='#2B5C99'><b>met</b></font>"
PARTIAL = "<font color='#9A4426'><b>partly met</b></font>"
NOT_MET = "<font color='#9A4426'><b>not met</b></font>"
MANUAL = "<font color='#545A7A'>manual</font>"


def part_srs(config, report, story):
    m, tg = report["model"], report["targets"]
    sets = report["sets"]
    severe = [r["rejected"] for r in report.get("robustness", []) if r["severity"] >= 0.7]
    mild = [r["rejected"] for r in report.get("robustness", []) if 0 < r["severity"] <= 0.15]
    story += [PartMarker("Part II · Software requirements specification"), PageBreak(), P("PART II", "kicker"), H("Software requirements specification", 0)]
    story += [P("Structured after ISO/IEC/IEEE 29148. Requirements use “shall”; each has an identifier, a priority (<b>Must</b>, <b>Should</b>, <b>Could</b>), the automated test or check that verifies it, and its status against the current build and model.")]

    story += [H("1. Introduction", 1), P("1.1 Purpose", "h3"), P("This specification defines what Fundalux must do and how well, so that the system can be built, verified, hosted and evaluated by clinicians against explicit criteria.")]
    story += [P("1.2 Scope", "h3"), P("In scope: triage of single colour fundus photographs for referable diabetic retinopathy; refusal of ungradable photographs; visual explanation; per-site threshold calibration; blinded clinician review; evidence and documentation pages; a screening-camp simulation; training and export of the models. Out of scope: diagnosis; diabetic macular oedema; other eye disease; combining both eyes or several visits; storing patient records; integration with hospital systems.")]
    story += [P("1.3 Definitions", "h3")]
    story.append(table(["Term", "Meaning"], [
        ["ICDR scale", "International Clinical Diabetic Retinopathy severity scale: 0 no DR, 1 mild NPDR, 2 moderate NPDR, 3 severe NPDR, 4 proliferative DR"],
        ["Referable DR", "ICDR grade 2 or worse. Macular oedema, normally also referable, is not assessed"],
        ["Gradable", "A photograph good enough for a grader to assign a grade; decided by the quality gate"],
        ["Referable-DR score", "The ensemble's averaged output, 0 to 1. A ranking, not a calibrated probability"],
        ["Global threshold", f"The score above which a referral is made when no site is selected ({num(m['global_threshold'])})"],
        ["Site profile", "A threshold calibrated on graded photographs from one camera, with its expected sensitivity and specificity"],
        ["Sensitivity / specificity", "Share of referable photographs referred / of non-referable photographs not referred"],
        ["Cross-fitting", "Tune a threshold on one half of the photographs, test on the other half, swap, repeat with random halvings"],
        ["Grad-CAM", "Gradient-weighted class activation map: where the last convolutional features raised the score"],
        ["Review set", "A fixed set of photographs graded blind by clinicians, under neutral ids"],
        ["Bundle", "The folder of trained models and their configuration and evaluation files (models/)"],
    ], [36, 134]))
    story += [P("1.4 References", "h3")]
    story += bullets([
        "ISO/IEC/IEEE 29148:2018, Systems and software engineering: requirements engineering.",
        "Wilkinson CP et al. International clinical diabetic retinopathy and diabetic macular edema disease severity scales. <i>Ophthalmology</i> 2003;110:1677–82.",
        "Gulshan V et al. Deep learning algorithm for detection of diabetic retinopathy in retinal fundus photographs. <i>JAMA</i> 2016;316:2402–10.",
        "Project documents: README.md, models/README.md.",
    ])

    story += [H("2. Overall description", 1), P("2.1 Product perspective", "h3")]
    story += [P("A self-contained web application and Python package. Models are trained offline on a cloud GPU and shipped as a bundle; the server runs them on CPU with ONNX Runtime. No external service is called at run time, and the fonts are self-hosted, so the system works offline in a screening camp.")]
    story += [P("2.2 User classes", "h3")]
    story.append(table(["User", "Goal", "Access"], [
        ["Screening operator", "Screen a photograph, retake refused ones, print the report", "Public pages"],
        ["Clinician", "Study how the system works and its evidence; grade the review set blind", "Public pages; review code"],
        ["Organiser / administrator", "Calibrate sites, create review sets, read review results", "Admin token"],
        ["Maintainer", "Retrain, export, test, host", "Repository, Kaggle notebook"],
    ], [38, 92, 40]))
    story += [P("2.3 Operating environment", "h3")]
    story += bullets([
        "Server: Linux container (python:3.12-slim) or macOS/Linux with Python 3.12; 2 vCPU and 2 GB of memory recommended; no GPU.",
        "Client: a current desktop or mobile browser; pages work without JavaScript except the interactive tools.",
        "Training: Kaggle notebook with one T4 GPU, about 6 hours per full run.",
    ])
    story += [P("2.4 Constraints", "h3")]
    story += bullets([
        "Development machine is an Intel Mac limited to PyTorch 2.2 and NumPy 1.x; the server must also run on NumPy 2 without PyTorch.",
        "Training data are public research datasets; their licences limit use to research and education.",
        "Not a medical device: every page and report must carry the research-prototype disclaimer.",
    ])
    story += [P("2.5 Assumptions and dependencies", "h3")]
    story += bullets([
        "Photographs are colour, macula- or disc-centred fundus images of one eye from a desktop or portable fundus camera.",
        "Reference grades used for calibration and review come from qualified graders.",
        "Runtime dependencies: FastAPI, Uvicorn, Jinja2, ONNX Runtime, NumPy, pandas, OpenCV, Pillow, SimPy.",
    ])

    story += [H("3. Functional requirements", 1)]
    story += [P("3.1 Screening", "h3")]
    story.append(requirement_table([
        ["FR-1", "The system shall accept JPEG, PNG, TIFF and BMP photographs up to a configurable size (default 20 MB) and reject anything that is not a readable colour image.", "Must", t("test_screen_rejects_bad_input"), MET],
        ["FR-2", "The system shall locate the fundus disc, including discs cut off at the frame edge, and crop a square around it.", "Must", t("test_find_fundus_handles_cut_off_disc", "test_find_fundus_ignores_position_and_scale"), MET],
        ["FR-3", "The system shall refuse a photograph that is too small, shows no retina, shows less than half the retina, or is not a colour fundus photograph, before any grading.", "Must", t("test_basic_checks", "test_black_image_is_not_a_fundus"), MET],
        ["FR-4", "The system shall refuse a photograph whose ungradable probability from the gate network reaches the configured threshold, and give the reason and a retake instruction.", "Must", t("test_pipeline_refuses_ungradable"), MET],
        ["FR-5", "The system shall explain a refusal in plain language (blur, darkness, glare, low contrast) from image statistics.", "Should", t("test_severe_blur_is_flagged_by_stats"), MET],
        ["FR-6", "The system shall correct illumination and equalise contrast before scoring.", "Must", t("test_enhance_evens_out_brightness"), MET],
        ["FR-7", "The system shall score each gradable photograph with every network in the ensemble at its own input size and average their probabilities.", "Must", t("test_ensemble_probabilities", "test_ensemble_grad_cam_feeds_each_model_its_own_size"), MET],
        ["FR-8", "The system shall decide referral at the selected site's threshold, or the global threshold when no site is selected, and give advice matching the decision and estimated grade.", "Must", t("test_pipeline_applies_a_site_threshold"), MET],
        ["FR-9", "Switching site shall re-decide an already screened photograph without re-running the networks.", "Should", t("test_pipeline_applies_a_site_threshold", "test_screen_tab_screens_a_sample_and_switches_site"), MET],
        ["FR-10", "The system shall show a heatmap of the regions that raised the referable score, averaged over the ensemble, in a colour that does not occur in fundus photographs.", "Must", t("test_grad_cam", "test_closed_form_grad_cam_matches_autograd"), MET],
        ["FR-11", "The system shall report an estimated ICDR grade with its probability distribution, marked as secondary to the referral decision.", "Should", t("test_pipeline_accepts_then_predicts"), MET],
        ["FR-12", "The screening report shall be printable with date, photograph, site, decision, images and disclaimer.", "Should", "Browser check (print preview)", MANUAL],
        ["FR-13", "For demonstration, the system shall simulate blur, motion, glare, haze and under-exposure at a chosen severity, and optionally grade refused photographs, labelled as not for action.", "Could", t("test_degradations", "test_degradations_accept_uncropped_photos"), MET],
    ]))
    story += [P("3.2 Site calibration", "h3")]
    story.append(requirement_table([
        ["FR-20", "The system shall fit a site threshold from graded photographs of that site, at a target sensitivity, using only photographs that pass the quality gate.", "Must", t("test_calibrate_site_joins_and_drops_rejected", "test_site_threshold_fixes_a_shifted_scale"), MET],
        ["FR-21", "The system shall refuse calibration with fewer than 20 referable or 20 non-referable gradable photographs, and warn below 50 of either.", "Must", t("test_calibration_refuses_too_few_cases"), MET],
        ["FR-22", "Each site profile shall report the sensitivity and specificity to expect, measured on photographs the threshold was not tuned on.", "Must", t("test_crossfit_is_honest", "test_builtin_profiles"), MET],
        ["FR-23", "Site profiles shall be bound to the model that produced the scores, and marked stale and hidden after retraining.", "Must", t("test_registry_roundtrip_and_staleness"), MET],
        ["FR-24", "Calibrating and removing sites shall require the admin token; built-in sites cannot be removed.", "Must", t("test_admin_endpoints_need_the_token", "test_calibrate_save_and_remove_a_site"), MET],
        ["FR-25", "Calibration shall also be possible from the command line for large batches.", "Should", t("test_read_labels_formats"), MET],
    ]))
    story += [P("3.3 Clinician review", "h3")]
    story.append(requirement_table([
        ["FR-30", "The system shall present a fixed review set under neutral ids so that file names cannot reveal the answer.", "Must", t("test_set_hides_names_and_stores_model_output"), MET],
        ["FR-31", "Reviewers shall record gradability, ICDR grade, referral for another reason, confidence and a note; a gradable answer requires a grade.", "Must", t("test_store_and_analysis", "test_review_flow_over_http"), MET],
        ["FR-32", "In blinded mode the model's answer shall never be shown; in teaching mode it shall be shown only after the reviewer has answered, and such answers shall be marked.", "Must", t("test_review_flow_over_http"), MET],
        ["FR-33", "Each reviewer shall see the photographs in their own fixed random order, and be able to stop, resume and skip.", "Should", t("test_set_hides_names_and_stores_model_output", "test_review_flow_over_http"), MET],
        ["FR-34", "The organiser shall see agreement between each reviewer and the system and between reviewers (kappa), each against the reference on the same photographs, quality-gate agreement, disagreements with notes, and a CSV of all answers.", "Must", t("test_store_and_analysis", "test_review_flow_over_http"), MET],
        ["FR-35", "Grading needs the review code; results, export and new sets need the admin token.", "Must", t("test_review_flow_over_http"), MET],
    ]))
    story += [P("3.4 Evidence, documentation and simulation", "h3")]
    story.append(requirement_table([
        ["FR-40", "The evidence and guide pages shall take every figure from the model bundle so they stay true after retraining.", "Must", t("test_pages_render", "test_camp_and_report"), MET],
        ["FR-41", "The evidence page shall show ROC curves with operating points, confidence intervals, results by grade measured out of sample, confusion matrices, heatmap validation and robustness.", "Must", t("test_camp_and_report"), MET],
        ["FR-42", "The camp model shall simulate a screening camp with and without AI triage and recommend the cheapest staffing within waiting-time and backlog limits.", "Should", t("test_ai_triage_cuts_ophthalmologist_workload", "test_sweep_and_recommendation"), MET],
        ["FR-43", "All functions shall be available through a documented JSON API.", "Should", t("test_screen_upload_sample_and_sites"), MET],
    ]))
    story += [P("3.5 Training and export", "h3")]
    story.append(requirement_table([
        ["FR-50", "Training shall split by patient so that both eyes of a patient are on the same side of every split, and keep two hospitals' datasets entirely out of training.", "Must", t("test_splits_keep_patients_together_and_external_sets_out"), MET],
        ["FR-51", "The decision threshold shall be chosen on validation data and frozen before any test photograph is scored.", "Must", "Notebook section 8", MET],
        ["FR-52", "The export shall produce ONNX copies whose scores and heatmaps match PyTorch, and stop if they do not.", "Must", t("test_exported_bundle_matches_torch"), MET],
        ["FR-53", "The notebook shall be generated from the package so training and serving share one implementation, and the build shall fail if it is out of date.", "Must", t("test_notebook_is_up_to_date"), MET],
    ]))

    story += [H("4. External interfaces", 1), P("4.1 User interface", "h3")]
    story += [P("Seven pages share one design system: lavender paper (#E9E8F8), indigo ink, powder blue (#8DB9F0) for instruments and calm results, warm sunset peach (#F5B69E) only for referrals; Newsreader for headings and IBM Plex Sans for text and numbers. Every text colour pair meets WCAG AA contrast (4.5:1).")]
    for name, cap, wmm in (("home.jpg", "Figure 2. Overview page.", 150), ("screen.jpg", "Figure 3. Screening report: decision, score instrument with the site threshold, quality score, images, heatmap and grade estimate.", 150),
                           ("review.jpg", "Figure 4. Blinded grading screen.", 150), ("evidence_idrid.jpg", "Figure 5. Evidence page, one test set: ROC curve with both operating points, referral by grade, confusion matrix.", 150)):
        story += image(SCREENS / name, wmm, cap)
    story += [P("4.2 Software interfaces", "h3")]
    story += bullets([
        "HTTP JSON API (Part IV), with interactive documentation at /api/docs.",
        "Model bundle files: config.json, metrics.json, sites.json, onnx/, reports/ (Part III).",
        "ONNX models: input <font name='%s'>image</font> float32 [N, 3, H, W] in 0–255 RGB; outputs <font name='%s'>features</font> and <font name='%s'>logits</font> (referable logit and five grade logits)." % (MONO, MONO, MONO),
    ])

    story += [H("5. Non-functional requirements", 1)]
    story.append(table(["ID", "Requirement", "Measured / implemented", "Status"], [
        ["NFR-1 Accuracy", f"Sensitivity above {pct(tg['sensitivity'], 0)} and specificity above {pct(tg['specificity'], 0)} for referable DR on held-out data", "With site thresholds: met on APTOS, IDRiD and Messidor-2; EyePACS specificity " + pct(sets.get("eyepacs", {}).get("site", {}).get("specificity")), PARTIAL],
        ["NFR-2 Safety", "Severe NPDR and proliferative DR shall almost never be missed", f"At least {pct(report['summary']['severe_referred_min'], 0)} referred in every test set", MET],
        ["NFR-3 Performance", "One photograph screened in under 30 s on a 2-vCPU CPU host", "On a 2012 Intel laptop: 5.4 s natively, 9–16 s in Docker, about 23 s through the public tunnel; models load in 3 s", MET],
        ["NFR-4 Footprint", "Run in 2 GB of memory, without PyTorch", "About 790 MB in the container; image 1.35 GB", MET],
        ["NFR-5 Privacy", "Photographs processed in memory only; never stored or logged", "Upload bytes released after each request; review set holds only de-identified research images", MET],
        ["NFR-6 Security", "Write operations behind secrets; strict browser policy; least privilege", "HMAC-compared tokens; CSP without inline or third-party scripts; X-Frame-Options DENY; nosniff; non-root container; path checks on file routes", MET],
        ["NFR-7 Robustness", "Refuse severely degraded photographs; accept good ones; never crash on bad input", f"Severe synthetic defects (severity ≥ 0.7) refused: {pct(min(severe))}; mild ones (0.15) refused at most {pct(max(mild))}; corrupt files return 422" if severe and mild else "Corrupt files return 422", MET],
        ["NFR-8 Concurrency", "Stay responsive while screening", "Models on worker threads; at most 2 concurrent screenings, others wait 90 s then get 503; health check answers in under 0.15 s while screening", MET],
        ["NFR-9 Accessibility", "Readable, keyboard-usable, responsive", "WCAG AA contrast, keyboard grading and tabs, reduced-motion support, phone layouts, print styles", MET],
        ["NFR-10 Offline", "No run-time internet dependency", "Fonts, scripts and styles self-hosted; no external calls", MET],
        ["NFR-11 Maintainability", "One implementation for training and serving; automated tests", "79 tests; notebook generated from the package and checked; documentation generated from code", MET],
        ["NFR-12 Portability", "Run on NumPy 1 and 2, macOS and Linux", "Verified in a torch-free NumPy 2.2 / pandas 3 environment and in the container", MET],
        ["NFR-13 Validation", "Prospective clinical validation before any clinical use", "Not done; the blinded review is the first step", NOT_MET],
    ], [26, 52, 74, 18]))
    story += [P("NFR-1 is the plan's original accuracy target. With one global threshold, specificity falls short at every test site (76–83%); calibrating each camera's threshold reaches the target at three of the four sites. Part III explains why.", "small")]

    story += [H("6. Safety, regulatory and ethical constraints", 1)]
    story += bullets([
        "Every page, report and API response carries: “Screening aid only – not a diagnosis. A qualified clinician must confirm every result.”",
        "No patient-identifying data is requested or stored; public deployments warn users not to upload identifiable photographs.",
        "The system must not be used for clinical decisions without prospective validation and regulatory review.",
        "Known failure modes are documented to users (Part III §11), including those that cause over-referral.",
    ])


# ----------------------------------------------------------------------------- model documentation


def part_model(config, report, story):
    m, tg, sets = report["model"], report["targets"], report["sets"]
    names = {"efficientnet_b3": "EfficientNet-B3", "efficientnet_v2_s": "EfficientNetV2-S", "convnext_tiny": "ConvNeXt-Tiny"}
    years = {"efficientnet_b3": "2019", "efficientnet_v2_s": "2021", "convnext_tiny": "2022"}
    story += [PartMarker("Part III · Model documentation"), PageBreak(), P("PART III", "kicker"), H("Model documentation", 0)]
    story += [P("A model card for the networks behind Fundalux: what they are for, what they were trained and tested on, how decisions are made, how well they perform, how the heatmaps were validated, and where they fail. All figures are for photographs the networks never trained on.")]

    story += [H("1. Model card summary", 1)]
    story.append(table(["Field", "Value"], [
        ["Task", "Binary triage of referable diabetic retinopathy from one colour fundus photograph, with a five-grade estimate and a heatmap"],
        ["Classifier", "Ensemble of " + ", ".join(f"{names.get(x['arch'], x['arch'])} ({x['input_size']} px)" for x in m["members"]) + "; probabilities averaged"],
        ["Quality gate", f"{m['quality_arch'].replace('efficientnet_b0', 'EfficientNet-B0')} at {m['quality_input']} px; refuses at P(ungradable) ≥ {num(m['quality_threshold'])}"],
        ["Training run", f"{m['trained'].get('run')} · {m['trained'].get('date')} · {m['trained'].get('hours', 0):.1f} h on one T4 GPU · PyTorch {m['trained'].get('torch')}"],
        ["Global threshold", f"{num(m['global_threshold'])}, tuned for {pct(tg['tuning_sensitivity'], 0)} sensitivity on pooled validation data"],
        ["Headline result", f"AUC {num(report['summary']['auc_min'])}–{num(report['summary']['auc_max'])} on four test sets; severe NPDR and PDR referred in at least {pct(report['summary']['severe_referred_min'], 0)}"],
        ["Explainability", f"Heatmap peak on an ophthalmologist-marked lesion in {pct(report['summary']['pointing'], 0)} of 81 photographs ({pct(report['summary']['pointing_baseline'], 0)} for untrained networks)"],
        ["Serving", "ONNX Runtime on CPU; closed-form Grad-CAM; matches PyTorch to within 1e-4"],
        ["Status", "Research prototype; retrospective evaluation only"],
    ], [34, 136]))

    story += [H("2. Intended use", 1)]
    story += bullets([
        "<b>Intended:</b> triage in diabetic-retinopathy screening programmes, deciding who an ophthalmologist sees first; teaching; research.",
        "<b>Users:</b> screening staff, with clinician confirmation of every result.",
        "<b>Not intended:</b> diagnosis; diabetic macular oedema; other eye disease (glaucoma, macular degeneration, occlusions); children; photographs other than colour fundus images; unsupervised use.",
    ])

    story += [H("3. Data", 1)]
    story.append(table(["Dataset", "Origin", "Role", "Reference grades"], [
        ["EyePACS (Kaggle 2015, resized)", "Many US clinics and cameras", "Training, validation, test", "Clinician grades; known label noise"],
        ["APTOS 2019", "Aravind Eye Hospital, India", "Training, validation, test", "Clinician grades"],
        ["IDRiD", "Nanded, India", "External test; 81 lesion masks for the heatmap check", "Expert grades; pixel-level lesion outlines"],
        ["Messidor-2", "France", "External test", "Adjudicated by retina specialists"],
        ["EyeQ", "Labels for EyePACS photographs", "Quality gate", "Good / usable / reject"],
    ], [36, 38, 52, 44]))
    imgs = report["data"].get("images", {})
    if imgs:
        story += [Spacer(1, 4)]
        rows = []
        for src in ("eyepacs", "aptos", "idrid", "messidor"):
            row = [src.replace("eyepacs", "EyePACS").replace("aptos", "APTOS").replace("idrid", "IDRiD").replace("messidor", "Messidor-2")]
            row += [f"{imgs.get(f'{src}/{s}', 0):,}" if imgs.get(f"{src}/{s}") else "–" for s in ("train", "val", "test", "external")]
            rows.append(row)
        story.append(KeepTogether([table(["Source", "Train", "Validation", "Test", "External test"], rows, [50, 30, 30, 30, 30], right=(1, 2, 3, 4)), P("Photographs per split. EyePACS and APTOS were split 80 / 10 / 10 by patient. IDRiD and Messidor-2 contributed nothing to training or to any choice made during training.", "caption")]))

    story += [H("4. Preprocessing", 1)]
    story += bullets([
        "<b>Crop:</b> threshold the brightest channel, take the largest bright region as the disc, cut a square around it (extended past the frame when the camera cut the disc off), resize, black out everything outside the disc.",
        "<b>Illumination:</b> divide lightness (CIELAB L) by a heavily smoothed copy of itself, removing vignetting and uneven flash while keeping colour.",
        "<b>Contrast:</b> CLAHE on lightness (clip limit 2, 8×8 tiles).",
        "The quality gate sees the crop <i>before</i> enhancement, because enhancement hides the defects it must detect.",
    ])
    story += image(FIGURES / "preprocessing_steps.png", 140, "Figure 6. One photograph from each dataset: original, cropped, illumination-corrected, contrast-equalised (the networks' input).")

    story += [H("5. Quality gate", 1)]
    qg = report["quality_gate"]
    story += [P(f"An EfficientNet-B0 at {m['quality_input']} px estimates the probability that a photograph is ungradable. It learnt from EyeQ's human quality labels ({qg.get('real_labels', 0):,} EyePACS photographs, {qg.get('real_ungradable', 0):,} marked reject) and from good photographs degraded on purpose: severe blur, motion, glare, haze and under-exposure labelled ungradable, mild versions labelled gradable. The threshold maximises Youden's J on validation data.")]
    rows = [[("Human labels only" if "real EyeQ" in k else "Human labels + synthetic defects"), f"{int(v['ungradable'])} of {int(v['images']):,}", num(v["AUC"]), pct(v["ungradable caught"]), pct(v["gradable accepted"])] for k, v in qg.get("test", {}).items()]
    if rows:
        story.append(table(["Test photographs", "Ungradable", "AUC", "Ungradable refused", "Gradable accepted"], rows, [56, 28, 22, 32, 32], right=(1, 2, 3, 4)))
    rr = qg.get("reject_rate_by_source", {})
    if rr:
        story.append(P("Refused on held-out photographs: " + "; ".join(f"{k.replace('eyepacs','EyePACS').replace('aptos','APTOS').replace('idrid','IDRiD').replace('messidor','Messidor-2')} {pct(v)}" for k, v in rr.items()) + ". EyePACS contains many genuinely poor photographs; the curated sets almost none.", "small"))

    story += [H("6. Diabetic-retinopathy ensemble", 1)]
    hist = {h["arch"]: h for h in m.get("member_history", [])}
    story.append(table(["Network", "Published", "Input", "Best validation AUC", "Epochs"], [
        [names.get(x["arch"], x["arch"]), years.get(x["arch"], ""), f"{x['input_size']} px", num(hist.get(x["arch"], {}).get("best_auc"), 4), str(hist.get(x["arch"], {}).get("epochs", ""))] for x in m["members"]
    ], [44, 24, 22, 40, 20], right=(2, 3, 4)))
    story += [P("Training procedure", "h3")]
    story += bullets([
        "ImageNet-pretrained backbones, fully fine-tuned; one linear head per network with a referable logit and five grade logits.",
        "Loss: binary cross-entropy (referable) plus 0.5 × cross-entropy with label smoothing 0.05 (grade), with grade-frequency weights.",
        "<b>Source balancing:</b> every source is sampled half referable, half not (EyePACS is about 20% referable and APTOS about 40%), so the camera cannot act as a shortcut to the label.",
        "Augmentation: any rotation, flips, zoom and shift, colour and gamma jitter, and mild blur, glare, haze and under-exposure.",
        f"AdamW (weight decay 0.01), warm-up then cosine learning rate, mixed precision, gradient clipping; {m.get('epochs')} epochs per network on {m.get('train_images', 0):,} photographs (EyeQ “reject” photographs excluded); best-validation weights kept.",
    ])
    story += image(FIGURES / "dr_training.png", 150, "Figure 7. Training loss and validation AUC per epoch for each network.")
    story += [P("Ensemble versus members (AUC, photographs accepted by the gate)", "h3")]
    ma = report.get("member_auc", {})
    if ma:
        archs = list(next(iter(ma.values())).keys())
        story.append(table(["Test set"] + [names.get(a, "Ensemble" if a == "ensemble" else a) for a in archs],
                           [[k.replace(" - external, India", "").replace(" - external", "")] + [("<b>%s</b>" % num(v)) if a == "ensemble" else num(v) for a, v in row.items()] for k, row in ma.items()],
                           [50] + [120 / len(archs)] * len(archs), right=tuple(range(1, len(archs) + 1))))
        story.append(P("The average is at least as good as the best single network on every set.", "caption"))

    story += [H("7. Decision thresholds and site calibration", 1)]
    story += [P(f"The score is a ranking, not a probability: training saw both classes in equal numbers, so a score overstates the chance of disease in a real screening population. A threshold turns it into a decision. The global threshold ({num(m['global_threshold'])}) was tuned on pooled validation photographs for {pct(tg['tuning_sensitivity'], 0)} sensitivity, a margin above the 90% goal, and frozen before testing.")]
    story += [P("Each camera shifts the scores, so one threshold suits some sites and over-refers at others. A site profile is tuned on that site's graded photographs for the same sensitivity and reports what to expect on new photographs: for sites with a validation split it is tuned there and measured on the test split; otherwise it is cross-fitted over 20 random halvings.")]
    story.append(table(["Site", "Threshold", "Photographs (referable)", "Expected sensitivity", "Expected specificity", "Target"], [
        [e(p["name"]), num(p["threshold"]), f"{p['n_images']:,} ({p['n_referable']:,})", pct(p["expected_sensitivity"]), pct(p["expected_specificity"]),
         MET if (p["expected_sensitivity"] > tg["sensitivity"] and p["expected_specificity"] > tg["specificity"]) else NOT_MET] for p in report["sites"] if not p["stale"]
    ], [46, 20, 32, 26, 26, 20], right=(1, 2, 3, 4)))

    story += [H("8. Evaluation", 1)]
    story += [P(f"Referable DR on photographs accepted by the quality gate. Brackets: 95% bootstrap intervals (1,000 resamples). Target: sensitivity above {pct(tg['sensitivity'], 0)} and specificity above {pct(tg['specificity'], 0)}.")]
    rows = []
    for key, x in sets.items():
        g, st = x["global"], x.get("site")
        ci = lambda pair: f"<br/><font size='6.4' color='#545A7A'>{pair[0] * 100:.1f}–{pair[1] * 100:.1f}</font>" if pair else ""
        rows.append([("&nbsp;&nbsp;" if key in ("eyepacs", "aptos") else "") + e(x["title"]), f"{x['graded']:,}", f"{x['referable']:,}", pct(x["rejected_by_gate"]),
                     num(x["auc"]) + (f"<br/><font size='6.4' color='#545A7A'>{x['auc_ci'][0]:.3f}–{x['auc_ci'][1]:.3f}</font>" if x.get("auc_ci") else ""),
                     pct(g["sensitivity"]) + ci(g.get("sensitivity_ci")), pct(g["specificity"]) + ci(g.get("specificity_ci")),
                     pct(st["sensitivity"]) if st else "–", pct(st["specificity"]) if st else "–"])
    story.append(table(["Test set", "Graded", "Referable", "Refused", "AUC", "Sens. global", "Spec. global", "Sens. site", "Spec. site"], rows, [38, 15, 18, 15, 18, 17, 17, 16, 16], right=tuple(range(1, 9))))
    story += image(FIGURES / "roc_and_confusion.png", 150, "Figure 8. ROC curves with the global operating point (left) and the five-grade confusion matrix on the internal test set (right).")
    story += [P("Results by reference grade", "h3"), P("Share referred at each reference grade: for referable grades this is sensitivity, for the others the false-referral rate. Site figures are out of sample.")]
    head = ["Test set"] + [f"{g}" for g in ("No DR", "Mild", "Moderate*", "Severe*", "PDR*")]
    rows = []
    for key in ("internal", "eyepacs", "aptos", "idrid", "messidor"):
        x = sets.get(key)
        if not x or "by_grade" not in x:
            continue
        cells = []
        for gr in x["by_grade"]:
            s = pct(gr["referred_global"], 0) if gr["n"] else "–"
            if gr.get("referred_site") is not None and gr["n"]:
                s += f" / {pct(gr['referred_site'], 0)}"
            cells.append(s)
        rows.append([e(x["title"])] + cells)
    story.append(table(head, rows, [50, 24, 24, 24, 24, 24], right=(1, 2, 3, 4, 5)))
    story.append(P("* referable grades. Each cell: global threshold / site threshold. Severe NPDR and PDR are referred almost always; misses concentrate in moderate NPDR, and false referrals in mild NPDR.", "caption"))
    kap = [(x["title"], x.get("kappa")) for x in sets.values() if x.get("kappa") is not None]
    if kap:
        story.append(P("Five-grade agreement with the reference (quadratic-weighted kappa): " + "; ".join(f"{e(t_)} {num(k, 2)}" for t_, k in kap) + ".", "small"))

    story += [H("9. Explainability: Grad-CAM", 1)]
    story += [P("For each network, Grad-CAM weights the channels of the last convolutional feature map <i>F</i> by the spatial mean of the gradient of the referable logit, keeps the positive part, and normalises to a peak of 1; the three maps are averaged. Because the head after <i>F</i> is global average pooling, an optional LayerNorm (ConvNeXt) and one linear layer, the gradient has a closed form, so the server needs no automatic differentiation:")]
    story += [P("linear head:&nbsp;&nbsp;&nbsp;&nbsp; dz/dF[c,h,w] = W[c] / (H*W)<br/>"
                "LayerNorm head:&nbsp; dz/dp = (g - mean(g) - p_hat * mean(g * p_hat)) / sigma,&nbsp; g = W * gamma<br/>"
                "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; dz/dF[c,h,w] = (dz/dp)[c] / (H*W)", "sig")]
    story += [P("Here <i>z</i> is the referable logit, <i>W</i> the head's weights for it, <i>p</i> the pooled feature vector, <i>p_hat</i> its normalised value, <i>sigma</i> its standard deviation and <i>gamma</i> the LayerNorm scale; products are element by element. Tests check that this equals PyTorch's autograd result to within 1e-4 for all three architecture families.")]
    cam = report.get("gradcam", {})
    if cam.get("trained"):
        story.append(table(["Measure (81 IDRiD photographs with lesion masks)", "Fundalux", "Untrained networks", "Chance"], [
            ["Hottest point on an ophthalmologist-marked lesion", pct(cam["trained"]["pointing hit rate"], 0), pct(cam["baseline"]["pointing hit rate"], 0), "–"],
            ["Heat on lesions ÷ lesion area (median)", num(cam["trained"]["energy on lesions / lesion area (median)"], 2), num(cam["baseline"]["energy on lesions / lesion area (median)"], 2), "1"],
            ["Heatmap as a pixel-level lesion detector (mean AUC)", num(cam["trained"]["pixel AUC (mean)"], 2) + f" ({num(cam['pixel_auc_ci'][0], 2)}–{num(cam['pixel_auc_ci'][1], 2)})", num(cam["baseline"]["pixel AUC (mean)"], 2), "0.50"],
            ["Photographs where Fundalux did better", f"{cam['beats_baseline_on']} of {cam['compared']}", "", ""],
        ], [80, 34, 32, 24], right=(1, 2, 3)))
    story += image(FIGURES / "gradcam_vs_lesion_masks.png", 120, "Figure 9. Best, typical and worst agreement: photograph, ophthalmologists' lesion outlines, heatmap.")
    story += [P("How to read a heatmap", "h3")]
    story += bullets([
        "It marks regions, not lesions: the feature maps are 14 to 16 cells across, each about 1/15 of the image width.",
        "Absence of heat is not absence of disease; it shows what raised this score most.",
        "At a low score it still highlights the least normal-looking areas, which are usually not lesions.",
        "Blue was chosen because it never occurs in a fundus photograph, so it cannot be mistaken for an exudate.",
    ])

    story += [H("10. Robustness to poor photographs", 1)]
    rob = report.get("robustness", [])
    if rob:
        sev = sorted({r["severity"] for r in rob})
        rows = []
        for kind in sorted({r["kind"] for r in rob}):
            cells = []
            for s in sev:
                r = next((r for r in rob if r["kind"] == kind and r["severity"] == s), None)
                cells.append("–" if r is None else f"{pct(r['rejected'], 0)} · {pct(r['spec_no_gate'], 0)}")
            rows.append([kind.capitalize()] + cells)
        story.append(table(["Defect"] + [f"Severity {s:g}" for s in sev], rows, [26] + [144 / len(sev)] * len(sev), right=tuple(range(1, len(sev) + 1))))
        story.append(P("Each cell: share refused by the gate · specificity without the gate, on good test photographs degraded on purpose. Severe defects are refused; moderate ones (around 0.3) pass while specificity falls, so the system over-refers rather than misses. The training notebook now teaches the gate to refuse from severity 0.3 (not yet trained).", "caption"))
    story += image(FIGURES / "robustness_field_conditions.png", 150, "Figure 10. Refusal rate, sensitivity and specificity against defect severity.")

    story += [H("11. Limitations and failure modes", 1)]
    story += bullets([
        "<b>Diabetic macular oedema is not assessed</b>; oedema with mild or no retinopathy will not be referred.",
        "<b>No other eye disease is detected.</b>",
        f"<b>Specificity depends on calibration:</b> {pct(report['summary']['global_spec_min'], 0)}–{pct(report['summary']['global_spec_max'], 0)} at the global threshold; EyePACS-like mixed populations miss 85% even with their own threshold.",
        "<b>Moderately degraded photographs pass the gate and are over-referred.</b>",
        "<b>Not validated on smartphone or portable cameras</b>, the devices most common in community screening.",
        "<b>Retrospective evidence only;</b> no prospective study or regulatory review.",
        "<b>Label noise</b> in EyePACS limits both training and its test figures.",
        "<b>One photograph at a time;</b> no combination of both eyes, two fields or earlier visits.",
        "<b>Recipe chosen after earlier results:</b> run 2's design responded to run 1's test results, so its held-out figures are slightly less independent than a single pre-registered run.",
    ])
    story += [H("12. Ethical considerations", 1)]
    story += bullets([
        "Over-referral costs specialist time and patient travel; under-referral risks sight. Thresholds are set sensitivity-first for that reason.",
        "Performance varies by camera and population; sites must calibrate and monitor before relying on it.",
        "Heatmaps can create unwarranted trust; they were validated against lesion masks and their limits are stated wherever they are shown.",
        "The datasets come from India, the USA and France; other populations are untested.",
    ])
    story += [H("13. Serving and versioning", 1)]
    story += bullets([
        "Each network is exported to ONNX with two outputs (feature map, logits); heads.npz holds the head weights, LayerNorm scale and epsilon for the heatmaps. The training notebook checks the ONNX copies against PyTorch before packaging.",
        "A model fingerprint (members, training run, threshold) is stored with site profiles and review sets; a mismatch marks them stale.",
        "Training history: run 1 (16 Nov 2026), one EfficientNet-B3 at 384 px, specificity 49–73%; run 2 (18 Nov 2026), the current ensemble with source balancing, specificity 76–83% at the global threshold.",
    ])


# ----------------------------------------------------------------------------- reference


def doc_html(text):
    """Docstring to paragraph markup: escape, keep paragraphs, keep indented blocks as code."""
    out = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.splitlines()
        if all(line.startswith("    ") or not line.strip() for line in lines):
            out.append(f"<font name='{MONO}' size='7.2'>" + "<br/>".join(e(line.strip()) for line in lines) + "</font>")
        else:
            out.append(e(" ".join(line.strip() for line in lines)))
    return re.sub(r"`([^`]+)`", rf"<font name='{MONO}' size='7.6'>\1</font>", "<br/><br/>".join(out))


def signature_html(name, obj):
    try:
        sig = str(inspect.signature(obj))
    except (TypeError, ValueError):
        sig = "(...)"
    sig = re.sub(r"(: |-> )'([^']*)'", r"\1\2", sig.replace("NoneType", "None"))  # string annotations from __future__
    return f"<b>{e(name)}</b>{e(sig)}".replace(", ", ",&nbsp;").replace(",&nbsp;", ", ")


MODULES = [
    ("fundalux.preprocess", "Image preprocessing"), ("fundalux.quality", "Quality-gate rules and hints"), ("fundalux.degrade", "Simulated camera defects"),
    ("fundalux.inference", "Inference backends and closed-form Grad-CAM"), ("fundalux.heatmap", "Heatmap drawing"), ("fundalux.pipeline", "End-to-end screening"),
    ("fundalux.sites", "Per-site thresholds"), ("fundalux.calibrate", "Calibrating a site"), ("fundalux.review", "Clinician review"),
    ("fundalux.metrics", "Screening metrics"), ("fundalux.screening_sim", "Screening-camp simulation"), ("fundalux.cli", "Command-line screening"),
    ("fundalux.export", "ONNX export (needs PyTorch)"), ("fundalux.models", "Network definition (needs PyTorch)"), ("fundalux.gradcam", "Grad-CAM in PyTorch and lesion overlap"),
    ("fundalux.data", "Training datasets (needs PyTorch)"), ("fundalux.datasets", "Dataset indexing and caching"), ("fundalux.synthetic", "Synthetic test data"),
    ("server.report", "Evidence report for the web pages"),
]


def module_members(mod):
    items = []
    for name, obj in vars(mod).items():
        if name.startswith("_") or getattr(obj, "__module__", None) != mod.__name__:
            continue
        if inspect.isclass(obj) or inspect.isfunction(obj):
            items.append((name, obj))
    return sorted(items, key=lambda kv: (not inspect.isclass(kv[1]), kv[0]))


def class_block(name, cls):
    out = [Paragraph(f"<font color='#545A7A'>class</font> " + signature_html(name, cls), S["sig"])]
    doc = cls.__dict__.get("__doc__")
    if doc and not doc.startswith(f"{name}("):  # skip the signature dataclasses generate as a docstring
        out.append(P(doc_html(inspect.cleandoc(doc)), "doc"))
    fields = getattr(cls, "__dataclass_fields__", None)
    if fields:
        rows = [[f"<font name='{MONO}' size='7'>{e(f)}</font>", f"<font name='{MONO}' size='7'>{e(str(v.type))}</font>"] for f, v in fields.items()]
        out.append(table(["Field", "Type"], rows, [48, 122]))
    for mname, member in sorted(vars(cls).items()):
        if mname.startswith("_"):
            continue
        fn = member.__func__ if isinstance(member, (classmethod, staticmethod)) else member
        if isinstance(member, property):
            d = inspect.getdoc(member.fget) or ""
            out.append(P(f"<font name='{MONO}' size='7.6'>.{e(mname)}</font> <font color='#545A7A'>(property)</font> {e(d)}", "doc"))
        elif inspect.isfunction(fn):
            kind = "classmethod " if isinstance(member, classmethod) else ""
            out.append(P(f"<font name='{MONO}' size='7.6'>{kind}.{signature_html(mname, fn)}</font>", "doc"))
            d = inspect.getdoc(fn)
            if d:
                out.append(Paragraph(doc_html(d), ParagraphStyle("mdoc", parent=S["doc"], leftIndent=16, textColor=MUTED, fontSize=8.2, leading=11.6)))
    return out


ENDPOINTS = {
    "GET /": "Overview: what Fundalux is, results at a glance, intended use, limitations",
    "GET /how-it-works": "Clinician's guide and model card",
    "GET /evidence": "ROC curves, results by grade, confusion matrices, heatmap validation, robustness",
    "GET /screen": "Screening tool",
    "GET /sites": "Site thresholds; calibrating a new site",
    "GET /camp": "Screening-camp model",
    "GET /review": "Blinded grading for clinicians",
    "GET /review/results": "Organiser's results page (data loaded with the admin token)",
    "GET /figures/{name}": "A PNG figure from models/figures; any other path returns 404",
    "GET /samples/{name}": "One of the bundled sample photographs (fixed list)",
    "GET /api/health": "Whether the model bundle loaded, the backend in use (onnx or torch), and any load error",
    "GET /api/report": "All evidence-page data as JSON",
    "POST /api/screen": "Screen one photograph (form fields below)",
    "GET /api/sites": "Current site profiles, with stale ones marked",
    "POST /api/sites/calibrate": "Upload graded photographs and a label file; starts a calibration job",
    "GET /api/jobs/{job_id}": "Progress and result of a calibration or review-set job",
    "POST /api/jobs/{job_id}/save": "Save a finished calibration as a site profile",
    "DELETE /api/sites/{name}": "Remove a calibrated site (built-in sites cannot be removed)",
    "POST /api/camp": "Run the camp simulation and staffing sweep",
    "GET /api/camp/defaults": "Default camp parameters, from the model's measured rates",
    "GET /api/review/next": "The reviewer's next unanswered photograph; ?skip= passes over one",
    "GET /api/review/image/{item_id}": "A review photograph, by neutral id",
    "POST /api/review/answer": "Record a reviewer's answer",
    "GET /api/review/results": "Agreement analysis; ?site= selects the threshold",
    "GET /api/review/export.csv": "Every answer with reference and model output, as CSV",
    "GET /api/review/thumb/{item_id}": "Thumbnail for the disagreement gallery",
    "POST /api/review/set": "Build a new review set from uploaded photographs; starts a job",
}


def part_reference(story):
    from fastapi.routing import APIRoute

    import server.main as srv

    story += [PartMarker("Part IV · Function and API reference"), PageBreak(), P("PART IV", "kicker"), H("Function and API reference", 0)]
    story += [P("Generated from the running code: routes from the FastAPI application, signatures and descriptions from the Python package. Interactive API documentation is also served at <font name='%s'>/api/docs</font>." % MONO)]

    story += [H("1. HTTP endpoints", 1)]
    rows = []
    for r in srv.app.routes:
        if not isinstance(r, APIRoute):
            continue
        deps = {getattr(d.call, "__name__", "") for d in r.dependant.dependencies}
        auth = "admin token" if "require_admin" in deps else "review code" if "require_reviewer" in deps else "public"
        doc = (inspect.getdoc(r.endpoint) or "").split("\n\n")[0].replace("\n", " ") or ENDPOINTS.get(f"{'/'.join(sorted(r.methods - {'HEAD'}))} {r.path}", "")
        kind = "API" if r.path.startswith("/api") else "file" if r.path.startswith(("/figures", "/samples")) else "page"
        rows.append([f"<font name='{MONO}' size='7'>{'/'.join(sorted(r.methods - {'HEAD'}))}</font>", f"<font name='{MONO}' size='7'>{e(r.path)}</font>", kind, auth, e(doc) or "–"])
    rows.sort(key=lambda r_: (r_[2] != "page", r_[1]))
    story.append(table(["Method", "Path", "Kind", "Access", "Description"], rows, [16, 50, 12, 20, 72]))
    story += [P("Authentication: send <font name='%s'>Authorization: Bearer &lt;token&gt;</font>. Tokens are compared in constant time. Without a configured token, the corresponding features are switched off." % MONO, "small")]
    story += [P("POST /api/screen", "h3")]
    story.append(table(["Form field", "Type", "Meaning"], [
        ["photo", "file", "The photograph (JPEG, PNG, TIFF, BMP), at most FUNDALUX_MAX_UPLOAD_MB"],
        ["sample_name", "string", "Instead of a photo: one of the bundled IDRiD samples"],
        ["degrade_kind", "string", "none, blur, motion, glare, haze or dark (demonstration)"],
        ["degrade_severity", "0–1", "Severity of the simulated defect"],
        ["grade_rejected", "bool", "Grade even if the gate refuses (result stays marked not gradable)"],
    ], [34, 18, 118]))
    story += [Spacer(1, 5), P("Response: gradable, reasons, ungradable_score, quality_threshold, referable_score, member_scores, grade, grade_name, grade_probs, decisions (one per threshold: site, threshold, referable, advice, expected sensitivity and specificity), images (cropped, enhanced, heatmap as JPEG data URIs), seconds, model, disclaimer. Errors: 413 too large, 422 unreadable or invalid, 503 busy.", "small")]

    story += [H("2. Configuration", 1)]
    story.append(table(["Variable", "Default", "Meaning"], [
        [code("FUNDALUX_ADMIN_TOKEN"), "unset", "Enables calibration, review results and new review sets"],
        [code("FUNDALUX_REVIEW_TOKEN"), "unset", "Code given to clinicians for /review"],
        [code("FUNDALUX_MODELS"), "models", "Model bundle folder"],
        [code("FUNDALUX_REVIEW_DIR"), "review", "Review set and answers"],
        [code("FUNDALUX_MAX_UPLOAD_MB"), "20", "Largest photograph accepted"],
        [code("FUNDALUX_CONCURRENCY"), "2", "Photographs screened at once"],
        [code("FUNDALUX_THREADS"), "auto", "ONNX Runtime threads per screening"],
        [code("FUNDALUX_BACKEND"), "auto", "onnx or torch; auto prefers ONNX when the bundle has it"],
        [code("PORT"), "8000", "Port inside the container"],
    ], [50, 20, 100]))

    story += [H("3. Command-line tools", 1)]
    story.append(table(["Command", "Purpose"], [
        [code("python -m fundalux.cli PHOTO [--models DIR] [--save-dir OUT]"), "Screen one photograph; JSON result; optional images"],
        [code("python -m fundalux.calibrate --site NAME --labels CSV --images DIR"), "Calibrate a site threshold from graded photographs"],
        [code("python -m fundalux.export [MODEL_DIR]"), "Export a bundle to ONNX for serving (needs PyTorch)"],
        [code("python -m fundalux.screening_sim [--metrics FILE]"), "Camp simulation and staffing sweep in the terminal"],
        [code("python scripts/build_site_profiles.py"), "Rebuild built-in site profiles from reports/predictions.csv"],
        [code("python scripts/build_review_set.py --images DIR --labels CSV"), "Build the clinician review set"],
        [code("python scripts/build_web_assets.py"), "Re-render the illustrations from the current models"],
        [code("python scripts/build_notebook.py [--check]"), "Regenerate (or check) the training notebook"],
        [code("python scripts/build_docs_pdf.py"), "Regenerate this document"],
    ], [92, 78]))

    story += [H("4. Python package", 1)]
    story += [P("Public classes and functions of each module, in the order a photograph meets them. Modules marked “needs PyTorch” are used for training and export only; the server imports none of them.")]
    for modname, title in MODULES:
        mod = importlib.import_module(modname)
        block = [CondPageBreak(60 * mm), Paragraph(f"<font name='{MONO}'>{modname}</font> &nbsp;<font color='#545A7A' size='9'>{e(title)}</font>", S["h2"])]
        block[-1].toc_level = 1
        mdoc = inspect.getdoc(mod)
        if mdoc:
            block.append(P(doc_html(mdoc), "doc"))
        story += block
        for name, obj in module_members(mod):
            if inspect.isclass(obj):
                story += class_block(name, obj)
            else:
                story.append(Paragraph(signature_html(name, obj), S["sig"]))
                d = inspect.getdoc(obj)
                if d:
                    story.append(P(doc_html(d), "doc"))

    story += [H("5. Test suite", 1)]
    by_file: dict[str, list[str]] = {}
    for name, f in TESTS.items():
        by_file.setdefault(f, []).append(name)
    story.append(table(["File", "Tests", "Covers"], [
        [f"<font name='{MONO}' size='7'>{f}</font>", str(len(v)), {
            "test_core.py": "preprocessing, degradations, metrics, models, Grad-CAM, pipeline, sites in the pipeline",
            "test_datasets.py": "dataset indexing, splits, caching, training datasets",
            "test_inference.py": "closed-form Grad-CAM, ONNX export equivalence, backend choice",
            "test_sites.py": "cross-fitting, calibration, registry, staleness, label parsing",
            "test_review.py": "review sets, blinding, storage, analysis, review HTTP flow",
            "test_server.py": "pages, screening API, input validation, security, calibration API",
            "test_app.py": "the Streamlit app end to end",
            "test_simulation.py": "camp simulation",
            "test_notebook.py": "the notebook is generated and self-contained",
        }.get(f, "")] for f, v in sorted(by_file.items())
    ], [40, 14, 116], right=(1,)))
    story.append(P(f"{len(TESTS)} test functions; with parameterisation pytest runs 79 cases.", "caption"))


# ----------------------------------------------------------------------------- build


def main() -> None:
    collect_tests()
    config, report, version = load()
    trained = config.get("trained_with", {})
    meta = {"date": dt.date.today().strftime("%d %B %Y"), "version": version, "run": "run 2 · three-network ensemble", "trained": trained.get("date", "–")}
    doc = Doc(OUT)
    doc.meta = meta
    doc.part = ""
    story = [Spacer(1, 1)]
    front_matter(meta, story)
    part_overview(config, report, story)
    part_srs(config, report, story)
    part_model(config, report, story)
    part_reference(story)
    doc.multiBuild(story)
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
