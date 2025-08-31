# pdf_export.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from io import BytesIO
from PIL import Image
import textwrap

def _wrap(c, text, x, y, width_chars=95, leading=12, max_lines=None):
    if not text:
        return y
    lines = textwrap.wrap(text, width=width_chars)
    if max_lines:
        lines = lines[:max_lines]
    for ln in lines:
        c.drawString(x, y, ln)
        y -= leading
    return y

def _kv(c, kv: dict, x, y):
    for k, v in kv.items():
        c.drawString(x, y, f"{k}: {v}")
        y -= 12
    return y

def build_pdf(
    *,
    brand_header: str,
    footer_note: str,
    product_meta: Dict[str, str],
    analysis: Dict[str, Any],
    plan: Dict[str, Any],
    phase_images: List[Tuple[str, Image.Image]],
    research: Dict[str, Any] | None = None,
) -> bytes:
    """Return PDF bytes."""
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    margin = 2 * cm

    # Cover
    c.setFont("Helvetica-Bold", 20)
    c.drawString(margin, H - margin, f"{brand_header} – Content Brief")
    c.setFont("Helvetica", 12)
    c.drawString(margin, H - margin - 20, f"Product: {product_meta.get('brand','')} {product_meta.get('name','')}")
    arch = analysis.get("archetype") or analysis.get("analysis", {}).get("archetype", "")
    gs = analysis.get("global_signals", {})
    c.drawString(margin, H - margin - 40, f"Archetype: {arch}")
    c.drawString(margin, H - margin - 55, f"Speech: {gs.get('speech_presence','')}  |  Tempo: {gs.get('tempo','')}")
    c.showPage()

    # Research (if any)
    if research and (research.get("claims") or research.get("features")):
        c.setFont("Helvetica-Bold", 16)
        c.drawString(margin, H - margin, "Web Research (safe feature claims)")
        y = H - margin - 20
        c.setFont("Helvetica", 11)
        y -= 8
        y = _wrap(c, f"Query: {research.get('query','')}", margin, y)
        y -= 6
        if research.get("features"):
            c.setFont("Helvetica-Bold", 12); c.drawString(margin, y, "Features:"); y -= 14
            c.setFont("Helvetica", 10)
            for ft in research["features"][:15]:
                y = _wrap(c, f"• {ft}", margin+10, y, width_chars=100)
                if y < margin + 40:
                    c.showPage(); y = H - margin
        if research.get("claims"):
            c.setFont("Helvetica-Bold", 12); c.drawString(margin, y, "Claims:"); y -= 14
            c.setFont("Helvetica", 10)
            for cl in research["claims"][:15]:
                y = _wrap(c, f"• {cl}", margin+10, y, width_chars=100)
                if y < margin + 40:
                    c.showPage(); y = H - margin
        if research.get("disclaimers"):
            c.setFont("Helvetica-Bold", 12); c.drawString(margin, y, "Disclaimers:"); y -= 14
            c.setFont("Helvetica", 10)
            for d in research["disclaimers"][:10]:
                y = _wrap(c, f"• {d}", margin+10, y, width_chars=100)
                if y < margin + 40:
                    c.showPage(); y = H - margin
        c.showPage()

    # Keyframes
    if phase_images:
        c.setFont("Helvetica-Bold", 16)
        c.drawString(margin, H - margin, "Reference Keyframes")
        y = H - margin - 30
        img_w = (W - 3 * margin) / 2
        img_h = img_w * 9 / 16
        for i, (label, img) in enumerate(phase_images):
            if y < margin + img_h:
                c.showPage()
                c.setFont("Helvetica-Bold", 16)
                c.drawString(margin, H - margin, "Reference Keyframes (cont.)")
                y = H - margin - 30
            x = margin if i % 2 == 0 else margin + img_w + margin
            im = img.copy().resize((int(img_w), int(img_h)))
            c.drawImage(ImageReader(im), x, y - img_h, img_w, img_h)
            c.setFont("Helvetica", 10)
            c.drawString(x, y - img_h - 12, label)
            if i % 2 == 1:
                y -= (img_h + 36)
        c.showPage()

    # Shot plan / Script
    c.setFont("Helvetica-Bold", 16)
    tit = plan.get("title") or f"{product_meta.get('name','')} — Shot Plan"
    c.drawString(margin, H - margin, tit)
    c.setFont("Helvetica", 11)
    c.drawString(margin, H - margin - 18, f"Target duration: {plan.get('target_runtime_s','')} s")
    y = H - margin - 40

    for sc in (plan.get("script") or {}).get("scenes", []):
        if y < margin + 120:
            c.showPage()
            y = H - margin - 20
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, f"• Scene {sc.get('idx','')}  ({sc.get('start_s','')}–{sc.get('end_s','')}s)")
        y -= 14
        c.setFont("Helvetica", 10)
        y = _wrap(c, f"Action: {sc.get('action','')}", margin+12, y, width_chars=100)
        y = _wrap(c, f"Camera: {sc.get('camera','')}", margin+12, y, width_chars=100)
        osd = sc.get("on_screen_text") or []
        if osd:
            y = _wrap(c, "OSD: " + " / ".join(osd), margin+12, y, width_chars=100)
        vo = sc.get("voiceover", "")
        if vo:
            y = _wrap(c, f"VO: {vo}", margin+12, y, width_chars=100)
        y -= 6

    # CTA + footer
    c.showPage()
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, H - margin, "CTA")
    c.setFont("Helvetica", 11)
    ct = ((plan.get("ctas") or {}).get("hard")) or ""
    c.drawString(margin, H - margin - 18, ct if ct else "(optional)")
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(margin, margin, footer_note)

    c.save()
    buf.seek(0)
    return buf.read()
