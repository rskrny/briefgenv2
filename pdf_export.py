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

def build_pdf(
    *,
    brand_header: str,
    footer_note: str,
    product_meta: Dict[str, str],
    analysis: Dict[str, Any],
    plan: Dict[str, Any],
    phase_images: List[Tuple[str, Image.Image]],
    research: Dict[str, Any] | None = None,
    extra_keyframes: List[Tuple[str, Image.Image]] | None = None,
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

    # Product Snapshot
    if research:
        c.setFont("Helvetica-Bold", 16)
        c.drawString(margin, H - margin, "Product Snapshot")
        y = H - margin - 22

        # Verified specs
        specs_d = research.get("specs_detailed") or []
        if specs_d:
            c.setFont("Helvetica-Bold", 12); c.drawString(margin, y, "Specs (verified):"); y -= 14
            c.setFont("Helvetica", 10)
            for s in specs_d[:16]:
                line = f"• {s.get('key','').title()}: {s.get('value','')}  (conf {s.get('confidence',0):.2f}, {len(s.get('sources',[]))} src)"
                y = _wrap(c, line, margin + 10, y, width_chars=100)
                if y < margin + 40:
                    c.showPage(); y = H - margin
                    c.setFont("Helvetica-Bold", 12); c.drawString(margin, y, "Specs (verified, cont.)"); y -= 14
                    c.setFont("Helvetica", 10)
        elif research.get("specs"):
            c.setFont("Helvetica-Bold", 12); c.drawString(margin, y, "Specs:"); y -= 14
            c.setFont("Helvetica", 10)
            for k, v in list(research["specs"].items())[:16]:
                y = _wrap(c, f"• {k.title()}: {v}", margin + 10, y, width_chars=100)
                if y < margin + 40:
                    c.showPage(); y = H - margin
                    c.setFont("Helvetica-Bold", 12); c.drawString(margin, y, "Specs (cont.)"); y -= 14
                    c.setFont("Helvetica", 10)

        # Verified features
        feats_d = research.get("features_detailed") or []
        if feats_d:
            y -= 6
            c.setFont("Helvetica-Bold", 12); c.drawString(margin, y, "Features (verified):"); y -= 14
            c.setFont("Helvetica", 10)
            for f in feats_d[:18]:
                line = f"• {f.get('text','')}  (conf {f.get('confidence',0):.2f}, {len(f.get('sources',[]))} src)"
                y = _wrap(c, line, margin + 10, y, width_chars=100)
                if y < margin + 40:
                    c.showPage(); y = H - margin
                    c.setFont("Helvetica-Bold", 12); c.drawString(margin, y, "Features (verified, cont.)"); y -= 14
                    c.setFont("Helvetica", 10)
        elif research.get("features"):
            y -= 6
            c.setFont("Helvetica-Bold", 12); c.drawString(margin, y, "Features:"); y -= 14
            c.setFont("Helvetica", 10)
            for ft in research.get("features", [])[:18]:
                y = _wrap(c, f"• {ft}", margin + 10, y, width_chars=100)
                if y < margin + 40:
                    c.showPage(); y = H - margin
                    c.setFont("Helvetica-Bold", 12); c.drawString(margin, y, "Features (cont.)"); y -= 14
                    c.setFont("Helvetica", 10)

        # Disclaimers
        ds = research.get("disclaimers") or []
        if ds:
            y -= 6
            c.setFont("Helvetica-Bold", 12); c.drawString(margin, y, "Disclaimers:"); y -= 14
            c.setFont("Helvetica", 10)
            for s in ds[:10]:
                y = _wrap(c, f"• {s}", margin + 10, y, width_chars=100)
                if y < margin + 40:
                    c.showPage(); y = H - margin
                    c.setFont("Helvetica-Bold", 12); c.drawString(margin, y, "Disclaimers (cont.)"); y -= 14
                    c.setFont("Helvetica", 10)

        # Sources (top)
        srcs = research.get("sources") or []
        if srcs:
            y -= 6
            c.setFont("Helvetica-Bold", 12); c.drawString(margin, y, "Sources:"); y -= 14
            c.setFont("Helvetica", 9)
            for s in srcs[:10]:
                title = s.get("title") or s.get("url") or ""
                url = s.get("url") or ""
                y = _wrap(c, f"• {title} — {url}", margin + 10, y, width_chars=110)
                if y < margin + 40:
                    c.showPage(); y = H - margin
                    c.setFont("Helvetica-Bold", 12); c.drawString(margin, y, "Sources (cont.)"); y -= 14
                    c.setFont("Helvetica", 9)

        c.showPage()

    # Reference keyframes (by phase)
    if phase_images:
        c.setFont("Helvetica-Bold", 16)
        c.drawString(margin, H - margin, "Reference Keyframes (by phase)")
        y = H - margin - 30
        img_w = (W - 3 * margin) / 2
        img_h = img_w * 9 / 16
        for i, (label, img) in enumerate(phase_images):
            if y < margin + img_h:
                c.showPage()
                c.setFont("Helvetica-Bold", 16)
                c.drawString(margin, H - margin, "Reference Keyframes (by phase, cont.)")
                y = H - margin - 30
            x = margin if i % 2 == 0 else margin + img_w + margin
            im = img.copy().resize((int(img_w), int(img_h)))
            c.drawImage(ImageReader(im), x, y - img_h, img_w, img_h)
            c.setFont("Helvetica", 10)
            c.drawString(x, y - img_h - 12, label)
            if i % 2 == 1:
                y -= (img_h + 36)
        c.showPage()

    # Additional frames (denser storyboard)
    if extra_keyframes:
        c.setFont("Helvetica-Bold", 16)
        c.drawString(margin, H - margin, "Additional Reference Frames")
        y = H - margin - 30
        cols = 3
        img_w = (W - (cols + 1) * margin) / cols
        img_h = img_w * 9 / 16
        for i, (label, img) in enumerate(extra_keyframes):
            if y < margin + img_h:
                c.showPage()
                c.setFont("Helvetica-Bold", 16)
                c.drawString(margin, H - margin, "Additional Reference Frames (cont.)")
                y = H - margin - 30
            col = i % cols
            x = margin + col * (img_w + margin)
            im = img.copy().resize((int(img_w), int(img_h)))
            c.drawImage(ImageReader(im), x, y - img_h, img_w, img_h)
            c.setFont("Helvetica", 9)
            c.drawString(x, y - img_h - 10, label)
            if col == cols - 1:
                y -= (img_h + 28)
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
            c.showPage(); y = H - margin - 20
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, f"• Scene {sc.get('idx','')}  ({sc.get('start_s','')}–{sc.get('end_s','')}s)")
        y -= 14
        c.setFont("Helvetica", 10)
        def _safe_join(x):
            if isinstance(x, list):
                return " / ".join([str(i) for i in x])
            return str(x or "")
        y = _wrap(c, f"Action: {_safe_join(sc.get('action',''))}", margin+12, y, width_chars=100)
        y = _wrap(c, f"Camera: {_safe_join(sc.get('camera',''))}", margin+12, y, width_chars=100)
        osd = sc.get("on_screen_text") or []
        if osd:
            y = _wrap(c, "OSD: " + _safe_join(osd), margin+12, y, width_chars=100)
        vo = sc.get("voiceover","")
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
