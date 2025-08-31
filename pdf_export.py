# pdf_export.py — v4.1 (verified-only lists + visual hints separated)
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
    if not text: return y
    lines = textwrap.wrap(text, width=width_chars)
    if max_lines: lines = lines[:max_lines]
    for ln in lines:
        c.drawString(x, y, ln); y -= leading
    return y

def build_pdf(
    *, brand_header: str, footer_note: str,
    product_meta: Dict[str,str],
    analysis: Dict[str,Any],
    plan: Dict[str,Any],
    phase_images: List[Tuple[str, Image.Image]],
    research: Dict[str,Any] | None = None,
    extra_keyframes: List[Tuple[str, Image.Image]] | None = None,
) -> bytes:
    buf = BytesIO(); c = canvas.Canvas(buf, pagesize=A4)
    W,H = A4; M=2*cm

    # cover
    c.setFont("Helvetica-Bold", 20); c.drawString(M, H-M, f"{brand_header} – Content Brief")
    c.setFont("Helvetica", 12); c.drawString(M, H-M-20, f"Product: {product_meta.get('brand','')} {product_meta.get('name','')}")
    arch = analysis.get("archetype") or analysis.get("analysis",{}).get("archetype","")
    gs = analysis.get("global_signals",{})
    c.drawString(M, H-M-40, f"Archetype: {arch}")
    c.drawString(M, H-M-55, f"Speech: {gs.get('speech_presence','')}  |  Tempo: {gs.get('tempo','')}")
    c.showPage()

    # product snapshot (verified only)
    if research:
        y = H-M; c.setFont("Helvetica-Bold", 16); c.drawString(M, y, "Product Snapshot (verified)"); y-=24
        specs_d = research.get("specs_detailed") or []
        feats_d = research.get("features_detailed") or []

        c.setFont("Helvetica-Bold", 12); c.drawString(M, y, "Specs:"); y-=14; c.setFont("Helvetica",10)
        if specs_d:
            for s in specs_d[:16]:
                line = f"• {s['key'].title()}: {s['value']}  (conf {s['confidence']:.2f}, {len(s['sources'])} src)"
                y = _wrap(c, line, M+10, y, width_chars=100)
                if y < M+60: c.showPage(); y=H-M; c.setFont("Helvetica",10)
        else:
            y = _wrap(c, "• (none verified)", M+10, y, width_chars=100)

        y -= 10
        c.setFont("Helvetica-Bold", 12); c.drawString(M, y, "Features:"); y-=14; c.setFont("Helvetica",10)
        if feats_d:
            for f in feats_d[:18]:
                line = f"• {f['text']}  (conf {f['confidence']:.2f}, {len(f['sources'])} src)"
                y = _wrap(c, line, M+10, y, width_chars=100)
                if y < M+60: c.showPage(); y=H-M; c.setFont("Helvetica",10)
        else:
            y = _wrap(c, "• (none verified)", M+10, y, width_chars=100)

        # visual hints — never verified
        hints = research.get("visual_hints") or []
        if hints:
            y -= 6
            c.setFont("Helvetica-Bold", 12); c.drawString(M, y, "Visual hints (not verified):"); y-=14
            c.setFont("Helvetica",10)
            y = _wrap(c, "• " + " • ".join(hints[:16]), M+10, y, width_chars=100)

        # sources that contributed
        srcs = research.get("sources") or []
        if srcs:
            y -= 6; c.setFont("Helvetica-Bold", 12); c.drawString(M, y, "Sources (used):"); y-=14
            c.setFont("Helvetica", 9)
            for s in srcs[:10]:
                y = _wrap(c, f"• {s.get('url','')}", M+10, y, width_chars=110)
                if y < M+60: c.showPage(); y=H-M; c.setFont("Helvetica",9)

        c.showPage()

    # reference frames by phase
    if phase_images:
        c.setFont("Helvetica-Bold", 16); c.drawString(M, H-M, "Reference Keyframes (by phase)")
        y = H-M-30
        img_w = (W - 3*M) / 2; img_h = img_w * 9/16
        for i,(label,img) in enumerate(phase_images):
            if y < M + img_h:
                c.showPage(); c.setFont("Helvetica-Bold",16)
                c.drawString(M, H-M, "Reference Keyframes (by phase, cont.)")
                y = H-M-30
            x = M if i%2==0 else M + img_w + M
            im = img.copy().resize((int(img_w), int(img_h)))
            c.drawImage(ImageReader(im), x, y-img_h, img_w, img_h)
            c.setFont("Helvetica",10); c.drawString(x, y-img_h-12, label)
            if i%2==1: y -= (img_h + 36)
        c.showPage()

    # dense storyboard
    if extra_keyframes:
        c.setFont("Helvetica-Bold", 16); c.drawString(M, H-M, "Additional Reference Frames")
        y = H-M-30; cols=3; img_w=(W - (cols+1)*M)/cols; img_h=img_w*9/16
        for i,(label,img) in enumerate(extra_keyframes):
            if y < M + img_h:
                c.showPage(); c.setFont("Helvetica-Bold",16)
                c.drawString(M, H-M, "Additional Reference Frames (cont.)")
                y = H-M-30
            col = i % cols; x = M + col*(img_w+M)
            im = img.copy().resize((int(img_w), int(img_h)))
            c.drawImage(ImageReader(im), x, y-img_h, img_w, img_h)
            c.setFont("Helvetica",9); c.drawString(x, y-img_h-10, label)
            if col==cols-1: y -= (img_h + 28)
        c.showPage()

    # shot plan
    c.setFont("Helvetica-Bold", 16)
    c.drawString(M, H-M, (plan.get("title") or f"{product_meta.get('name','')} — Shot Plan"))
    c.setFont("Helvetica",11)
    c.drawString(M, H-M-18, f"Target duration: {plan.get('target_runtime_s','')} s")
    y = H-M-40

    for sc in (plan.get("script") or {}).get("scenes", []):
        if y < M+120: c.showPage(); y=H-M-20
        c.setFont("Helvetica-Bold", 12)
        c.drawString(M, y, f"• Scene {sc.get('idx','')}  ({sc.get('start_s','')}–{sc.get('end_s','')}s)")
        y -= 14
        c.setFont("Helvetica", 10)
        def _sj(x): return " / ".join(x) if isinstance(x,list) else str(x or "")
        y = _wrap(c, f"Action: {_sj(sc.get('action',''))}", M+12, y, width_chars=100)
        y = _wrap(c, f"Camera: {_sj(sc.get('camera',''))}", M+12, y, width_chars=100)
        if sc.get("on_screen_text"):
            y = _wrap(c, "OSD: " + _sj(sc.get('on_screen_text')), M+12, y, width_chars=100)
        if sc.get("voiceover"):
            y = _wrap(c, f"VO: {sc.get('voiceover')}", M+12, y, width_chars=100)
        y -= 6

    c.showPage()
    c.setFont("Helvetica-Bold", 16); c.drawString(M, H-M, "CTA")
    c.setFont("Helvetica", 11); c.drawString(M, H-M-18, ((plan.get("ctas") or {}).get("hard") or ""))
    c.setFont("Helvetica-Oblique", 9); c.drawString(M, M, footer_note)
    c.save(); buf.seek(0); return buf.read()
