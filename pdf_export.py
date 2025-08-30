# pdf_export.py
# Polished PDF with: cover, claims/disclaimers, narrative map, influencer DNA,
# beat grid, keyframes (embedded w/ captions), style-transfer mapping, scene table, CTAs, checklist, evidence quotes.

from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
import os

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


def _mini(text: str, n: int = 140) -> str:
    if text is None:
        return ""
    s = str(text).strip()
    return s if len(s) <= n else s[: n - 1] + "…"

def _basename(p: str) -> str:
    try:
        return os.path.basename(p)
    except Exception:
        return p

def _resolve_image_path(image_ref: str, keyframe_images: List[Dict], kf_lookup: Dict[str, Dict] | None) -> str | None:
    # try lookup dict first (basename->full path)
    if kf_lookup:
        rec = kf_lookup.get(image_ref)
        if rec and os.path.exists(rec.get("path", "")):
            return rec["path"]
    # else try to match by basename in provided list
    for fr in keyframe_images or []:
        if _basename(fr.get("path", "")) == image_ref and os.path.exists(fr.get("path", "")):
            return fr["path"]
    return None

def _add_table(story, data, col_sizes=None, font_size=9, header_bg=colors.lightgrey):
    table = Table(data, colWidths=col_sizes, repeatRows=1) if col_sizes else Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), header_bg),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("FONTSIZE", (0,0), (-1,-1), font_size),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

def make_brief_pdf(
    out_path: str,
    analyzer: Dict[str, Any],
    script: Dict[str, Any],
    product_facts: Dict[str, Any],
    keyframe_images: List[Dict],
    title: str = "Influencer Brief",
    research: Dict[str, Any] | None = None,
    kf_lookup: Dict[str, Dict] | None = None,
) -> bytes:
    doc = SimpleDocTemplate(out_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story: List = []

    # ---------- Cover ----------
    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 16))
    story.append(Paragraph(f"Brand: {product_facts.get('brand','')}", styles["Normal"]))
    story.append(Paragraph(f"Product: {product_facts.get('product','')}", styles["Normal"]))
    dur = analyzer.get("video_metadata", {}).get("duration_s")
    if dur:
        story.append(Paragraph(f"Reference duration: {dur:.2f}s", styles["Normal"]))
    story.append(Spacer(1, 16))

    story.append(Paragraph("Approved Claims", styles["Heading3"]))
    for c in product_facts.get("approved_claims", []):
        story.append(Paragraph(f"• {_mini(c, 160)}", styles["Normal"]))
    story.append(Spacer(1, 10))

    reqd = product_facts.get("required_disclaimers", []) or []
    if reqd:
        story.append(Paragraph("Required Disclaimers", styles["Heading3"]))
        for d in reqd:
            story.append(Paragraph(f"• {_mini(d, 180)}", styles["Normal"]))
    story.append(PageBreak())

    # ---------- Narrative Map ----------
    story.append(Paragraph("Narrative Phases (Hook → Pain → Solution → Proof → CTA)", styles["Heading2"]))
    ndata = [["Phase", "Start", "End", "Purpose"]]
    for n in analyzer.get("narrative", []):
        ndata.append([n.get("phase",""), f"{n.get('start_s','')}s", f"{n.get('end_s','')}s", _mini(n.get("purpose",""), 120)])
    _add_table(story, ndata, font_size=9)
    story.append(Spacer(1, 6))

    # ---------- Influencer DNA & Edit Grammar ----------
    story.append(Paragraph("Influencer DNA", styles["Heading2"]))
    dna = analyzer.get("influencer_DNA", {}) or {}
    for k in ["persona","energy","tone","camera_presence"]:
        if dna.get(k):
            story.append(Paragraph(f"{k.replace('_',' ').title()}: {_mini(dna[k], 200)}", styles["Normal"]))
    story.append(Spacer(1, 8))
    eg = dna.get("edit_grammar", []) or []
    rd = dna.get("retention_devices", []) or []
    if eg:
        story.append(Paragraph("Edit Grammar", styles["Heading3"]))
        for e in eg:
            story.append(Paragraph(f"• {_mini(e, 120)}", styles["Normal"]))
    if rd:
        story.append(Paragraph("Retention Devices", styles["Heading3"]))
        for r in rd:
            story.append(Paragraph(f"• {_mini(r, 120)}", styles["Normal"]))
    story.append(PageBreak())

    # ---------- Beat Grid ----------
    story.append(Paragraph("Beat Grid", styles["Heading2"]))
    bdata = [["t (s)", "Type"]]
    for b in analyzer.get("beats", []) or []:
        bdata.append([f"{b.get('t_s','')}", b.get("type","")])
    _add_table(story, bdata, font_size=9)

    # ---------- Keyframes with Captions ----------
    story.append(Paragraph("Keyframes (what to capture)", styles["Heading2"]))
    for k in analyzer.get("keyframes", []) or []:
        image_ref = k.get("image_ref") or ""
        why = k.get("why") or ""
        img_path = _resolve_image_path(image_ref, keyframe_images, kf_lookup)
        story.append(Paragraph(f"{image_ref} @ {k.get('t_s','?')}s — {_mini(why, 160)}", styles["Heading3"]))
        if img_path and os.path.exists(img_path):
            story.append(Image(img_path, width=250, height=444))  # scale for 9:16
        else:
            story.append(Paragraph("(image not found in session — ensure ingestion keyframes exist)", styles["Italic"]))
        story.append(Spacer(1, 12))
    story.append(PageBreak())

    # ---------- Style-Transfer Mapping ----------
    story.append(Paragraph("Style-Transfer Mapping", styles["Heading2"]))
    stf = script.get("style_transfer", {}) or {}
    pres = stf.get("preserve", []) or []
    adap = stf.get("adapt", []) or []
    amap = stf.get("affordance_map", []) or []
    if pres:
        story.append(Paragraph("Preserve", styles["Heading3"]))
        for p in pres:
            story.append(Paragraph(f"• {_mini(p, 120)}", styles["Normal"]))
    if adap:
        story.append(Paragraph("Adapt", styles["Heading3"]))
        for a in adap:
            story.append(Paragraph(f"• {_mini(a, 120)}", styles["Normal"]))
    if amap:
        story.append(Paragraph("Affordance Map", styles["Heading3"]))
        for m in amap:
            story.append(Paragraph(f"• {_mini(m.get('from',''), 80)} → {_mini(m.get('to',''), 120)}", styles["Normal"]))
    story.append(PageBreak())

    # ---------- Scene Table ----------
    story.append(Paragraph("Scenes (timestamped)", styles["Heading2"]))
    sdata = [["#", "Time", "Camera", "Action", "Dialogue", "On-Screen Text", "Transition", "SFX/Music", "Retention"]]
    for s in script.get("script", {}).get("scenes", []) or []:
        time_str = f"{s.get('start_s',0)}–{s.get('end_s',0)}s"
        osd = "\n".join((s.get("on_screen_text", []) or [])[:2])
        sdata.append([
            s.get("idx",""),
            time_str,
            _mini(s.get("camera",""), 60),
            _mini(s.get("action",""), 120),
            _mini(s.get("dialogue",""), 160),
            _mini(osd, 80),
            _mini(s.get("transition_out",""), 40),
            _mini(s.get("sfx_or_music",""), 60),
            _mini(s.get("retention_note",""), 80),
        ])
    _add_table(story, sdata, font_size=8)
    story.append(PageBreak())

    # ---------- CTAs & Checklist ----------
    story.append(Paragraph("CTAs", styles["Heading2"]))
    for cta in script.get("script", {}).get("cta_options", []) or []:
        story.append(Paragraph(f"{cta.get('variant')}: {cta.get('line')}", styles["Normal"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Checklist", styles["Heading2"]))
    for chk in script.get("checklist", []) or []:
        story.append(Paragraph(f"• {_mini(chk, 140)}", styles["Normal"]))
    story.append(Spacer(1, 12))

    # ---------- Evidence quotes (from research) ----------
    if research:
        ev = research.get("evidence_quotes", []) or []
        if ev:
            story.append(Paragraph("Evidence (for reference only)", styles["Heading2"]))
            for e in ev[:6]:
                line = f"“{_mini(e.get('text',''), 200)}” — {_mini(e.get('source',''), 120)}"
                story.append(Paragraph(line, styles["Normal"]))

    doc.build(story)
    with open(out_path, "rb") as f:
        return f.read()
