# pdf_export.py
# Build a polished PDF influencer brief with ReportLab

from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


def make_brief_pdf(
    out_path: str,
    analyzer: Dict[str, Any],
    script: Dict[str, Any],
    product_facts: Dict[str, Any],
    keyframe_images: List[Dict],
    title: str = "AI-Generated Influencer Brief",
) -> bytes:
    """
    Build PDF; return bytes.
    - analyzer: Analyzer JSON
    - script: Script JSON
    - product_facts: {"brand":..,"product":..,"approved_claims":..,"required_disclaimers":..}
    - keyframe_images: [{"t":..,"path":..}]
    """
    buf = out_path
    doc = SimpleDocTemplate(buf, pagesize=letter)
    styles = getSampleStyleSheet()
    story: List = []

    # --- Cover ---
    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Brand: {product_facts.get('brand')}", styles["Normal"]))
    story.append(Paragraph(f"Product: {product_facts.get('product')}", styles["Normal"]))
    story.append(Spacer(1, 20))

    story.append(Paragraph("Approved Claims:", styles["Heading3"]))
    for c in product_facts.get("approved_claims", []):
        story.append(Paragraph(f"• {c}", styles["Normal"]))

    story.append(Spacer(1, 20))
    if product_facts.get("required_disclaimers"):
        story.append(Paragraph("Required Disclaimers:", styles["Heading3"]))
        for d in product_facts.get("required_disclaimers", []):
            story.append(Paragraph(f"• {d}", styles["Normal"]))
    story.append(PageBreak())

    # --- Narrative Map ---
    story.append(Paragraph("Narrative Phases", styles["Heading2"]))
    data = [["Phase", "Start", "End", "Purpose"]]
    for n in analyzer.get("narrative", []):
        data.append([
            n.get("phase"),
            f"{n.get('start_s', '')}s",
            f"{n.get('end_s', '')}s",
            n.get("purpose", ""),
        ])
    table = Table(data, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
    ]))
    story.append(table)
    story.append(PageBreak())

    # --- Scene Table ---
    story.append(Paragraph("Scenes", styles["Heading2"]))
    sdata = [["#", "Time", "Camera", "Action", "Dialogue", "On-Screen Text", "Retention"]]
    for s in script.get("script", {}).get("scenes", []):
        sdata.append([
            s.get("idx"),
            f"{s.get('start_s',0)}–{s.get('end_s',0)}s",
            s.get("camera",""),
            s.get("action",""),
            s.get("dialogue",""),
            "\n".join(s.get("on_screen_text",[]) or []),
            s.get("retention_note",""),
        ])
    stable = Table(sdata, repeatRows=1)
    stable.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("FONTSIZE", (0,0), (-1,-1), 8),
    ]))
    story.append(stable)
    story.append(PageBreak())

    # --- Keyframes ---
    story.append(Paragraph("Keyframes", styles["Heading2"]))
    for fr in keyframe_images:
        try:
            story.append(Paragraph(f"t={fr['t']}s", styles["Heading3"]))
            story.append(Image(fr["path"], width=250, height=444))  # adjust aspect
            story.append(Spacer(1, 12))
        except Exception:
            continue
    story.append(PageBreak())

    # --- CTAs & Checklist ---
    story.append(Paragraph("CTAs", styles["Heading2"]))
    for cta in script.get("script", {}).get("cta_options", []):
        story.append(Paragraph(f"{cta.get('variant')}: {cta.get('line')}", styles["Normal"]))
    story.append(Spacer(1, 20))

    story.append(Paragraph("Checklist", styles["Heading2"]))
    for chk in script.get("checklist", []):
        story.append(Paragraph(f"• {chk}", styles["Normal"]))

    doc.build(story)
    with open(buf, "rb") as f:
        return f.read()
