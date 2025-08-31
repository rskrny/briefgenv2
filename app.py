# app.py
# Streamlit UI that:
# 1) Analyzes a reference video with archetype-aware, evidence-first logic + few-shots
# 2) Generates a creator-ready script JSON based on analyzer output
# 3) (Optionally) builds a PDF via pdf_export if present in the repo

from __future__ import annotations
import os
import io
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import streamlit as st
from PIL import Image

# --- Local modules expected in your repo ---
# llm.py should expose: gemini_json(prompt: str) -> dict, openai_json(prompt: str) -> dict
from llm import gemini_json, openai_json
# media_tools.py should expose: extract_duration(path), grab_even_keyframes(path, every_s=2.5, limit=16), frame_at_time(path, t)
from media_tools import extract_duration, grab_even_keyframes
# ocr_tools.py should expose: ocr_images(keyframes) -> [{"t":float,"lines":[...],"image_path":str}]
from ocr_tools import ocr_images
# validators.py should expose: validate_analyzer_json(dict)->List[str], validate_script_json(dict, target_runtime_s)->List[str]
from validators import validate_analyzer_json, validate_script_json
# prompts.py (this file) exposes builders + schema + archetype grammar
from prompts import (
    build_analyzer_prompt_with_fewshots,
    build_script_messages,
    ANALYZER_SCHEMA_EXAMPLE,
    ARCHETYPES,
    ARCHETYPE_PHASES,
)

# pdf_export is optional; if absent or signature differs, we skip PDF
try:
    import pdf_export as pe
except Exception:  # noqa
    pe = None

st.set_page_config(page_title="briefgenv2 – Analyzer v3", page_icon="🎬", layout="wide")

# =============================================================================
# Few-shot EXEMPLARS (tiny, in-app). You can move these to a separate file later.
# Each exemplar has: name, archetype, mini_analysis (PHASE NAMES ONLY)
# =============================================================================
EXEMPLARS: List[Dict[str, Any]] = [
    {
        "name": "ASMR Unbox – Gadget",
        "archetype": "SHOWCASE",
        "mini_analysis": {
            "phases": [
                {"name":"Unbox"},
                {"name":"Handle/Features"},
                {"name":"Demo"},
                {"name":"Outro"}
            ]
        }
    },
    {
        "name": "Hands-only Beauty ASMR",
        "archetype": "SHOWCASE",
        "mini_analysis": {"phases":[
            {"name":"Handle/Features"},
            {"name":"Demo"}
        ]}
    },
    {
        "name": "Talky Review – Tech",
        "archetype": "NARRATIVE",
        "mini_analysis": {"phases":[
            {"name":"Hook"},
            {"name":"Solution"},
            {"name":"Proof"},
            {"name":"CTA"}
        ]}
    },
    {
        "name": "Tutorial – 3 Steps",
        "archetype": "TUTORIAL",
        "mini_analysis": {"phases":[
            {"name":"Step 1"},{"name":"Step 2"},{"name":"Step 3"},{"name":"Result"}
        ]}
    },
    {
        "name": "Comparison – Side by Side",
        "archetype": "COMPARISON",
        "mini_analysis": {"phases":[
            {"name":"Comparison Setup"},{"name":"Side-by-side Test"},{"name":"Result"}
        ]}
    },
    {
        "name": "Field Test – Outdoor",
        "archetype": "TEST_DEMO",
        "mini_analysis": {"phases":[
            {"name":"Setup"},{"name":"Test"},{"name":"Results"},{"name":"Takeaway"}
        ]}
    }
]

# =============================================================================
# Lightweight evidence features (no heavy ML)
# =============================================================================

def caption_density(ocr_frames: List[Dict[str,Any]]) -> float:
    if not ocr_frames: return 0.0
    total_lines = sum(len(f.get("lines", [])) for f in ocr_frames)
    return min(1.0, total_lines / 30.0)

def speech_proxy(transcript_text: str, cap_density: float) -> float:
    words = len((transcript_text or "").strip().split())
    w_score = min(1.0, words / 80.0)
    return max(w_score, cap_density)

def cut_density_from_keyframes(key_ts: List[float]) -> float:
    if len(key_ts) < 2: return 0.0
    return min(1.0, len(key_ts) / 20.0)  # crude but sufficient for tempo gating

def motion_series(video_path: str, key_ts: List[float]) -> List[float]:
    try:
        import cv2
    except Exception:
        return [0.0 for _ in key_ts]
    vals = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [0.0 for _ in key_ts]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    prev = None
    for t in key_ts:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok:
            vals.append(0.0); continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        if prev is None:
            vals.append(0.0)
        else:
            diff = cv2.absdiff(gray, prev)
            vals.append(float(np.mean(diff))/255.0)
        prev = gray
    cap.release()
    # normalize to 0..1
    if not vals: return []
    mx = max(vals) or 1e-6
    return [v/mx for v in vals]

# =============================================================================
# Helpers
# =============================================================================

def save_upload(upload, tmp_dir: Path) -> Path:
    p = tmp_dir / upload.name
    with open(p, "wb") as f:
        f.write(upload.getbuffer())
    return p

def try_transcribe(video_path: str) -> str:
    # Optional: if you have a transcribe_audio() in media_tools, use it.
    try:
        from media_tools import transcribe_audio
        return transcribe_audio(video_path) or ""
    except Exception:
        return ""

def images_for_display(video_path: str, phases: List[Dict[str,Any]]) -> List[Tuple[str, Image.Image]]:
    """Grab mid-frame of each phase for quick visual checks."""
    from media_tools import frame_at_time  # available in your repo
    out = []
    for ph in phases[:8]:  # cap to 8
        name = ph.get("phase") or ph.get("name") or "Phase"
        st = ph.get("start_s") or 0.0
        en = ph.get("end_s") or st
        t = (st + en) / 2.0 if en > st else st
        try:
            img = frame_at_time(video_path, float(t))
        except Exception:
            img = Image.new("RGB", (1280, 720), color=(240,240,240))
        out.append((f"{name} @ {t:.1f}s", img))
    return out

# =============================================================================
# UI
# =============================================================================

with st.sidebar:
    st.title("⚙️ Settings")
    provider = st.selectbox("Model Provider", ["Gemini", "OpenAI"], index=0)
    openai_key = st.text_input("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""), type="password")
    gemini_key = st.text_input("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY", ""), type="password")

    if openai_key: os.environ["OPENAI_API_KEY"] = openai_key
    if gemini_key: os.environ["GOOGLE_API_KEY"] = gemini_key

    st.markdown("---")
    st.caption("PDF Branding (optional if pdf_export present)")
    pdf_brand = st.text_input("Brand Header", "BrandPal")
    pdf_footer = st.text_input("Footer Note", "Generated with briefgenv2")

st.title("🎬 briefgenv2 — Analyzer v3")

colA, colB = st.columns(2)
with colA:
    brand = st.text_input("Brand", "")
    product = st.text_input("Product Name", "")
with colB:
    platform = st.selectbox("Target Platform", ["tiktok", "reels", "ytshorts"], index=0)
    target_runtime = st.number_input("Target Runtime (s)", min_value=6, max_value=90, value=20)

video_file = st.file_uploader("Upload Reference Video", type=["mp4","mov","m4v","webm"])
run_btn = st.button("Analyze & Generate", type="primary", disabled=not(video_file and brand and product))

# Optional: approved claims / disclaimers inputs (can be empty)
st.markdown("#### Optional: Claims & Disclaimers")
approved_claims = st.text_area("Approved Claims (one per line)", value="").strip().splitlines()
required_disclaimers = st.text_area("Required Disclaimers (one per line)", value="").strip().splitlines()

if run_btn:
    tmp_dir = Path(tempfile.mkdtemp())
    video_path = save_upload(video_file, tmp_dir)

    # ---------- Evidence extraction ----------
    with st.spinner("Extracting keyframes & OCR…"):
        duration = extract_duration(str(video_path)) or 0.0
        kfs = grab_even_keyframes(str(video_path), every_s=2.5, limit=16)  # [{"t","image_path"}]
        key_ts = [x["t"] for x in kfs]
        ocr = ocr_images(kfs)  # [{"t","lines":[...],"image_path": str}]
        cap_den = caption_density(ocr)
        transcript = try_transcribe(str(video_path))
        sp_ratio = speech_proxy(transcript, cap_den)
        cuts = cut_density_from_keyframes(key_ts)
        motion_vals = motion_series(str(video_path), key_ts)
        motion_mean = float(np.mean(motion_vals)) if motion_vals else 0.0

        keyframes_meta = [{"t": round(x["t"],2), "ref": f"kf@{round(x['t'],2)}"} for x in kfs]
        ocr_frames = [{"t": round(f["t"],2), "lines": f.get("lines", [])[:3]} for f in ocr]  # trim lines for token safety

    # ---------- Build analyzer prompt ----------
    with st.spinner("Analyzing reference video (archetype-aware)…"):
        analyzer_prompt = build_analyzer_prompt_with_fewshots(
            platform=platform,
            duration_s=duration,
            aspect_ratio="9:16",
            keyframes_meta=keyframes_meta,
            ocr_frames=ocr_frames,
            transcript_text=transcript,
            archetype_menu=ARCHETYPES,
            grammar_table=ARCHETYPE_PHASES,
            fewshots=EXEMPLARS,
            schema_example=ANALYZER_SCHEMA_EXAMPLE,
        )
        if provider == "Gemini":
            analysis = gemini_json(analyzer_prompt)
        else:
            analysis = openai_json(analyzer_prompt)

        # Attach derived signals for UI visibility even if model didn't include them
        analysis.setdefault("video_metadata", {})
        analysis["video_metadata"].setdefault("duration_s", duration)
        analysis["video_metadata"].setdefault("aspect_ratio", "9:16")
        analysis.setdefault("global_signals", {})
        if "speech_presence" not in analysis["global_signals"]:
            analysis["global_signals"]["speech_presence"] = (
                "none" if sp_ratio < 0.15 else ("low" if sp_ratio < 0.35 else ("medium" if sp_ratio < 0.7 else "high"))
            )
        if "tempo" not in analysis["global_signals"]:
            analysis["global_signals"]["tempo"] = "calm" if cuts < 0.22 else ("moderate" if cuts < 0.5 else "fast")

        # Validate & soft-repair
        issues = validate_analyzer_json(analysis)
        if issues:
            fix_prompt = f"""
You produced JSON, but it has issues:\n- {'; '.join(issues[:8])}
Return a corrected JSON ONLY, preserving your choices wherever possible. Keep phases within archetype grammar.
JSON you produced:
{json.dumps(analysis, ensure_ascii=False)}
"""
            analysis = gemini_json(fix_prompt) if provider == "Gemini" else openai_json(fix_prompt)

    st.success("Analysis complete.")
    with st.expander("Analyzer JSON", expanded=True):
        st.json(analysis)

    st.subheader("Analyzer Signals")
    gs = analysis.get("global_signals", {})
    arch = analysis.get("archetype") or analysis.get("analysis", {}).get("archetype", "")
    conf = analysis.get("confidence") or analysis.get("analysis", {}).get("confidence", "")
    st.write(f"**Archetype:** {arch or '(model did not set — see JSON)'}")
    st.write(f"**Speech presence:** {gs.get('speech_presence','?')}  |  **Tempo:** {gs.get('tempo','?')}")
    st.caption(f"Few-shots used: {', '.join([e['name'] for e in EXEMPLARS])}")

    # ---------- Show phase keyframes ----------
    try:
        phases = analysis.get("phases") or analysis.get("analysis", {}).get("phases", [])
        imgs = images_for_display(str(video_path), phases)
        if imgs:
            st.subheader("Reference Keyframes")
            cols = st.columns(min(3, len(imgs)))
            for i, (label, img) in enumerate(imgs):
                with cols[i % len(cols)]:
                    st.image(img, caption=label, use_column_width=True)
    except Exception as e:
        st.warning(f"Could not render phase keyframes: {e}")

    # ---------- Script generation ----------
    with st.spinner("Generating script & shot list…"):
        script_prompt = build_script_messages(
            analyzer_json=analysis,
            brand=brand,
            product=product,
            approved_claims=[c for c in approved_claims if c.strip()],
            required_disclaimers=[d for d in required_disclaimers if d.strip()],
            target_runtime_s=float(target_runtime),
            platform=platform,
            brand_voice={},  # wire in if you maintain a brand voice object
        )
        script_json = gemini_json(script_prompt) if provider == "Gemini" else openai_json(script_prompt)

        # Validate script
        s_issues = validate_script_json(script_json, float(target_runtime))
        if s_issues:
            fix_prompt = f"""
You produced JSON, but it has issues:\n- {'; '.join(s_issues[:10])}
Return a corrected JSON ONLY that fixes these problems.
JSON you produced:
{json.dumps(script_json, ensure_ascii=False)}
"""
            script_json = gemini_json(fix_prompt) if provider == "Gemini" else openai_json(fix_prompt)

    st.success("Script generated.")
    with st.expander("Script JSON", expanded=True):
        st.json(script_json)

    # ---------- Optional PDF ----------
    if pe is not None:
        try:
            st.subheader("Export PDF")
            pdf_bytes: bytes = pe.build_pdf(
                brand_header=pdf_brand,
                footer_note=pdf_footer,
                product_meta={"brand": brand, "name": product, "platform": platform},
                analysis=analysis,
                plan=script_json,
                # If your pdf_export expects raw images, pass them instead:
                phase_images=imgs if 'imgs' in locals() else []
            )
            st.download_button("⬇️ Download PDF", data=pdf_bytes, file_name=f"brief_{product}.pdf", mime="application/pdf")
        except Exception as e:
            st.info(f"PDF export skipped (pdf_export.build_pdf not available or different signature): {e}")
    else:
        st.info("PDF export module not found. Skipping PDF creation.")
