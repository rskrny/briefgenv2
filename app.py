# app.py
# briefgenv2 — Analyzer v4: rock-solid product research (consensus + PDF/manuals), URL or upload
from __future__ import annotations

import os
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import streamlit as st
from PIL import Image

# Local modules expected in this repo
from llm import gemini_json, openai_json
from media_tools import (
    extract_duration,
    grab_even_keyframes,
    frame_at_time,
    transcribe_audio,
    download_video_from_url,
)
from ocr_tools import ocr_images
from validators import validate_analyzer_json, validate_script_json
from prompts import (
    build_analyzer_prompt_with_fewshots,
    build_script_messages,
    ANALYZER_SCHEMA_EXAMPLE,
    ARCHETYPES,
    ARCHETYPE_PHASES,
)

# NEW: robust research stack
from product_research import research_product  # returns provenance + confidence
from vision_tools import infer_visual_features

# Optional PDF export
try:
    import pdf_export as pe  # must expose build_pdf(...)
except Exception:
    pe = None

st.set_page_config(page_title="briefgenv2 — Analyzer v4", page_icon="🎬", layout="wide")

# ------------------------------
# Few-shot anchors (tiny, inline)
# ------------------------------
EXEMPLARS: List[Dict[str, Any]] = [
    {"name": "ASMR Unbox – Gadget", "archetype": "SHOWCASE",
     "mini_analysis": {"phases": [{"name": "Unbox"}, {"name": "Handle/Features"}, {"name": "Demo"}, {"name": "Outro"}]}},
    {"name": "Hands-only Beauty ASMR", "archetype": "SHOWCASE",
     "mini_analysis": {"phases": [{"name": "Handle/Features"}, {"name": "Demo"}]}},
    {"name": "Talky Review – Tech", "archetype": "NARRATIVE",
     "mini_analysis": {"phases": [{"name": "Hook"}, {"name": "Solution"}, {"name": "Proof"}, {"name": "CTA"}]}},
    {"name": "Tutorial – 3 Steps", "archetype": "TUTORIAL",
     "mini_analysis": {"phases": [{"name": "Step 1"}, {"name": "Step 2"}, {"name": "Step 3"}, {"name": "Result"}]}},
    {"name": "Comparison – Side by Side", "archetype": "COMPARISON",
     "mini_analysis": {"phases": [{"name": "Comparison Setup"}, {"name": "Side-by-side Test"}, {"name": "Result"}]}},
    {"name": "Field Test – Outdoor", "archetype": "TEST_DEMO",
     "mini_analysis": {"phases": [{"name": "Setup"}, {"name": "Test"}, {"name": "Results"}, {"name": "Takeaway"}]}},
]

# ------------------------------
# Lightweight evidence helpers
# ------------------------------
def caption_density(ocr_frames: List[Dict[str, Any]]) -> float:
    if not ocr_frames:
        return 0.0
    total_lines = sum(len(f.get("lines", [])) for f in ocr_frames)
    return min(1.0, total_lines / 30.0)

def speech_proxy(transcript_text: str, cap_density: float) -> float:
    words = len((transcript_text or "").strip().split())
    w_score = min(1.0, words / 80.0)
    return max(w_score, cap_density)

def cut_density_from_keyframes(key_ts: List[float]) -> float:
    if len(key_ts) < 2:
        return 0.0
    return min(1.0, len(key_ts) / 20.0)

def motion_series(video_path: str, key_ts: List[float]) -> List[float]:
    try:
        import cv2
    except Exception:
        return [0.0 for _ in key_ts]
    vals: List[float] = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [0.0 for _ in key_ts]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    prev = None
    for t in key_ts:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok:
            vals.append(0.0)
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if prev is None:
            vals.append(0.0)
        else:
            diff = cv2.absdiff(gray, prev)
            vals.append(float(np.mean(diff)) / 255.0)
        prev = gray
    cap.release()
    if not vals:
        return []
    mx = max(vals) or 1e-6
    return [v / mx for v in vals]

def propose_scene_candidates(
    duration: float, key_ts: List[float], motion_vals: List[float], threshold: float = 0.45, min_len: float = 2.0
) -> List[Dict[str, Any]]:
    """Simple cut suggestions from motion jumps around keyframe times."""
    if not key_ts:
        if duration >= 9.0:
            thirds = [0.0, round(duration / 3, 2), round(2 * duration / 3, 2), round(duration, 2)]
            return [{"start_s": thirds[i], "end_s": thirds[i + 1], "why": "equal split"} for i in range(3)]
        return [{"start_s": 0.0, "end_s": round(duration, 2), "why": "whole clip"}]

    cuts = [0.0]
    for i in range(1, len(motion_vals)):
        jump = abs(motion_vals[i] - motion_vals[i - 1])
        t_here = key_ts[i]
        if jump >= threshold and (t_here - cuts[-1]) >= min_len:
            cuts.append(round(t_here, 2))
    if (duration - cuts[-1]) >= min_len:
        cuts.append(round(duration, 2))

    segs = [{"start_s": cuts[i], "end_s": cuts[i + 1], "why": "motion jump" if i else "start"} for i in range(len(cuts) - 1)]
    return segs or [{"start_s": 0.0, "end_s": round(duration, 2), "why": "whole clip"}]

def save_upload(upload, tmp_dir: Path) -> Path:
    p = tmp_dir / upload.name
    with open(p, "wb") as f:
        f.write(upload.getbuffer())
    return p

def images_for_display(video_path: str, phases: List[Dict[str, Any]]) -> List[Tuple[str, Image.Image]]:
    out: List[Tuple[str, Image.Image]] = []
    for ph in (phases or [])[:8]:
        name = ph.get("phase") or ph.get("name") or "Phase"
        stt = ph.get("start_s") or 0.0
        end = ph.get("end_s") or stt
        t = (stt + end) / 2.0 if end > stt else stt
        try:
            img = frame_at_time(video_path, float(t))
        except Exception:
            img = Image.new("RGB", (1280, 720), color=(240, 240, 240))
        out.append((f"{name} @ {t:.1f}s", img))
    return out

# ------------------------------
# Sidebar settings
# ------------------------------
with st.sidebar:
    st.title("⚙️ Settings")
    provider = st.selectbox("Model Provider", ["Gemini", "OpenAI"], index=0)
    openai_key = st.text_input("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""), type="password")
    gemini_key = st.text_input("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY", ""), type="password")
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    if gemini_key:
        os.environ["GOOGLE_API_KEY"] = gemini_key

    st.markdown("---")
    st.caption("PDF Branding (optional if pdf_export present)")
    pdf_brand = st.text_input("Brand Header", "BrandPal")
    pdf_footer = st.text_input("Footer Note", "Generated with briefgenv2")

    st.markdown("---")
    st.caption("Fetcher (optional)")
    use_playwright = st.checkbox("Render pages with Playwright (JS-heavy sites)", value=False)
    os.environ["USE_PLAYWRIGHT"] = "1" if use_playwright else "0"

# ------------------------------
# Main form
# ------------------------------
st.title("📽️ briefgenv2 — Analyzer v4")

colA, colB = st.columns(2)
with colA:
    brand = st.text_input("Brand", "")
    product = st.text_input("Product Name", "")
with colB:
    platform = st.selectbox("Target Platform", ["tiktok", "reels", "ytshorts"], index=0)
    target_runtime = st.number_input("Target Runtime (s)", min_value=6, max_value=90, value=20)

st.markdown("#### Provide a video by **URL** or **upload a file**")
video_url = st.text_input("Video URL (TikTok / YouTube / Instagram / direct .mp4)", "")
video_file = st.file_uploader("…or upload a video file", type=["mp4", "mov", "m4v", "webm", "mpeg4"])

st.markdown("#### Product Research")
product_url_override = st.text_input("Product page URL (optional — boosts accuracy)", "")
auto_research = st.checkbox("Auto research from web + visuals (consensus + PDFs)", value=True)
confidence_cutoff = st.slider("Min confidence to include (0–1)", 0.0, 1.0, 0.6, 0.05)

st.markdown("#### Optional: Claims & Disclaimers")
approved_claims_manual = st.text_area("Approved Claims (one per line)", value="").strip().splitlines()
required_disclaimers_manual = st.text_area("Required Disclaimers (one per line)", value="").strip().splitlines()

run_btn = st.button(
    "Analyze & Generate",
    type="primary",
    disabled=not ((video_url or video_file) and brand and product),
)

# ------------------------------
# Pipeline
# ------------------------------
if run_btn:
    tmp_dir = Path(tempfile.mkdtemp())

    # 1) Resolve source to local file
    try:
        if video_url:
            with st.spinner("Fetching video from URL…"):
                video_path = download_video_from_url(video_url.strip(), tmp_dir)
        else:
            with st.spinner("Saving uploaded file…"):
                video_path = save_upload(video_file, tmp_dir)
    except Exception as e:
        st.error(f"Could not retrieve the video: {e}")
        st.stop()

    # 2) Evidence extraction
    with st.spinner("Extracting keyframes & OCR…"):
        duration = extract_duration(str(video_path)) or 0.0
        kfs = grab_even_keyframes(str(video_path), every_s=2.0, limit=18)  # [{"t","image_path"}]
        key_ts = [x["t"] for x in kfs]
        ocr = ocr_images(kfs)  # [{"t","lines":[...]}]
        cap_den = caption_density(ocr)
        transcript = transcribe_audio(str(video_path)) or ""
        sp_ratio = speech_proxy(transcript, cap_den)
        motion_vals = motion_series(str(video_path), key_ts)
        scene_cands = propose_scene_candidates(duration, key_ts, motion_vals)
        cuts = cut_density_from_keyframes(key_ts)

        keyframes_meta = [{"t": round(x["t"], 2), "ref": f"kf@{round(x['t'], 2)}"} for x in kfs]
        ocr_frames = [{"t": round(f["t"], 2), "lines": f.get("lines", [])[:3]} for f in ocr]

    # 3) Analyze (archetype & phases)
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
            scene_candidates=scene_cands,
        )
        analysis = gemini_json(analyzer_prompt) if provider == "Gemini" else openai_json(analyzer_prompt)

        # Attach derived signals if missing
        analysis.setdefault("video_metadata", {})
        analysis["video_metadata"].setdefault("duration_s", duration)
        analysis["video_metadata"].setdefault("aspect_ratio", "9:16")
        analysis.setdefault("global_signals", {})
        analysis["global_signals"].setdefault(
            "speech_presence",
            "none" if sp_ratio < 0.15 else ("low" if sp_ratio < 0.35 else ("medium" if sp_ratio < 0.7 else "high")),
        )
        analysis["global_signals"].setdefault(
            "tempo", "calm" if cuts < 0.22 else ("moderate" if cuts < 0.5 else "fast")
        )

        # Validate & soft-repair
        issues = validate_analyzer_json(analysis)
        if issues:
            fix_prompt = f"""
You produced JSON, but it has issues:
- {'; '.join(issues[:8])}
Return a corrected JSON ONLY, preserving your choices wherever possible. Keep phases within archetype grammar.
JSON you produced:
{json.dumps(analysis, ensure_ascii=False)}
"""
            analysis = gemini_json(fix_prompt) if provider == "Gemini" else openai_json(fix_prompt)

        # Fallback segmentation for silent SHOWCASE
        arch = analysis.get("archetype") or analysis.get("analysis", {}).get("archetype", "")
        phases = analysis.get("phases") or analysis.get("analysis", {}).get("phases", [])
        if arch == "SHOWCASE" and duration >= 10 and len(phases) <= 1:
            segs = scene_cands[:3] if len(scene_cands) >= 2 else [{"start_s": 0.0, "end_s": duration, "why": "whole clip"}]
            names = ["Unbox", "Handle/Features", "Demo", "Outro"]
            analysis["phases"] = [
                {
                    "phase": names[min(i, len(names) - 1)],
                    "start_s": s["start_s"],
                    "end_s": s["end_s"],
                    "what_happens": "visible action per segment",
                    "camera_notes": "",
                    "audio_notes": "silent",
                    "on_screen_text": "",
                    "evidence": [f"kf@{round((s['start_s'] + s['end_s']) / 2, 2)}"],
                }
                for i, s in enumerate(segs)
            ]

    st.success("Analysis complete.")
    with st.expander("Analyzer JSON", expanded=True):
        st.json(analysis)

    st.subheader("Analyzer Signals")
    gs = analysis.get("global_signals", {})
    st.write(
        f"**Archetype:** {analysis.get('archetype','(see JSON)')}  |  "
        f"**Speech:** {gs.get('speech_presence','?')}  |  **Tempo:** {gs.get('tempo','?')}"
    )

    # Phase previews
    imgs: List[Tuple[str, Image.Image]] = []
    try:
        phases = analysis.get("phases") or analysis.get("analysis", {}).get("phases", [])
        imgs = images_for_display(str(video_path), phases)
        if imgs:
            st.subheader("Reference Keyframes")
            cols = st.columns(min(2, max(1, len(imgs))))
            for i, (label, img) in enumerate(imgs):
                with cols[i % len(cols)]:
                    st.image(img, caption=label, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render phase keyframes: {e}")

    # 4) Product research (consensus + PDFs) with vision + OCR hints
    research = {
        "query": "", "sources": [], "features": [], "specs": [],
        "disclaimers": [], "claims": [], "notes": []
    }
    extra_strip_imgs: List[Tuple[str, Image.Image]] = []  # for PDF storyboard density
    if auto_research:
        with st.spinner("Researching the product (consensus + PDF/manuals + visuals)…"):
            # Vision hints from a subset of frames
            kf_paths_for_vision = [k["image_path"] for k in kfs[:8]]
            vision = infer_visual_features(kf_paths_for_vision, provider=provider)
            vision_hints = (vision.get("visible_features") or [])[:8]

            # OCR hints: short text fragments on screen
            ocr_hints: List[str] = []
            for f in ocr[:8]:
                for ln in f.get("lines", [])[:2]:
                    if len(ln.split()) <= 6:
                        ocr_hints.append(ln)

            research = research_product(
                brand=brand,
                product=product,
                product_url_override=product_url_override.strip(),
                ocr_hints=ocr_hints,
                vision_hints=vision_hints,
                max_results=10,
                min_confidence=confidence_cutoff,
            )

            # Build an extra keyframe strip (6 evenly spaced frames)
            if kfs:
                step = max(1, len(kfs) // 6)
                pick = kfs[::step][:6]
                for item in pick:
                    try:
                        img = Image.open(item["image_path"]).convert("RGB")
                    except Exception:
                        img = Image.new("RGB", (1280, 720), color=(240, 240, 240))
                    extra_strip_imgs.append((f"Ref @ {item['t']:.1f}s", img))

        st.subheader("Product Snapshot")
        if research.get("sources"):
            st.markdown("**Sources used**")
            for s in research["sources"][:10]:
                st.write(f"- [{s.get('title') or s.get('url')}]({s.get('url')})")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**High-confidence Features**")
            feats = research.get("features") or []
            if feats:
                for f in feats[:12]:
                    txt = f.get("text") if isinstance(f, dict) else str(f)
                    st.write(f"• {txt}")
            else:
                st.info("No features reached confidence threshold.")
        with col2:
            sp = research.get("specs") or []
            if sp:
                st.markdown("**Specs**")
                for spec in sp[:12]:
                    if isinstance(spec, dict):
                        st.write(f"• **{spec.get('key','').title()}**: {spec.get('value','')}")
                    else:
                        st.write(f"• {spec}")
            ds = research.get("disclaimers") or []
            if ds:
                st.markdown("**Disclaimers**")
                for d in ds[:8]:
                    txt = d.get("text") if isinstance(d, dict) else str(d)
                    st.write(f"• {txt}")

    # Merge manual + research into script grounding
    def _dedupe(lst: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in lst:
            k = x.strip().lower()
            if k and k not in seen:
                seen.add(k)
                out.append(x.strip())
        return out

    # Extract clean lists from research with confidence cutoff
    research_features = []
    for f in (research.get("features") or []):
        if isinstance(f, dict):
            if f.get("confidence", 0) >= confidence_cutoff:
                research_features.append(f.get("text", ""))
        elif isinstance(f, str):
            research_features.append(f)
    spec_bullets: List[str] = []
    for s in (research.get("specs") or []):
        if isinstance(s, dict):
            if s.get("confidence", 0) >= confidence_cutoff and s.get("value"):
                spec_bullets.append(f"{s.get('key','').title()}: {s.get('value')}")
        elif isinstance(s, str):
            spec_bullets.append(s)

    # Prefer concrete features/specs only (no free-form 'claims')
    final_claims = _dedupe(
        (approved_claims_manual or [])
        + research_features
        + spec_bullets
    )
    final_disclaimers = _dedupe(
        (required_disclaimers_manual or [])
        + [d.get("text","") if isinstance(d,dict) else str(d) for d in (research.get("disclaimers") or [])]
    )

    # 5) Script generation
    with st.spinner("Generating script & shot list…"):
        script_prompt = build_script_messages(
            analyzer_json=analysis,
            brand=brand,
            product=product,
            approved_claims=final_claims,
            required_disclaimers=final_disclaimers,
            target_runtime_s=float(target_runtime),
            platform=platform,
            brand_voice={},
        )
        script_json = gemini_json(script_prompt) if provider == "Gemini" else openai_json(script_prompt)

        # Validate script JSON and soft-fix
        s_issues = validate_script_json(script_json, float(target_runtime))
        if s_issues:
            fix_prompt = f"""
You produced JSON, but it has issues:
- {'; '.join(s_issues[:10])}
Return a corrected JSON ONLY that fixes these problems.
JSON you produced:
{json.dumps(script_json, ensure_ascii=False)}
"""
            script_json = gemini_json(fix_prompt) if provider == "Gemini" else openai_json(fix_prompt)

    st.success("Script generated.")
    with st.expander("Script JSON", expanded=True):
        st.json(script_json)

    # 6) PDF export
    if pe is not None:
        try:
            pdf_bytes: bytes = pe.build_pdf(
                brand_header=pdf_brand,
                footer_note=pdf_footer,
                product_meta={"brand": brand, "name": product, "platform": platform},
                analysis=analysis,
                plan=script_json,
                phase_images=imgs if 'imgs' in locals() else [],
                research=research,
                extra_keyframes=extra_strip_imgs,
            )
            st.subheader("Export PDF")
            st.download_button("⬇️ Download PDF", data=pdf_bytes, file_name=f"brief_{product}.pdf", mime="application/pdf")
        except Exception as e:
            st.info(f"PDF export skipped: {e}")
    else:
        st.info("PDF export module not found. Skipping PDF creation.")
