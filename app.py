# app.py — briefgenv2 Analyzer v4.1
from __future__ import annotations
import os, json, tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import streamlit as st
from PIL import Image

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
from product_research import research_product
from vision_tools import infer_visual_features

try:
    import pdf_export as pe
except Exception:
    pe = None

st.set_page_config(page_title="briefgenv2 — Analyzer v4.1", page_icon="🎬", layout="wide")

# ---------------- Few-shot anchors ----------------
EXEMPLARS = [
    {"name":"ASMR Unbox","archetype":"SHOWCASE","mini_analysis":{"phases":[{"name":"Unbox"},{"name":"Handle/Features"},{"name":"Demo"}]}},
    {"name":"Hands-only Beauty","archetype":"SHOWCASE","mini_analysis":{"phases":[{"name":"Handle/Features"},{"name":"Demo"}]}},
    {"name":"Tutorial","archetype":"TUTORIAL","mini_analysis":{"phases":[{"name":"Step 1"},{"name":"Step 2"},{"name":"Result"}]}},
]

# ---------------- Helpers ----------------
def caption_density(ocr_frames: List[Dict[str, Any]]) -> float:
    if not ocr_frames: return 0.0
    total_lines = sum(len(f.get("lines", [])) for f in ocr_frames)
    return min(1.0, total_lines / 30.0)

def speech_proxy(transcript_text: str, cap_density: float) -> float:
    words = len((transcript_text or "").strip().split())
    return max(min(1.0, words / 80.0), cap_density)

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
    mx = max(vals) or 1e-6
    return [v/mx for v in vals]

def propose_scene_candidates(duration: float, key_ts: List[float], motion_vals: List[float],
                             threshold: float = 0.45, min_len: float = 2.0):
    if not key_ts:
        return [{"start_s":0.0,"end_s":round(duration,2),"why":"whole clip"}]
    cuts=[0.0]
    for i in range(1,len(motion_vals)):
        jump = abs(motion_vals[i]-motion_vals[i-1])
        t_here = key_ts[i]
        if jump>=threshold and (t_here-cuts[-1])>=min_len:
            cuts.append(round(t_here,2))
    if (duration - cuts[-1]) >= min_len:
        cuts.append(round(duration,2))
    return [{"start_s":cuts[i],"end_s":cuts[i+1],"why":"motion"} for i in range(len(cuts)-1)]

def save_upload(upload, tmp: Path)->Path:
    p = tmp / upload.name
    with open(p,"wb") as f: f.write(upload.getbuffer())
    return p

def frame_labels(video_path: str, phases: List[Dict[str,Any]]):
    out=[]
    for ph in (phases or [])[:8]:
        name = ph.get("phase") or ph.get("name") or "Phase"
        stt = float(ph.get("start_s") or 0.0)
        end = float(ph.get("end_s") or stt)
        t = (stt+end)/2.0 if end>stt else stt
        try: img = frame_at_time(video_path, t)
        except Exception: img = Image.new("RGB",(1280,720),(240,240,240))
        out.append((f"{name} @ {t:.1f}s", img))
    return out

# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("⚙️ Settings")
    provider = st.selectbox("Model Provider", ["Gemini","OpenAI"], index=0)
    openai_key = st.text_input("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY",""), type="password")
    gemini_key = st.text_input("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY",""), type="password")
    if openai_key: os.environ["OPENAI_API_KEY"] = openai_key
    if gemini_key: os.environ["GOOGLE_API_KEY"] = gemini_key
    st.markdown("---")
    pdf_brand = st.text_input("Brand Header", "BrandPal")
    pdf_footer = st.text_input("Footer Note", "Generated with briefgenv2")

# ---------------- Main ----------------
st.title("📽️ briefgenv2 — Analyzer v4.1")

c1,c2 = st.columns(2)
with c1:
    brand = st.text_input("Brand","")
    product = st.text_input("Product Name (model if you have it)","")
with c2:
    platform = st.selectbox("Target Platform", ["tiktok","reels","ytshorts"], index=0)
    target_runtime = st.number_input("Target Runtime (s)", min_value=6, max_value=90, value=20)

st.markdown("#### Provide a video by **URL** or **upload a file**")
video_url = st.text_input("Video URL (TikTok / YouTube / Instagram / direct .mp4)","")
video_file = st.file_uploader("…or upload a video file", type=["mp4","mov","m4v","webm","mpeg4"])

st.markdown("#### Product Research")
product_url_override = st.text_input("Product page URL (optional — helps accuracy)","")
auto_research = st.checkbox("Auto research from web (manufacturer/consensus first)", value=True)

st.markdown("#### Optional: Claims & Disclaimers")
approved_claims_manual = st.text_area("Approved Claims (one per line)", value="")
required_disclaimers_manual = st.text_area("Required Disclaimers (one per line)", value="")

run = st.button("Analyze & Generate", type="primary",
                disabled=not ((video_url or video_file) and brand and product))

if run:
    tmp = Path(tempfile.mkdtemp())

    # ----------- 1) Fetch video -----------
    try:
        if video_url:
            with st.spinner("Fetching video from URL…"):
                video_path = download_video_from_url(video_url.strip(), tmp)
        else:
            with st.spinner("Saving uploaded file…"):
                video_path = save_upload(video_file, tmp)
    except Exception as e:
        st.error(f"Could not retrieve the video: {e}"); st.stop()

    # ----------- 2) Evidence -----------
    with st.spinner("Extracting frames / OCR / audio…"):
        duration = extract_duration(str(video_path)) or 0.0
        kfs = grab_even_keyframes(str(video_path), every_s=2.0, limit=18)
        key_ts = [x["t"] for x in kfs]
        ocr = ocr_images(kfs)
        cap_den = caption_density(ocr)
        transcript = transcribe_audio(str(video_path)) or ""
        sp_ratio = speech_proxy(transcript, cap_den)
        motion_vals = motion_series(str(video_path), key_ts)
        scene_cands = propose_scene_candidates(duration, key_ts, motion_vals)
        keyframes_meta = [{"t":round(x["t"],2),"ref":f"kf@{round(x['t'],2)}"} for x in kfs]
        ocr_frames = [{"t":round(f["t"],2),"lines":f.get("lines",[])[:3]} for f in ocr]

    # ----------- 3) Analyzer (robust SHOWCASE handling) -----------
    with st.spinner("Analyzing reference video…"):
        analyzer_prompt = build_analyzer_prompt_with_fewshots(
            platform=platform, duration_s=duration, aspect_ratio="9:16",
            keyframes_meta=keyframes_meta, ocr_frames=ocr_frames,
            transcript_text=transcript, archetype_menu=ARCHETYPES,
            grammar_table=ARCHETYPE_PHASES, fewshots=EXEMPLARS,
            schema_example=ANALYZER_SCHEMA_EXAMPLE, scene_candidates=scene_cands,
        )
        analysis = gemini_json(analyzer_prompt) if provider=="Gemini" else openai_json(analyzer_prompt)
        # attach signals
        analysis.setdefault("video_metadata",{})
        analysis["video_metadata"].setdefault("duration_s",duration)
        analysis["video_metadata"].setdefault("aspect_ratio","9:16")
        analysis.setdefault("global_signals",{})
        analysis["global_signals"].setdefault(
            "speech_presence","none" if sp_ratio<0.15 else ("low" if sp_ratio<0.35 else ("medium" if sp_ratio<0.7 else "high"))
        )
        # validate + soft-fix
        issues = validate_analyzer_json(analysis)
        if issues:
            fix = f"You output JSON with issues: {'; '.join(issues[:8])}\nReturn corrected JSON ONLY.\nJSON:\n{json.dumps(analysis)}"
            analysis = gemini_json(fix) if provider=="Gemini" else openai_json(fix)

    st.success("Analysis complete.")
    with st.expander("Analyzer JSON", expanded=False):
        st.json(analysis)

    # phase preview
    try:
        phases = analysis.get("phases") or analysis.get("analysis",{}).get("phases",[])
        phase_imgs = frame_labels(str(video_path), phases)
        if phase_imgs:
            st.subheader("Reference Keyframes")
            cols = st.columns(min(2, max(1, len(phase_imgs))))
            for i,(label,img) in enumerate(phase_imgs):
                with cols[i % len(cols)]:
                    st.image(img, caption=label, use_container_width=True)
    except Exception:
        phase_imgs=[]

    # ----------- 4) Vision cues (for category checks only) -----------
    with st.spinner("Inferring visual category…"):
        kf_paths = [k["image_path"] for k in kfs[:8]]
        vision = infer_visual_features(kf_paths, provider=provider)
        vision_tags = set(x.lower() for x in (vision.get("category_tags") or []))
        visual_hints = [s for s in (vision.get("visible_features") or []) if isinstance(s,str)]

    # ----------- 5) Product research (now category-safe) -----------
    research = {
        "query":"", "sources":[], "features":[], "specs":{},
        "features_detailed":[], "specs_detailed":[], "disclaimers":[],
        "visual_hints": visual_hints, "warnings":[]
    }
    if auto_research:
        with st.spinner("Researching the product (manufacturer/consensus)…"):
            research = research_product(
                brand=brand,
                product=product,
                product_url_override=product_url_override.strip(),
                vision_category_tags=list(vision_tags),
            )

    # show warnings
    if research.get("warnings"):
        for w in research["warnings"]:
            st.warning(w)

    # Verified snapshot (only items with sources)
    st.subheader("Product Snapshot (verified)")
    cL, cR = st.columns(2)
    with cL:
        if research.get("features_detailed"):
            st.markdown("**Features**")
            for f in research["features_detailed"][:14]:
                st.write(f"• {f['text']} _(conf {f['confidence']:.2f}, {len(f['sources'])} src)_")
        else:
            st.caption("No verified features found.")
    with cR:
        if research.get("specs_detailed"):
            st.markdown("**Specs**")
            for s in research["specs_detailed"][:14]:
                st.write(f"• **{s['key'].title()}**: {s['value']} _(conf {s['confidence']:.2f}, {len(s['sources'])} src)_")
        else:
            st.caption("No verified specs found.")

    if research.get("disclaimers"):
        st.markdown("**Disclaimers**")
        for d in research["disclaimers"][:8]:
            st.write(f"• {d}")

    if research.get("sources"):
        st.markdown("**Sources (used)**")
        for s in research["sources"][:10]:
            st.write(f"- [{s.get('title') or s.get('url')}]({s.get('url')})")

    # Visual hints (never used as claims)
    if research.get("visual_hints"):
        st.subheader("Visual Hints (not verified)")
        st.write(", ".join(research["visual_hints"][:12]))

    # ----------- 6) Script generation -----------
    def _dedupe(lines: List[str]) -> List[str]:
        seen=set(); out=[]
        for x in (lines or []):
            k=x.strip().lower()
            if k and k not in seen:
                seen.add(k); out.append(x.strip())
        return out

    manual_claims = [x for x in (approved_claims_manual or "").splitlines() if x.strip()]
    manual_disclaimers = [x for x in (required_disclaimers_manual or "").splitlines() if x.strip()]

    verified_feature_bullets = [f["text"] for f in research.get("features_detailed",[])]
    verified_spec_bullets = [f"{s['key'].title()}: {s['value']}" for s in research.get("specs_detailed",[])]

    final_claims = _dedupe(manual_claims + verified_feature_bullets + verified_spec_bullets)
    final_disclaimers = _dedupe(manual_disclaimers + (research.get("disclaimers") or []))

    with st.spinner("Generating script & shot list…"):
        script_prompt = build_script_messages(
            analyzer_json=analysis,
            brand=brand,
            product=product,
            approved_claims=final_claims,              # verified only
            required_disclaimers=final_disclaimers,
            target_runtime_s=float(target_runtime),
            platform=platform,
            brand_voice={},
        )
        script_json = gemini_json(script_prompt) if provider=="Gemini" else openai_json(script_prompt)
        s_issues = validate_script_json(script_json, float(target_runtime))
        if s_issues:
            fix = f"You output JSON with issues: {'; '.join(s_issues[:10])}\nReturn corrected JSON ONLY.\nJSON:\n{json.dumps(script_json)}"
            script_json = gemini_json(fix) if provider=="Gemini" else openai_json(fix)

    st.success("Script generated.")
    with st.expander("Script JSON", expanded=False):
        st.json(script_json)

    # ----------- 7) PDF export -----------
    if pe is not None:
        try:
            # build a few additional frames to densify storyboard
            extra_strip=[]
            if kfs:
                step = max(1, len(kfs)//6)
                pick = kfs[::step][:6]
                from PIL import Image as PILImage
                for item in pick:
                    try: img = PILImage.open(item["image_path"]).convert("RGB")
                    except Exception: img = PILImage.new("RGB",(1280,720),(240,240,240))
                    extra_strip.append((f"Ref @ {item['t']:.1f}s", img))
            pdf_bytes = pe.build_pdf(
                brand_header=pdf_brand,
                footer_note=pdf_footer,
                product_meta={"brand": brand, "name": product, "platform": platform},
                analysis=analysis,
                plan=script_json,
                phase_images=phase_imgs if 'phase_imgs' in locals() else [],
                research=research,
                extra_keyframes=extra_strip,
            )
            st.subheader("Export PDF")
            st.download_button("⬇️ Download PDF", data=pdf_bytes,
                               file_name=f"brief_{product}.pdf", mime="application/pdf")
        except Exception as e:
            st.info(f"PDF export skipped: {e}")
    else:
        st.info("PDF export module not found. Skipping PDF creation.")
