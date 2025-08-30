# app.py
# Ingestion backbone UI:
# - Paste URL or upload video
# - Download via yt-dlp (if URL)
# - Probe duration
# - Extract keyframes
# - OCR each keyframe
# - Inspect and download JSON

import json
from pathlib import Path
from typing import Optional

import streamlit as st
from media_tools import download_video, probe_duration, extract_keyframes
from ocr_tools import ocr_keyframes

st.set_page_config(page_title="BriefGen v2 — Ingestion", layout="wide")
st.title("🎬 BriefGen v2 — Stage 1: Ingestion")

# ---- State
if "video_path" not in st.session_state:
    st.session_state["video_path"] = ""
if "keyframes" not in st.session_state:
    st.session_state["keyframes"] = []
if "ocr" not in st.session_state:
    st.session_state["ocr"] = []
if "duration" not in st.session_state:
    st.session_state["duration"] = 0.0

# ---- Inputs
st.header("1) Provide a reference video")
col1, col2 = st.columns([2, 1])

with col1:
    url = st.text_input("Paste TikTok / Instagram / YouTube URL", placeholder="https://...")
with col2:
    uploaded = st.file_uploader("…or upload a video file", type=["mp4", "mov", "m4v"], accept_multiple_files=False)

run = st.button("⚙️ Fetch & Prepare", type="primary", use_container_width=True)

# ---- Actions
if run:
    try:
        if url:
            st.info("Downloading with yt-dlp…")
            path = download_video(url)
        elif uploaded:
            data_dir = Path("data") / "downloads"
            data_dir.mkdir(parents=True, exist_ok=True)
            path = data_dir / uploaded.name
            with open(path, "wb") as f:
                f.write(uploaded.read())
        else:
            st.error("Please paste a URL or upload a video.")
            st.stop()

        st.session_state["video_path"] = str(path)
        st.success(f"Saved video → `{path.name}`")

        st.info("Probing duration…")
        dur = probe_duration(path)
        st.session_state["duration"] = float(dur)
        st.write(f"Duration: **{dur:.2f}s**")

        st.info("Extracting keyframes…")
        kfs = extract_keyframes(path, max_frames=6)
        st.session_state["keyframes"] = kfs

        st.info("Running OCR on keyframes…")
        ocr = ocr_keyframes(kfs)
        st.session_state["ocr"] = ocr

        st.success("Ingestion complete ✅")

    except Exception as e:
        st.error(f"Ingestion failed: {e}")
        st.stop()

# ---- Output
st.markdown("---")
st.header("2) Inspect results")

if st.session_state["video_path"]:
    st.write("**Video file:**", st.session_state["video_path"])
    st.write("**Duration:**", f"{st.session_state['duration']:.2f}s")

if st.session_state["keyframes"]:
    st.subheader("Keyframes")
    cols = st.columns(3)
    for i, fr in enumerate(st.session_state["keyframes"]):
        with cols[i % 3]:
            st.image(fr["path"], caption=f"t={fr['t']}s", use_container_width=True)

if st.session_state["ocr"]:
    st.subheader("OCR (per keyframe)")
    for rec in st.session_state["ocr"]:
        st.write(f"**t={rec['t']}s** • {rec['image_path']}")
        if rec["text"]:
            for line in rec["text"]:
                st.markdown(f"- {line}")
        else:
            st.markdown("*(no overlay text detected)*")

    # Downloads
    st.download_button(
        "⬇️ Download keyframes_meta.json",
        data=json.dumps(st.session_state["keyframes"], ensure_ascii=False, indent=2),
        file_name="keyframes_meta.json",
        mime="application/json",
    )
    st.download_button(
        "⬇️ Download ocr_frames.json",
        data=json.dumps({"frames": st.session_state["ocr"]}, ensure_ascii=False, indent=2),
        file_name="ocr_frames.json",
        mime="application/json",
    )

st.caption(
    "Next: Product Research → Analyzer (narrative phases, beats, DNA) → Style-Transfer Script → PDF Brief."
)
