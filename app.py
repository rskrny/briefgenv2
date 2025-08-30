# app.py — BriefGen v2: Ingestion + Product Research (auto-search on brand+product)

import json
from pathlib import Path
import streamlit as st

from media_tools import download_video, probe_duration, extract_keyframes
from ocr_tools import ocr_keyframes
from product_research import auto_collect_product_docs, summarize_to_claims

st.set_page_config(page_title="BriefGen v2 — Ingest + Auto Research", layout="wide")
st.title("🎬 BriefGen v2 — Stage 1: Ingestion · Stage 2: Product Research (auto)")

# ---------------- State ----------------
defaults = {
    "video_path": "", "keyframes": [], "ocr": [], "duration": 0.0,
    "brand": "", "product": "", "extra_urls": "", "sources_used": [],
    "research_json": {}, "approved_claims": []
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# ---------------- Ingestion ----------------
st.header("1) Reference video")
c1, c2 = st.columns([2, 1])
with c1:
    url = st.text_input("Paste TikTok / IG / YouTube URL", placeholder="https://...")
with c2:
    uploaded = st.file_uploader("…or upload a file", type=["mp4", "mov", "m4v"])

if st.button("⚙️ Fetch & Prepare", type="primary", use_container_width=True):
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
            st.error("Paste a URL or upload a file.")
            st.stop()

        st.session_state.video_path = str(path)
        st.success(f"Saved → `{path.name}`")

        dur = probe_duration(path)
        st.session_state.duration = float(dur)
        st.write(f"Duration: **{dur:.2f}s**")

        kfs = extract_keyframes(path, max_frames=6)
        st.session_state.keyframes = kfs

        ocr = ocr_keyframes(kfs)
        st.session_state.ocr = ocr

        st.success("Ingestion complete ✅")

    except Exception as e:
        st.error(f"Ingestion failed: {e}")
        st.stop()

st.markdown("---")
st.header("2) Inspect (keyframes + OCR)")
if st.session_state.video_path:
    st.write("**Video:**", st.session_state.video_path)
    st.write("**Duration:**", f"{st.session_state.duration:.2f}s")

if st.session_state.keyframes:
    cols = st.columns(3)
    for i, fr in enumerate(st.session_state.keyframes):
        with cols[i % 3]:
            st.image(fr["path"], caption=f"t={fr['t']}s", use_container_width=True)

if st.session_state.ocr:
    st.subheader("OCR")
    for rec in st.session_state.ocr:
        st.write(f"**t={rec['t']}s** • {rec['image_path']}")
        for line in rec["text"] or []:
            st.markdown(f"- {line}")

    st.download_button(
        "⬇️ Download keyframes_meta.json",
        data=json.dumps(st.session_state.keyframes, ensure_ascii=False, indent=2),
        file_name="keyframes_meta.json", mime="application/json",
    )
    st.download_button(
        "⬇️ Download ocr_frames.json",
        data=json.dumps({"frames": st.session_state.ocr}, ensure_ascii=False, indent=2),
        file_name="ocr_frames.json", mime="application/json",
    )

# ---------------- Product Research (auto) ----------------
st.markdown("---")
st.header("3) Product research (auto-find features)")

st.session_state.brand = st.text_input("Brand", value=st.session_state.brand or "YourBrand")
st.session_state.product = st.text_input("Product", value=st.session_state.product or "YourProduct")

with st.expander("Optional: add your own source URLs (one per line)"):
    st.session_state.extra_urls = st.text_area(
        "Extra URLs (optional)",
        value=st.session_state.extra_urls or "",
        placeholder="https://example.com/product\nhttps://retailer.com/listing"
    )

if st.button("🔎 Auto Research (Gemini)", use_container_width=True):
    with st.spinner("Searching the web for sources…"):
        extra = [u.strip() for u in st.session_state.extra_urls.splitlines() if u.strip()]
        docs, sources = auto_collect_product_docs(st.session_state.brand, st.session_state.product, extra_urls=extra)
        st.session_state.sources_used = sources

    with st.spinner("Summarizing to proposed claims…"):
        research = summarize_to_claims(st.session_state.brand, st.session_state.product, docs)
        st.session_state.research_json = research

if st.session_state.sources_used:
    st.subheader("Sources used")
    for u in st.session_state.sources_used:
        st.markdown(f"- {u}")

if st.session_state.research_json:
    conf = st.session_state.research_json.get("confidence", "unknown")
    st.write(f"Model confidence: **{conf}**")
    st.subheader("Proposed claims (review & approve)")
    proposed = st.session_state.research_json.get("proposed_claims", [])
    if not proposed:
    st.info("No claims extracted yet. Try refining the Brand/Product name, or click Auto Research again.\n"
            "Tip: include exact model code (e.g., BTW73) so search finds the right pages.")
    approved = []
    for i, c in enumerate(proposed):
        if st.checkbox(c, key=f"claim_{i}", value=True):
            approved.append(c)
    st.session_state.approved_claims = approved

    st.subheader("Required disclaimers (from research)")
    for d in st.session_state.research_json.get("required_disclaimers", []) or []:
        st.markdown(f"- {d}")

    sugg = st.session_state.research_json.get("source_suggestions", []) or []
    if sugg:
        with st.expander("Suggested sources to verify"):
            for s in sugg:
                st.markdown(f"- {s}")

    with st.expander("Raw research JSON", expanded=False):
        st.json(st.session_state.research_json)

    st.success(f"Approved {len(st.session_state.approved_claims)} claims ✅")

st.caption("Next up (Batch C): Analyzer (style DNA + narrative) → Script (style-transfer) → PDF brief.")
