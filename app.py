# app.py — BriefGen v2: Ingest + Auto Research + Analyzer + Script (Gemini)

import json
from pathlib import Path
import streamlit as st

from media_tools import download_video, probe_duration, extract_keyframes
from ocr_tools import ocr_keyframes
from product_research import auto_collect_product_docs, summarize_to_claims
from prompts import build_analyzer_messages, build_script_messages
from validators import validate_analyzer_json, validate_script_json
from llm import gemini_json

st.set_page_config(page_title="BriefGen v2 — Ingest · Research · Analyze · Script", layout="wide")
st.title("🎬 BriefGen v2")

# ---------------- State ----------------
defaults = {
    "video_path": "", "keyframes": [], "ocr": [], "duration": 0.0,
    "brand": "", "product": "", "extra_urls": "", "sources_used": [],
    "research_json": {}, "approved_claims": [], "required_disclaimers": [],
    "analyzer_json": {}, "script_json": {}, "platform": "tiktok", "target_runtime_s": 20.0
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# ---------------- Controls ----------------
with st.sidebar:
    st.header("Runtime / Platform")
    st.session_state.platform = st.selectbox("Platform", ["tiktok", "reels", "ytshorts"], index=0)
    st.session_state.target_runtime_s = st.slider("Target runtime (s)", 7, 60, 20, 1)
    st.markdown("---")
    st.header("Brand Voice (optional)")
    tone = st.text_input("Tone", value="conversational, confident, no hype")
    must_include = st.text_area("Must include (one per line)", value="")
    avoid = st.text_area("Avoid words (one per line)", value="insane\nultimate\nbest ever")
    brand_voice = {
        "tone": tone,
        "must_include": [x.strip() for x in must_include.splitlines() if x.strip()],
        "avoid": [x.strip() for x in avoid.splitlines() if x.strip()],
    }

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
        "⬇️ keyframes_meta.json",
        data=json.dumps(st.session_state.keyframes, ensure_ascii=False, indent=2),
        file_name="keyframes_meta.json", mime="application/json",
    )
    st.download_button(
        "⬇️ ocr_frames.json",
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
    with st.spinner("Searching sources…"):
        extra = [u.strip() for u in st.session_state.extra_urls.splitlines() if u.strip()]
        docs, sources = auto_collect_product_docs(st.session_state.brand, st.session_state.product, extra_urls=extra)
        st.session_state.sources_used = sources
    with st.spinner("Summarizing to proposed claims…"):
        research = summarize_to_claims(st.session_state.brand, st.session_state.product, docs)
        st.session_state.research_json = research
        st.session_state.required_disclaimers = research.get("required_disclaimers", []) or []

if st.session_state.sources_used:
    st.subheader("Sources used")
    for u in st.session_state.sources_used:
        st.markdown(f"- {u}")

if st.session_state.research_json:
    conf = st.session_state.research_json.get("confidence", "unknown")
    st.write(f"Model confidence: **{conf}**")
    st.subheader("Proposed claims (review & approve)")
    proposed = st.session_state.research_json.get("proposed_claims", []) or []
    if not proposed:
        st.info("No claims extracted. Try refining Brand/Product or re-run research.")
    approved = []
    for i, c in enumerate(proposed):
        if st.checkbox(c, key=f"claim_{i}", value=True):
            approved.append(c)
    st.session_state.approved_claims = approved

    st.subheader("Required disclaimers (from research)")
    for d in st.session_state.required_disclaimers:
        st.markdown(f"- {d}")

    with st.expander("Raw research JSON", expanded=False):
        st.json(st.session_state.research_json)

    st.success(f"Approved {len(st.session_state.approved_claims)} claims ✅")

# ---------------- Analyzer (Gemini) ----------------
st.markdown("---")
st.header("4) Analyze reference (director view)")

if st.button("🧠 Run Analyzer", use_container_width=True):
    if not st.session_state.keyframes:
        st.warning("Run ingestion first to extract keyframes/OCR.")
    else:
        with st.spinner("Analyzing reference video (Gemini)…"):
            # Build keyframe meta with only basename for image_ref convenience
            kf_meta = [{"t": k["t"], "path": k["path"], "image_ref": Path(k["path"]).name} for k in st.session_state.keyframes]
            prompt = build_analyzer_messages(
                platform=st.session_state.platform,
                duration_s=st.session_state.duration or None,
                keyframes_meta=kf_meta,
                ocr_frames=st.session_state.ocr,
                transcript_text=None,  # placeholder for future ASR
                aspect_ratio="9:16",
            )
            raw = gemini_json(prompt)
            try:
                j = json.loads(raw)
            except Exception:
                j = {}
            st.session_state.analyzer_json = j
            errs = validate_analyzer_json(j)
            if errs:
                st.error("Analyzer JSON issues:\n- " + "\n- ".join(errs))
            else:
                st.success("Analyzer JSON looks good ✅")

if st.session_state.analyzer_json:
    with st.expander("Analyzer JSON", expanded=False):
        st.json(st.session_state.analyzer_json)
    st.download_button(
        "⬇️ analyzer.json",
        data=json.dumps(st.session_state.analyzer_json, ensure_ascii=False, indent=2),
        file_name="analyzer.json", mime="application/json",
    )

# ---------------- Script (Gemini) ----------------
st.markdown("---")
st.header("5) Generate script (style-transfer)")

if st.button("🎬 Generate Script", use_container_width=True):
    if not st.session_state.analyzer_json:
        st.warning("Run the Analyzer first.")
    elif not st.session_state.approved_claims:
        st.warning("Approve at least one claim in Product Research.")
    else:
        with st.spinner("Authoring style-transfer script (Gemini)…"):
            prompt = build_script_messages(
                analyzer_json=st.session_state.analyzer_json,
                brand=st.session_state.brand,
                product=st.session_state.product,
                approved_claims=st.session_state.approved_claims,
                required_disclaimers=st.session_state.required_disclaimers,
                target_runtime_s=float(st.session_state.target_runtime_s),
                platform=st.session_state.platform,
                brand_voice=brand_voice,
            )
            raw = gemini_json(prompt)
            try:
                j = json.loads(raw)
            except Exception:
                j = {}
            st.session_state.script_json = j
            errs = validate_script_json(j, target_runtime_s=float(st.session_state.target_runtime_s))
            if errs:
                st.warning("Script JSON warnings:\n- " + "\n- ".join(errs))
            else:
                st.success("Script JSON ready ✅")

if st.session_state.script_json:
    with st.expander("Script JSON", expanded=False):
        st.json(st.session_state.script_json)
    st.download_button(
        "⬇️ script.json",
        data=json.dumps(st.session_state.script_json, ensure_ascii=False, indent=2),
        file_name="script.json", mime="application/json",
    )

st.markdown("---")
st.caption("Next: PDF brief export with embedded keyframes & timestamped scene table. Say the word and I’ll add it.")
