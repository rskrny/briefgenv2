# briefgenv2 — Ingestion Backbone

Paste a TikTok/IG/YT URL (or upload a file), then:
- Download video via **yt-dlp**
- Probe duration via **ffprobe**
- Extract 3–6 **keyframes** via a fast shot-change heuristic (OpenCV)
- **OCR** the frames with **Tesseract**
- Inspect and download the JSON (keyframes + OCR)

## Deploy on Streamlit Cloud

**Secrets (App → Settings → Secrets):**
