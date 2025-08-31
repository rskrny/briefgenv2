# media_tools.py
# Lightweight video helpers used by app.py
# - extract_duration(video_path) -> float seconds
# - grab_even_keyframes(video_path, every_s=2.5, limit=16) -> [{"t": float, "image_path": str}]
# - frame_at_time(video_path, t_s) -> PIL.Image.Image
# - transcribe_audio(video_path) -> str   (stub; returns "")
# - download_video_from_url(url: str, tmp_dir: pathlib.Path, max_mb: int = 200) -> pathlib.Path

from __future__ import annotations
import os
import io
import math
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from PIL import Image, ImageOps

# Try OpenCV; degrade gracefully if unavailable
try:
    import cv2  # opencv-python-headless
except Exception:
    cv2 = None

# Requests is only used for direct media URLs (e.g., .mp4)
try:
    import requests
except Exception:
    requests = None


# -----------------------------
# Duration
# -----------------------------
def extract_duration(video_path: str) -> float:
    """Return duration in seconds (best effort)."""
    if not cv2:
        return 0.0
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    try:
        return float(frames) / float(fps) if fps else 0.0
    except Exception:
        return 0.0


# -----------------------------
# Grab evenly spaced keyframes
# -----------------------------
def _read_frame_at(cap, frame_idx: int) -> Image.Image:
    ok = cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(0, frame_idx)))
    ok, frame = cap.read()
    if not ok or frame is None:
        # fallback placeholder
        img = Image.new("RGB", (1280, 720), color=(240, 240, 240))
        return ImageOps.expand(img, border=1, fill="black")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)

def grab_even_keyframes(video_path: str, every_s: float = 2.5, limit: int = 16) -> List[Dict[str, Any]]:
    """
    Extracts up to `limit` frames roughly every `every_s` seconds.
    Saves JPEGs into a temp directory and returns [{"t": t_sec, "image_path": path}, ...]
    """
    out: List[Dict[str, Any]] = []
    if not cv2:
        # Produce blank images to keep pipeline alive
        tmp_dir = tempfile.mkdtemp(prefix="kf_")
        for i in range(min(limit, 6)):
            t = round(i * every_s, 2)
            img = Image.new("RGB", (1280, 720), color=(240, 240, 240))
            img = ImageOps.expand(img, border=1, fill="black")
            p = os.path.join(tmp_dir, f"kf_{i:02d}.jpg")
            img.save(p, "JPEG", quality=85)
            out.append({"t": float(t), "image_path": p})
        return out

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return out

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    dur_s = float(frames) / float(fps) if fps else 0.0
    if dur_s <= 0:
        cap.release()
        return out

    tmp_dir = tempfile.mkdtemp(prefix="kf_")
    t = 0.0
    idx = 0
    while t <= dur_s and idx < limit:
        frame_idx = int(t * fps)
        img = _read_frame_at(cap, frame_idx)
        path = os.path.join(tmp_dir, f"kf_{idx:02d}.jpg")
        img.save(path, "JPEG", quality=85)
        out.append({"t": float(round(t, 2)), "image_path": path})
        idx += 1
        t += max(0.5, float(every_s))

    if not out:
        img = _read_frame_at(cap, 0)
        path = os.path.join(tmp_dir, f"kf_00.jpg")
        img.save(path, "JPEG", quality=85)
        out.append({"t": 0.0, "image_path": path})

    cap.release()
    return out


# -----------------------------
# Single frame at time (for previews / PDF)
# -----------------------------
def frame_at_time(video_path: str, t_s: float) -> Image.Image:
    if not cv2:
        img = Image.new("RGB", (1280, 720), color=(240, 240, 240))
        return ImageOps.expand(img, border=1, fill="black")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        img = Image.new("RGB", (1280, 720), color=(240, 240, 240))
        return ImageOps.expand(img, border=1, fill="black")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = int(max(0.0, float(t_s)) * fps)
    img = _read_frame_at(cap, frame_idx)
    cap.release()
    return img


# -----------------------------
# Optional transcription (stub)
# -----------------------------
def transcribe_audio(video_path: str) -> str:
    """
    Stub that returns empty string. Replace with a real VAD/ASR if desired.
    The app treats empty transcript as SHOWCASE-friendly (minimal speech).
    """
    return ""


# -----------------------------
# Download via URL (TikTok/YouTube/Instagram/direct .mp4)
# -----------------------------
def _download_direct_file(url: str, tmp_dir: Path, max_mb: int = 200) -> Optional[Path]:
    if requests is None:
        return None
    # Try to enforce size limits
    try:
        head = requests.head(url, timeout=10, allow_redirects=True)
        size = int(head.headers.get("Content-Length", "0"))
        if size and size > max_mb * 1024 * 1024:
            raise RuntimeError(f"File too large (> {max_mb}MB).")
    except Exception:
        # If HEAD fails, continue; we'll stream with a cap
        pass

    fn = tmp_dir / "input.mp4"
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total = 0
            with open(fn, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    total += len(chunk)
                    if total > max_mb * 1024 * 1024:
                        raise RuntimeError(f"Downloaded size exceeded {max_mb}MB cap.")
        return fn
    except Exception:
        return None

def download_video_from_url(url: str, tmp_dir: Path, max_mb: int = 200) -> Path:
    """
    Attempts to download a video from a URL. Supports:
      - Direct media URLs (.mp4/.mov/.m4v/.webm)
      - Pages (TikTok/YouTube/Instagram/etc.) via yt-dlp
    Returns a local file path. Raises on failure.
    """
    url_lower = url.lower().strip()

    # Direct media link?
    if any(url_lower.endswith(ext) for ext in (".mp4", ".mov", ".m4v", ".webm", ".mpeg4")):
        p = _download_direct_file(url, tmp_dir, max_mb=max_mb)
        if p and p.exists():
            return p
        raise RuntimeError("Failed to download direct media file.")

    # Use yt-dlp for page URLs
    try:
        from yt_dlp import YoutubeDL
        outtmpl = str(tmp_dir / "input.%(ext)s")
        ydl_opts = {
            "outtmpl": outtmpl,
            "quiet": True,
            "noprogress": True,
            "nocheckcertificate": True,
            "merge_output_format": "mp4",
            "format": "mp4/bestvideo+bestaudio/best",
            "retries": 2,
            "fragment_retries": 2,
        }
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            out_path = Path(ydl.prepare_filename(info))
            # Prefer mp4 if merge produced it
            mp4_path = out_path.with_suffix(".mp4")
            if mp4_path.exists():
                out_path = mp4_path
            # Quick size cap check
            if out_path.stat().st_size > max_mb * 1024 * 1024:
                raise RuntimeError(f"Downloaded file exceeds {max_mb}MB.")
            return out_path
    except Exception as e:
        raise RuntimeError(f"yt-dlp failed to fetch this URL: {e}")
