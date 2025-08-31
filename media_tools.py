# media_tools.py
# Lightweight video helpers used by app.py
# - extract_duration(video_path) -> float seconds
# - grab_even_keyframes(video_path, every_s=2.5, limit=16) -> [{"t": float, "image_path": str}]
# - frame_at_time(video_path, t_s) -> PIL.Image.Image
# - transcribe_audio(video_path) -> str   (stub; returns "")

from __future__ import annotations
import os
import math
import tempfile
from typing import List, Dict, Any

import numpy as np
from PIL import Image, ImageOps

# Try OpenCV; degrade gracefully if unavailable
try:
    import cv2  # opencv-python-headless
except Exception:
    cv2 = None


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

    # If we got 0 (super short video), grab first frame
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
