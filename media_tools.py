# media_tools.py
# yt-dlp download, ffprobe duration, keyframe extraction using OpenCV histogram diffs
from __future__ import annotations

import os
import math
import subprocess
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
import yt_dlp

DATA_DIR = Path("data")
DL_DIR = DATA_DIR / "downloads"
FRAMES_DIR = DATA_DIR / "frames"

def _ensure_dirs():
    DL_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

def download_video(url: str, out_dir: Path = DL_DIR) -> Path:
    """Download a video via yt-dlp and return the local file path."""
    _ensure_dirs()
    ydl_opts = {
        "format": "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "noprogress": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filepath = Path(ydl.prepare_filename(info))
    # Normalize to .mp4 when possible (remux, no re-encode)
    if filepath.suffix.lower() not in [".mp4", ".m4v", ".mov"]:
        out_mp4 = filepath.with_suffix(".mp4")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(filepath), "-c", "copy", str(out_mp4)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
            )
            filepath = out_mp4
        except Exception:
            pass
    return filepath

def probe_duration(video_path: Path) -> float:
    """Use ffprobe to get duration in seconds (float)."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1",
            str(video_path),
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
        return float(out)
    except Exception:
        return 0.0

def _hist_cut_points(cap: cv2.VideoCapture, step: int = 10, thresh: float = 0.5) -> List[int]:
    """
    Fast cut detection: every `step` frames, compute grayscale hist diff.
    Return a list of frame indices that look like cuts.
    """
    cuts = [0]
    prev_hist = None
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok2, frame = cap.retrieve()
            if not ok2:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            if prev_hist is not None:
                diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                if diff >= thresh:
                    cuts.append(idx)
            prev_hist = hist
        idx += 1
    return sorted(list(set(cuts)))

def extract_keyframes(video_path: Path, max_frames: int = 6) -> List[Dict]:
    """
    Extract up to `max_frames` representative frames.
    Strategy:
      1) Detect rough cut points (histogram diff)
      2) Pick midpoints of segments
      3) If we still have < max_frames, fill with evenly spaced frames across the video
    Returns: [{"t": seconds, "path": "data/frames/<stem>/kf_01.jpg"}, ...]
    """
    _ensure_dirs()
    stem_dir = FRAMES_DIR / video_path.stem
    stem_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return []

    # 1) detect cuts
    cuts = _hist_cut_points(cap, step=max(1, int(fps // 3)), thresh=0.55)
    if not cuts or cuts[0] != 0:
        cuts = [0] + cuts
    if cuts[-1] != total - 1:
        cuts.append(total - 1)

    # 2) segment midpoints
    picks = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        mid = (a + b) // 2
        picks.append(mid)

    # 3) ensure count: top up evenly spaced if needed
    if len(picks) < max_frames:
        extra_needed = max_frames - len(picks)
        candidates = np.linspace(0, total - 1, num=max_frames * 2, dtype=int).tolist()
        # Keep unique and not already picked
        extras = [c for c in candidates if c not in picks]
        picks = (picks + extras)[:max_frames]
    elif len(picks) > max_frames:
        step = math.ceil(len(picks) / max_frames)
        picks = picks[::step][:max_frames]

    # De-dup & in-range
    picks = sorted(list({max(0, min(total - 1, p)) for p in picks}))

    # Save images
    out = []
    for i, fidx in enumerate(picks, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            continue
        t = fidx / fps
        out_path = stem_dir / f"kf_{i:02d}.jpg"
        cv2.imwrite(str(out_path), frame)
        out.append({"t": round(t, 3), "path": str(out_path)})
    cap.release()
    return out
