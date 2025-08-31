# analyzer.py
from __future__ import annotations
import math
import json
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from PIL import Image

from archetypes import ARCHETYPES, ARCHETYPE_PHASES
from exemplars import top_k_exemplars
from prompts import build_analyzer_prompt_with_fewshots
from validators import validate_analyzer_json

# Reuse your existing helpers if you have them:
# - get_keyframes(video) -> [{"t": float, "image_path": str}]
# - run_ocr(image_path) -> {"t": float, "lines": [str,...]}
# - read_transcript(video) -> str
# - call_openai_json / call_gemini_json
from media_tools import extract_duration, grab_even_keyframes
from ocr_tools import ocr_images
from llm import gemini_json, openai_json  # your wrappers

# ---------- Low-level features (lightweight) ----------
def frame_motion_series(video_path: str, key_ts: List[float]) -> List[float]:
    """
    Approximate motion using frame diffs at key timestamps.
    Returns list same length as key_ts in [0..1] scale.
    """
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
    # scale to 0..1 robustly
    if not vals: return []
    mx = max(vals) or 1e-6
    return [v/mx for v in vals]

def cut_density(key_ts: List[float]) -> float:
    if len(key_ts) < 2: return 0.0
    # assume even sampling; use count as a proxy for cuts
    # (for true cuts, wire in a detector; this is robust enough for routing)
    n = len(key_ts)
    return min(1.0, n / 20.0)  # ~20 keyframes -> 1.0

def caption_density(ocr_frames: List[Dict[str,Any]]) -> float:
    if not ocr_frames: return 0.0
    total_lines = sum(len(f.get("lines", [])) for f in ocr_frames)
    return min(1.0, total_lines / 30.0)

def speech_proxy(transcript: str, cap_density: float) -> float:
    words = len(transcript.strip().split())
    w_score = min(1.0, words / 80.0)
    return max(w_score, cap_density)  # if captions present, treat as speechy

def build_feature_vector(speech_ratio: float, cut_den: float, motion_mean: float, ocr_den: float) -> List[float]:
    return [round(speech_ratio,3), round(cut_den,3), round(motion_mean,3), round(ocr_den,3)]

# ---------- Main entry ----------
def analyze_reference_video(
    *,
    video_path: str,
    provider: str = "Gemini",
    aspect_ratio: str = "9:16",
    platform: str = "tiktok",
) -> Dict[str, Any]:
    duration = extract_duration(video_path) or 0.0
    # keyframes ~ every 2.5s; keep small for speed
    kfs = grab_even_keyframes(video_path, every_s=2.5, limit=16)  # [{"t","image_path"}]
    key_ts = [x["t"] for x in kfs]

    # OCR
    ocr = ocr_images(kfs)  # [{"t":float,"lines":[...],"image_path":str}]
    ocr_den = caption_density(ocr)

    # Transcript (your existing method; empty string ok)
    try:
        from media_tools import transcribe_audio  # if you have it
        transcript = transcribe_audio(video_path) or ""
    except Exception:
        transcript = ""

    # Motion & cut proxies
    motion_vals = frame_motion_series(video_path, key_ts)
    motion_mean = float(np.mean(motion_vals)) if motion_vals else 0.0
    cuts = cut_density(key_ts)

    # Speech ratio proxy
    sp_ratio = speech_proxy(transcript, ocr_den)

    # Build retrieval vector & get fewshots
    vec = build_feature_vector(sp_ratio, cuts, motion_mean, ocr_den)
    fewshots = top_k_exemplars(vec, k=2)

    # Build compact keyframes_meta for the prompt
    keyframes_meta = [{"t": round(x["t"],2), "ref": f"kf@{round(x['t'],2)}"} for x in kfs]
    ocr_frames = [{"t": round(f["t"],2), "lines": f.get("lines", [])[:3]} for f in ocr]  # trim for token safety

    # Prepare prompt
    from prompts import ANALYZER_SCHEMA_EXAMPLE  # if you kept the earlier file
    prompt = build_analyzer_prompt_with_fewshots(
        platform=platform,
        duration_s=duration,
        aspect_ratio=aspect_ratio,
        keyframes_meta=keyframes_meta,
        ocr_frames=ocr_frames,
        transcript_text=transcript,
        archetype_menu=ARCHETYPES,
        grammar_table=ARCHETYPE_PHASES,
        fewshots=fewshots,
        schema_example=ANALYZER_SCHEMA_EXAMPLE,
    )

    # Call model
    if provider.lower().startswith("gemini"):
        raw = gemini_json(prompt)
    else:
        raw = openai_json(prompt)

    # Self-check & light repair
    issues = validate_analyzer_json(raw)
    if issues:
        # second-pass fix prompt (compact)
        fix_prompt = f"""
You produced JSON, but it has issues:\n- {chr(10).join(issues[:8])}
Return a corrected JSON ONLY, preserving your choices wherever possible. Keep phases within archetype grammar. 
JSON you produced:
{json.dumps(raw, ensure_ascii=False)}
"""
        raw = gemini_json(fix_prompt) if provider.lower().startswith("gemini") else openai_json(fix_prompt)

    # Attach our low-level signals for downstream use (not for the model)
    raw.setdefault("video_metadata", {})
    raw["video_metadata"]["duration_s"] = duration
    raw["video_metadata"]["aspect_ratio"] = aspect_ratio
    raw.setdefault("global_signals", {})
    raw["global_signals"].update({
        "speech_presence": "none" if sp_ratio < 0.15 else ("low" if sp_ratio < 0.35 else ("medium" if sp_ratio < 0.7 else "high")),
        "tempo": "calm" if cuts < 0.22 else ("moderate" if cuts < 0.5 else "fast")
    })
    raw["_debug"] = {
        "feature_vector": vec,
        "retrieved_fewshots": [f["name"] for f in fewshots],
    }
    return raw
