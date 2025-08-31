# ocr_tools.py
# Minimal OCR utility used by app.py:
#   ocr_images(keyframes) -> [{"t": float, "lines": [str, ...], "image_path": str}]
#
# - If Tesseract (pytesseract) is available, we OCR each keyframe.
# - If not, we return empty lines (graceful fallback).

from __future__ import annotations
from typing import List, Dict, Any

import os
from PIL import Image

# Try to import pytesseract (Python wrapper). If not installed or binary missing,
# we'll fall back to no-OCR mode.
try:
    import pytesseract  # requires the Tesseract binary in PATH to actually work
    _HAS_TESS = True
except Exception:
    pytesseract = None
    _HAS_TESS = False


def _ocr_lines_pytesseract(img: Image.Image) -> List[str]:
    """
    OCR using pytesseract. Returns a few clean, short lines.
    If pytesseract is present but the Tesseract binary is missing, this will raise.
    We catch it in the caller and degrade gracefully.
    """
    raw = pytesseract.image_to_string(img)  # simple & robust
    lines = [ln.strip() for ln in raw.splitlines()]
    # Keep only non-empty, short-ish lines to avoid junk
    lines = [ln for ln in lines if ln and len(ln) <= 60]
    # Deduplicate while preserving order
    seen = set()
    out = []
    for ln in lines:
        if ln.lower() not in seen:
            seen.add(ln.lower())
            out.append(ln)
    # Limit to a few lines to save tokens
    return out[:4]


def ocr_images(keyframes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    keyframes: [{"t": float, "image_path": str}, ...]
    returns:   [{"t": float, "lines": [str,...], "image_path": str}, ...]
    """
    results: List[Dict[str, Any]] = []
    for kf in keyframes:
        t = float(kf.get("t", 0.0))
        path = kf.get("image_path", "")
        lines: List[str] = []

        if _HAS_TESS and path and os.path.exists(path):
            try:
                with Image.open(path) as img:
                    # Optional pre-processing for better OCR on captions:
                    # convert to grayscale & raise contrast a bit
                    img = img.convert("L")
                    lines = _ocr_lines_pytesseract(img)
            except Exception:
                # Any failure -> fall back to no text
                lines = []
        else:
            # No OCR available in this environment -> empty lines
            lines = []

        results.append({"t": t, "lines": lines, "image_path": path})
    return results
