# ocr_tools.py
# Tesseract OCR with simple denoise and confidence filtering

from __future__ import annotations
from typing import List, Dict

import cv2
import numpy as np
import pytesseract

def _preprocess(img):
    # grayscale → light denoise → slight threshold to improve text extraction
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 35, 35)
    # adaptive threshold helps on mixed overlays
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 35, 11)
    return th

def ocr_keyframes(frames: List[Dict]) -> List[Dict]:
    """
    frames: [{"t": 0.0, "path": ".../kf_01.jpg"}, ...]
    returns: [{"t": 0.0, "text": ["SALE 20% OFF","Tap to shop"], "image_path": "..."}]
    """
    results: List[Dict] = []
    for fr in frames:
        path = fr["path"]
        t = float(fr["t"])
        img = cv2.imread(path)
        if img is None:
            results.append({"t": t, "text": [], "image_path": path})
            continue
        proc = _preprocess(img)

        data = pytesseract.image_to_data(proc, output_type=pytesseract.Output.DICT)
        words = []
        for txt, conf in zip(data["text"], data["conf"]):
            try:
                c = float(conf)
            except Exception:
                c = -1
            if c >= 55 and txt and txt.strip():
                words.append(txt.strip())
        # Merge to short lines (rough heuristic)
        text = []
        if words:
            buff = []
            for w in words:
                buff.append(w)
                if len(" ".join(buff)) > 26:  # keep lines short for overlays
                    text.append(" ".join(buff))
                    buff = []
            if buff:
                text.append(" ".join(buff))
        results.append({"t": round(t, 3), "text": text, "image_path": path})
    return results
