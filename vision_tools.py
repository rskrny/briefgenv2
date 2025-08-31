# vision_tools.py
from __future__ import annotations
from typing import List, Dict, Any
from llm import gemini_json_from_images, openai_json_from_images
import os

VISION_PROMPT = """
You are an outdoor gear visual expert. Look at the images (keyframes from a short product video).
Return JSON ONLY with:
{
  "visible_features": ["short, product-specific visual features (no metrics)"],
  "materials_guess": ["aluminum frame","mesh seat","nylon straps", "..."],
  "avoid_claims": ["comparatives or certifications to avoid"],
  "confidence": 0.0-1.0
}
Rules:
- Only describe what is visually likely (e.g., "reclining back with straps", "aluminum poles", "mesh seat", "carry bag", "cup-holder ring", "low seat height").
- Keep each feature ≤ 7 words. 5–10 features preferred.
- No metrics, no #1/best claims, no certifications.
"""

def infer_visual_features(image_paths: List[str], provider: str = "Gemini") -> Dict[str, Any]:
    if not image_paths:
        return {"visible_features": [], "materials_guess": [], "avoid_claims": [], "confidence": 0.0}
    if provider.lower().startswith("gem"):
        out = gemini_json_from_images(VISION_PROMPT, image_paths)
    else:
        out = openai_json_from_images(VISION_PROMPT, image_paths)
    # Soft validation
    if not isinstance(out, dict):
        return {"visible_features": [], "materials_guess": [], "avoid_claims": [], "confidence": 0.0}
    out.setdefault("visible_features", [])
    out.setdefault("materials_guess", [])
    out.setdefault("avoid_claims", [])
    out.setdefault("confidence", 0.5)
    # Trim & normalize
    out["visible_features"] = [s.strip() for s in out["visible_features"] if isinstance(s, str) and s.strip()]
    out["materials_guess"] = [s.strip() for s in out["materials_guess"] if isinstance(s, str) and s.strip()]
    out["avoid_claims"] = [s.strip() for s in out["avoid_claims"] if isinstance(s, str) and s.strip()]
    return out
