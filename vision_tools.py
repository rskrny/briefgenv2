# vision_tools.py
from __future__ import annotations
from typing import List, Dict, Any
from llm import gemini_json_from_images, openai_json_from_images

VISION_PROMPT = """
You are a product-visual analyst. Look at the images (keyframes from a short product video).
Return JSON ONLY:

{
  "visible_features": ["short, product-agnostic bullets (≤7 words)"],
  "category_tags": ["high-level product types (e.g., 'folding chair', 'headphones')"],
  "materials_guess": ["aluminum","plastic","mesh","glass","leather", "..."],
  "avoid_claims": ["phrases to avoid if speculative"],
  "confidence": 0.0-1.0
}

Rules:
- Only describe what can be reasonably inferred visually (parts, mechanisms, ports, controls, included accessories).
- No performance metrics, no certifications, no '#1/best' superlatives.
- Keep language universal; do not assume a brand or model.
"""

def infer_visual_features(image_paths: List[str], provider: str = "Gemini") -> Dict[str, Any]:
    if not image_paths:
        return {"visible_features": [], "category_tags": [], "materials_guess": [], "avoid_claims": [], "confidence": 0.0}
    if provider.lower().startswith("gem"):
        out = gemini_json_from_images(VISION_PROMPT, image_paths)
    else:
        out = openai_json_from_images(VISION_PROMPT, image_paths)
    if not isinstance(out, dict):
        return {"visible_features": [], "category_tags": [], "materials_guess": [], "avoid_claims": [], "confidence": 0.0}
    out.setdefault("visible_features", [])
    out.setdefault("category_tags", [])
    out.setdefault("materials_guess", [])
    out.setdefault("avoid_claims", [])
    out.setdefault("confidence", 0.5)
    # normalize
    out["visible_features"] = [s.strip() for s in out["visible_features"] if isinstance(s, str) and s.strip()]
    out["category_tags"] = [s.strip() for s in out["category_tags"] if isinstance(s, str) and s.strip()]
    out["materials_guess"] = [s.strip() for s in out["materials_guess"] if isinstance(s, str) and s.strip()]
    out["avoid_claims"] = [s.strip() for s in out["avoid_claims"] if isinstance(s, str) and s.strip()]
    return out
