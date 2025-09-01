# gemini_fetcher.py — 2025-09-01 (category-aware; strict brand+model+title/URL guards)
"""
Fetch product specs/features via Gemini with Google Search Retrieval.

Goals:
- Enforce BOTH brand and model in page title/URL (exact-token check).
- Apply category allow/deny markers so we never cross into the wrong product type.
- If no compliant pages are found, return {"status":"NOT_FOUND"} instead of guessing.

Env:
  GOOGLE_API_KEY  (Streamlit secrets)
Deps:
  google-generativeai >= 0.7.0
"""

from __future__ import annotations
import os, json, logging
import google.generativeai as genai
from typing import Optional

log = logging.getLogger(__name__)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

CANDIDATE_MODELS = ["gemini-1.5-pro", "gemini-1.5-flash"]

# ---------------- Category Profiles ----------------
# Keep this small but high-signal; it’s easy to extend.
CATEGORY_PROFILES = {
    # audio wearables
    "headphones": {
        "aliases": ["headphone", "headphones", "earbuds", "headset"],
        "allow": ["headphone", "earbuds", "headset", "bluetooth", "anc", "noise cancellation",
                  "drivers", "impedance", "frequency response", "codec", "aac", "ldac", "aptx"],
        "deny":  ["wh", "portable power station", "generator", "inverter", "lifepo4",
                  "ac outlet", "solar input", "welding", "lawn mower"],
    },
    "earbuds": {
        "aliases": ["earbud", "earbuds", "headphones", "headset"],
        "allow": ["earbud", "earbuds", "bluetooth", "anc", "noise cancellation", "ipx", "codec"],
        "deny":  ["wh", "generator", "inverter", "lifepo4", "ac outlet", "solar input"],
    },
    # consumer tech
    "smartphone": {
        "aliases": ["phone", "smartphone", "android", "iphone"],
        "allow": ["smartphone", "android", "ios", "camera", "display", "battery", "soc", "snapdragon", "exynos"],
        "deny":  ["portable power station", "generator", "vacuum", "sofa", "detergent", "shampoo"],
    },
    "laptop": {
        "aliases": ["laptop", "notebook"],
        "allow": ["laptop", "notebook", "intel", "amd", "ryzen", "ram", "ssd", "display", "keyboard"],
        "deny":  ["generator", "sofa", "toothbrush", "shampoo"],
    },
    "camera": {
        "aliases": ["camera", "mirrorless", "dslr", "action camera"],
        "allow": ["camera", "sensor", "iso", "aperture", "lens", "shutter", "fps", "4k"],
        "deny":  ["generator", "chair", "shampoo"],
    },
    "speaker": {
        "aliases": ["speaker", "soundbar", "smart speaker"],
        "allow": ["speaker", "soundbar", "bluetooth", "watt", "woofer", "tweeter", "dolby"],
        "deny":  ["generator", "lifepo4", "ac outlet"],
    },
    # home & living (few examples)
    "vacuum": {
        "aliases": ["vacuum", "vacuum cleaner", "robot vacuum", "stick vacuum"],
        "allow": ["vacuum", "suction", "pa", "dustbin", "hepa", "run time"],
        "deny":  ["smartphone", "headphones", "generator"],
    },
    "coffee maker": {
        "aliases": ["coffee maker", "espresso", "drip coffee", "keurig"],
        "allow": ["coffee", "espresso", "bar", "water tank", "brew"],
        "deny":  ["smartphone", "headphones", "generator"],
    },
    # power/energy
    "portable power station": {
        "aliases": ["portable power station", "generator", "power station"],
        "allow": ["wh", "kwh", "dc", "ac outlet", "inverter", "lifepo4", "solar input", "cycle life"],
        "deny":  ["headphones", "earbuds", "anc", "drivers", "impedance", "frequency response"],
    },
}

def _resolve_category(cat_hint: Optional[str]) -> tuple[str, dict]:
    """Return the normalized category key and its profile."""
    if not cat_hint:
        return ("", {})
    c = str(cat_hint).strip().lower()
    # direct match
    if c in CATEGORY_PROFILES:
        return c, CATEGORY_PROFILES[c]
    # alias match
    for k, v in CATEGORY_PROFILES.items():
        if c in v.get("aliases", []):
            return k, v
    return (c, {})  # unknown → empty profile (minimal constraints)


# ---------------- Prompt ----------------
SYSTEM_BASE = (
    "You are a strictly factual Product-Spec Retriever with Google Search access.\n"
    "RULES:\n"
    "• Check the product pages yourself with Google Search Retrieval.\n"
    "• A page is VALID only if BOTH brand and model appear in its TITLE or URL as exact tokens.\n"
    "• Prioritize brand PDP, then major retailers (Amazon/Walmart/BestBuy/Target/B&H/REI), then reputable reviews.\n"
    "• Accept a spec/feature only if it appears in ≥2 independent sources (brand PDP counts as one).\n"
    "• Reject UI/menus/shortcuts/breadcrumbs. Reject irrelevant category pages.\n"
    "• Output ONLY JSON matching the schema. If you cannot confirm ≥3 specs, return {\"status\":\"NOT_FOUND\"}."
)

SCHEMA = """{
  "status": "OK",
  "product_title": "Brand Model ...",
  "category": "normalized category",
  "specs": { "key_snake": "value", "...": "..." },
  "features": ["compact bullets <=120 chars", "..."],
  "citations": [{ "attr": "key_snake_or_feature_index", "url": "https://..." }],
  "pages_used": [{"title":"...", "url":"https://..."}]
}"""

PROMPT_TEMPLATE = """
### SYSTEM
{system}

### USER
Brand: {brand}
Model: {model}
Desired Category: {category}
Category Allow Markers: {allow_markers}
Category Deny Markers: {deny_markers}

STRICT FILTERS YOU MUST ENFORCE:
1) Title or URL must contain both BRAND and MODEL as exact tokens (case-insensitive).
2) Pages must contain at least one ALLOW marker for the Desired Category, and must not contain any DENY markers.
3) If compliant pages cannot be found, return {{"status":"NOT_FOUND"}}.

Return ONLY JSON that matches this schema:
{schema}
"""

def _make_model(model_name: str):
    return genai.GenerativeModel(
        model_name=model_name,
        tools=[{"google_search_retrieval": {}}],
        generation_config={"temperature": 0.15, "max_output_tokens": 1400},
        safety_settings={"HARASSMENT": "block_none"},
    )

def _strip_code_fence(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("` \n")
        if t.lower().startswith("json"):
            t = t[4:].lstrip()
    return t

def gemini_product_info(
    brand: str,
    model: str,
    category_hint: Optional[str] = None,
    timeout_s: int = 60,
) -> dict:
    cat_key, profile = _resolve_category(category_hint)
    allow = ", ".join(profile.get("allow", [])) if profile else ""
    deny  = ", ".join(profile.get("deny", []))  if profile else ""
    system = SYSTEM_BASE
    prompt = PROMPT_TEMPLATE.format(
        system=system,
        brand=brand,
        model=model,
        category=cat_key or (category_hint or "product (match brand+model)"),
        allow_markers=allow or "(none — still enforce brand+model in title/URL)",
        deny_markers=deny or "(none — still enforce brand+model in title/URL)",
        schema=SCHEMA,
    )

    last_err = None
    for m in CANDIDATE_MODELS:
        try:
            gmodel = _make_model(m)
            resp = gmodel.generate_content(prompt, request_options={"timeout": timeout_s})
            data = json.loads(_strip_code_fence(resp.text or ""))
            return data
        except Exception as exc:
            last_err = exc
            log.warning("Gemini call failed on %s: %s", m, exc, exc_info=False)
            continue

    return {"status": "ERROR", "error": str(last_err) if last_err else "Unknown Gemini error"}
