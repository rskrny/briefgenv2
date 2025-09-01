# gemini_fetcher.py — 2025-09-01 (category-aware, product-only extraction)
"""
Fetch product specs/features via Gemini with Google Search Retrieval,
constrained by an explicit category hint (e.g., 'headphones').

Env:
  GOOGLE_API_KEY  (from Streamlit secrets)
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

SYSTEM_BASE = (
    "You are a strictly factual Product-Spec Retriever with Google Search access.\n"
    "OBJECTIVE: Given BRAND and MODEL, return a verified, concise JSON with only product information.\n"
    "HARD RULES:\n"
    "1) Open authoritative pages: the BRAND's product page first, then major retailers "
    "(Amazon/Walmart/BestBuy/Target/B&H/REI), then trusted review sites.\n"
    "2) Accept a spec/feature only if it appears in ≥2 independent sources.\n"
    "3) REJECT any content that looks like UI text, keyboard shortcuts, menus, breadcrumbs, or generic help.\n"
    "4) Before extracting, confirm the page title mentions BOTH brand and model (or clearly the same product).\n"
    "5) Keep features short (≤120 chars). No fluff or duplicates.\n"
    "6) Output ONLY JSON. If you cannot confirm ≥3 specs, output {\"status\":\"NOT_FOUND\"}."
)

SCHEMA = """{
  "status": "OK",
  "product_title": "Brand Model ...",
  "category": "e.g., headphones",
  "specs": { "weight_g": "250", "...": "..." },
  "features": ["...", "..."],
  "citations": [
    { "attr": "weight_g", "url": "https://..." },
    { "attr": "feature_0", "url": "https://..." }
  ],
  "pages_used": [
    {"title":"...", "url":"https://..."},
    {"title":"...", "url":"https://..."}
  ]
}"""

PROMPT_TEMPLATE = """
### SYSTEM
{system}

### USER
Brand: {brand}
Model: {model}
Desired Category: {category_hint}

STRICT CATEGORY CONSTRAINT:
- Only return results for items that are {category_hint}.
- If the web results point to a different category (e.g., 'portable power station'),
  return {{"status":"NOT_FOUND"}}.

JSON schema you MUST follow exactly:
{schema}
Return ONLY JSON. Reject UI/shortcut/menu/breadcrumb content.
"""

def _system_with_category(category_hint: str) -> str:
    # Add category-specific acceptance/denial lists to reduce cross-category matches.
    cat = (category_hint or "").strip().lower()
    deny_examples = ""
    allow_markers = ""

    if cat in {"headphones", "earbuds", "headset"}:
        allow_markers = (
            "Signals to ACCEPT: 'headphones', 'earbuds', 'headset', 'Bluetooth', "
            "'ANC', 'audio drivers', 'impedance', 'frequency response'."
        )
        deny_examples = (
            "If the page contains terms like 'power station', 'generator', 'inverter', "
            "'Wh', 'AC outlets', 'solar charging', REJECT it."
        )
    elif cat in {"portable power station", "generator"}:
        allow_markers = (
            "Signals to ACCEPT: 'portable power station', 'generator', 'AC outlets', "
            "'Wh', 'LiFePO4', 'inverter', 'solar input'."
        )
        deny_examples = (
            "If the page contains only 'headphones', 'earbuds', 'headset', or audio specs, REJECT it."
        )
    else:
        allow_markers = "Prefer results explicitly stating the same category as Desired Category."
        deny_examples = "Reject pages that clearly indicate a different physical product category."

    return SYSTEM_BASE + "\n" + allow_markers + "\n" + deny_examples

def _make_model(model_name: str):
    return genai.GenerativeModel(
        model_name=model_name,
        tools=[{"google_search_retrieval": {}}],
        generation_config={"temperature": 0.2, "max_output_tokens": 1400},
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
    cat_hint = (category_hint or "").strip() or "product (match brand+model)"
    system = _system_with_category(cat_hint)
    prompt = PROMPT_TEMPLATE.format(system=system, brand=brand, model=model, category_hint=cat_hint, schema=SCHEMA)

    last_err = None
    for m in CANDIDATE_MODELS:
        try:
            gmodel = _make_model(m)
            resp = gmodel.generate_content(prompt, request_options={"timeout": timeout_s})
            text = _strip_code_fence(resp.text or "")
            data = json.loads(text)
            return data
        except Exception as exc:
            last_err = exc
            log.warning("Gemini call failed on %s: %s", m, exc, exc_info=False)
            continue

    return {"status": "ERROR", "error": str(last_err) if last_err else "Unknown Gemini error"}
