# gemini_fetcher.py — 2025-09-01 (hardened for product-only extraction)
"""
Fetch product specs/features via Gemini with Google Search Retrieval.

Key changes:
- Stricter instructions to avoid UI/shortcut/breadcrumb noise.
- Requires brand/model match in page title and a real product category.
- Model fallback: gemini-1.5-pro → gemini-1.5-flash.
"""

from __future__ import annotations
import os, json, logging
import google.generativeai as genai

log = logging.getLogger(__name__)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

SYSTEM = (
    "You are a strictly factual Product-Spec Retriever with Google search access.\n"
    "OBJECTIVE: Given BRAND and MODEL, return a verified, concise JSON with only product information.\n"
    "HARD RULES:\n"
    "1) Open authoritative pages: the BRAND's product page first, then major retailers "
    "(Amazon/Walmart/BestBuy/Target/B&H), then trusted review sites.\n"
    "2) Accept a spec/feature only if it appears in ≥2 independent sources.\n"
    "3) REJECT any content that looks like UI text, keyboard shortcuts, menus, breadcrumbs, site categories, "
    "or generic help. Examples to REJECT: 'Search alt+/', 'Home shift+H', 'Chairs', 'Camping & Hiking', 'Orders', 'Cart'.\n"
    "4) Before extracting, confirm the page title mentions BOTH brand and model (or clearly the same product), "
    "and the page has a concrete product category (e.g., 'camping chair').\n"
    "5) Keep features as short bullets (≤120 chars). No marketing fluff or duplicates.\n"
    "6) Output ONLY JSON, no markdown, exactly in the schema. If you cannot confirm ≥3 specs, output {\"status\":\"NOT_FOUND\"}.\n"
)

SCHEMA = """{
  "status": "OK",
  "product_title": "Brand Model ...",
  "category": "e.g., camping chair",
  "specs": { "weight_g": "166.34", "dimensions_cm": "52 x 49 x 62", "...": "..." },
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
JSON schema you MUST follow exactly:
{schema}
Return ONLY JSON. Reject UI/shortcut/menu/breadcrumb content.
"""

CANDIDATE_MODELS = ["gemini-1.5-pro", "gemini-1.5-flash"]


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


def gemini_product_info(brand: str, model: str, timeout_s: int = 60) -> dict:
    prompt = PROMPT_TEMPLATE.format(system=SYSTEM, brand=brand, model=model, schema=SCHEMA)
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
