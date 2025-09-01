# gemini_fetcher.py — 2025-09-01
"""
Single call to Gemini-Pro that searches the web and returns product specs &
features in a strict JSON schema.

Prereq: GOOGLE_API_KEY is set (Streamlit secrets already covers this).
"""

from __future__ import annotations
import os, json, logging
import google.generativeai as genai

log = logging.getLogger(__name__)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

SYSTEM = (
    "You are a strictly factual Product-Spec Retriever with Google-search access. "
    "Given BRAND and MODEL, you will:\n"
    "1. Open the most relevant web pages (brand PDP first, major retailer or "
    "trusted review sites second).\n"
    "2. Accept a spec or feature only if it appears in at least TWO independent sources.\n"
    "3. Summarise features as short bullets (≤120 characters each).\n"
    "4. Return ONLY the JSON object matching the schema below. "
    "If you cannot confirm at least THREE specs, return "
    '{"status":"NOT_FOUND"} and nothing else.'
)

SCHEMA = """{
  "status": "OK",
  "specs": { "battery_life_h": "8", "...": "..." },
  "features": ["...", "..."],
  "citations": [
    { "attr": "battery_life_h", "url": "https://..." },
    { "attr": "feature_0", "url": "https://..." }
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
-- Remember: no markdown, no extra keys.
"""

def gemini_product_info(brand: str, model: str, timeout_s: int = 45) -> dict:
    """Returns parsed JSON from Gemini or {'status':'ERROR', ...} on failure."""
    prompt = PROMPT_TEMPLATE.format(system=SYSTEM, brand=brand, model=model, schema=SCHEMA)
    model_g = genai.GenerativeModel("gemini-pro")
    try:
        resp = model_g.generate_content(
            prompt,
            generation_config={"temperature": 0.2, "max_output_tokens": 1024},
            safety_settings={"HARASSMENT": "block_none"},
            request_options={"timeout": timeout_s},
        )
        data = json.loads(resp.text)
        return data
    except Exception as exc:
        log.warning("Gemini fetch failed: %s", exc, exc_info=False)
        return {"status": "ERROR", "error": str(exc)}
