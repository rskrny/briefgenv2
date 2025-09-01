# gemini_fetcher.py — 2025-09-01 (uses google_search_retrieval tool)
"""
Fetch product specs/features via Gemini with Google Search Retrieval.

Prereqs:
- env var GOOGLE_API_KEY is set (Streamlit secrets covers this).
- google-generativeai >= 0.7.0

Behavior:
- Tries gemini-1.5-pro; if unavailable, falls back to gemini-1.5-flash.
- Enables the Google Search Retrieval tool so Gemini can look up pages.
- Returns a strict JSON dict or {"status": "ERROR", "error": "..."} on failure.
"""

from __future__ import annotations
import os, json, logging
import google.generativeai as genai

log = logging.getLogger(__name__)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

SYSTEM = (
    "You are a strictly factual Product-Spec Retriever with Google-search access.\n"
    "RULES:\n"
    "• Use Google Search to find authoritative pages: brand PDP first, then major retailers, then trusted review sites.\n"
    "• Accept a spec/feature only if it appears in at least TWO independent sources.\n"
    "• Write features as short bullets (<=120 chars each), 4–10 total.\n"
    "• Output ONLY the JSON object that matches the schema exactly. No markdown, no extra keys.\n"
    "• If you cannot confirm at least THREE specs, return {\"status\":\"NOT_FOUND\"}."
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
Return ONLY JSON.
"""

CANDIDATE_MODELS = ["gemini-1.5-pro", "gemini-1.5-flash"]


def _make_model(name: str):
    """
    Construct a GenerativeModel with Google Search Retrieval enabled.
    """
    return genai.GenerativeModel(
        name=name,
        tools=[{"google_search_retrieval": {}}],  # <-- correct tool name
        generation_config={"temperature": 0.2, "max_output_tokens": 1200},
        safety_settings={"HARASSMENT": "block_none"},
    )


def _strip_code_fence(text: str) -> str:
    if not text:
        return text
    t = text.strip()
    if t.startswith("```"):
        # remove surrounding fences; also drop a leading 'json' tag if present
        t = t.strip("` \n")
        if t.lower().startswith("json"):
            t = t[4:].lstrip()
    return t


def gemini_product_info(brand: str, model: str, timeout_s: int = 60) -> dict:
    """
    Returns parsed JSON from Gemini or {'status':'ERROR', ...} on failure.
    """
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
