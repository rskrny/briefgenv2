# product_research.py
# Scrape product pages → summarize to proposed claims (Gemini) → return structured JSON

from __future__ import annotations
from typing import List, Dict
import re
import requests
from readability import Document
from bs4 import BeautifulSoup

from llm import gemini_json

def fetch_pages(urls: List[str]) -> List[Dict]:
    docs: List[Dict] = []
    for url in urls:
        if not url or not url.strip():
            continue
        try:
            r = requests.get(url, timeout=15, headers={"User-Agent":"Mozilla/5.0"})
            r.raise_for_status()
            html = r.text
            doc = Document(html)
            main_html = doc.summary() or html
            soup = BeautifulSoup(main_html, "lxml")
            text = soup.get_text("\n", strip=True)
            # light cleanup
            text = re.sub(r"\n{2,}", "\n", text)
            docs.append({"url": url, "text": text[:12000]})  # cap per page
        except Exception:
            docs.append({"url": url, "text": ""})
    return docs

def summarize_to_claims(brand: str, product: str, docs: List[Dict], model: str = "gemini-1.5-pro") -> Dict:
    """
    Returns JSON with fields:
      product {brand,name,aliases}, key_features[], differentiators[], evidence_quotes[], proposed_claims[], risks[], required_disclaimers[]
    """
    # Build a compact prompt (no external dependencies)
    docs_json = [{"source": d["url"], "excerpt": d["text"]} for d in docs if d.get("text")]
    prompt = f"""
You are a compliance-aware product researcher. Read the input documents and propose factual, defensible claims.

Return JSON ONLY with this exact shape:
{{
  "product": {{"brand":"", "name":"", "aliases":[]}},
  "key_features": [],
  "differentiators": [],
  "evidence_quotes": [{{"text":"", "source":""}}],
  "proposed_claims": [],
  "risks": [],
  "required_disclaimers": []
}}

Rules:
- Only use facts present in the documents; do not invent.
- Phrase proposed_claims in simple, ad-safe language; no superlatives unless proven.
- Add a few short required_disclaimers if needed (e.g., battery life may vary).
- Keep lists short (3–8 bullets each).

Brand: {brand}
Product: {product}
Documents (JSON): {docs_json}
"""
    out = gemini_json(prompt, model=model, temperature=0.1)
    return safe_parse_json(out)

def safe_parse_json(s: str) -> Dict:
    import json
    try:
        return json.loads(s)
    except Exception:
        return {}
