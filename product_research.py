# product_research.py
# Auto-search + scrape product pages → summarize to proposed claims (Gemini)

from __future__ import annotations
from typing import List, Dict, Tuple
import re
import time
import urllib.parse
import requests
from readability import Document
from bs4 import BeautifulSoup

from llm import gemini_json


# ---------------------------
# Web search (no API key)
# ---------------------------
def ddg_search(query: str, k: int = 6, timeout: int = 12) -> List[str]:
    """
    DuckDuckGo HTML search (no key). Returns a list of external result URLs.
    We de-dupe and filter obvious redirect/track links.
    """
    url = "https://duckduckgo.com/html/"
    params = {"q": query}
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        r.raise_for_status()
    except Exception:
        return []

    soup = BeautifulSoup(r.text, "lxml")

    links: List[str] = []
    # DuckDuckGo HTML often puts results in <a class="result__a"> or <a rel="nofollow" class="result__url">
    for a in soup.select("a.result__a, a.result__url"):
        href = a.get("href") or ""
        if not href:
            continue
        # Try to decode /l/?uddg= wrapped URLs
        if "/l/?" in href and "uddg=" in href:
            try:
                qs = urllib.parse.urlparse(href).query
                uddg = urllib.parse.parse_qs(qs).get("uddg", [""])[0]
                if uddg:
                    href = urllib.parse.unquote(uddg)
            except Exception:
                pass
        if href.startswith("http"):
            links.append(href)

    # De-dupe, drop obvious junk
    clean = []
    seen = set()
    for u in links:
        if any(host in u for host in ["duckduckgo.com", "google.com/search"]):
            continue
        key = re.sub(r"https?://(www\.)?", "", u).rstrip("/")
        if key not in seen:
            seen.add(key)
            clean.append(u)
    return clean[:k]


# ---------------------------
# Fetch & clean pages
# ---------------------------
def fetch_pages(urls: List[str], per_page_char_cap: int = 12000) -> List[Dict]:
    docs: List[Dict] = []
    headers = {"User-Agent": "Mozilla/5.0"}
    for url in urls:
        if not url or not url.strip():
            continue
        try:
            r = requests.get(url, timeout=15, headers=headers)
            r.raise_for_status()
            html = r.text
            doc = Document(html)
            main_html = doc.summary() or html
            soup = BeautifulSoup(main_html, "lxml")
            text = soup.get_text("\n", strip=True)
            text = re.sub(r"\n{2,}", "\n", text)
            docs.append({"url": url, "text": text[:per_page_char_cap]})
        except Exception:
            docs.append({"url": url, "text": ""})
        # be polite
        time.sleep(0.4)
    return docs


# ---------------------------
# Product research orchestrator
# ---------------------------
def auto_collect_product_docs(brand: str, product: str, extra_urls: List[str] | None = None) -> Tuple[List[Dict], List[str]]:
    """
    Search the web for brand+product sources. Optionally merge user URLs.
    Returns (docs, sources_used).
    """
    extra_urls = [u.strip() for u in (extra_urls or []) if u.strip()]

    # Craft a search query that tends to pull official/spec/review pages
    q = f'{brand} {product} specs site:{brand.lower()}.com OR "{brand} {product}" review OR retailer listing'
    candidates = ddg_search(q, k=7)

    # If we got nothing, fall back to a simpler query
    if not candidates:
        candidates = ddg_search(f"{brand} {product} product features", k=6)

    # Merge with user-provided URLs (if any), preserving order and uniqueness
    merged = []
    seen = set()
    for u in (extra_urls + candidates):
        key = re.sub(r"https?://(www\.)?", "", u).rstrip("/")
        if key not in seen:
            seen.add(key)
            merged.append(u)

    docs = fetch_pages(merged)
    sources = [d["url"] for d in docs if d.get("text")]
    return docs, sources


def summarize_to_claims(brand: str, product: str, docs: List[Dict], model: str = "gemini-1.5-pro") -> Dict:
    """
    Returns JSON with fields:
      product {brand,name,aliases}, key_features[], differentiators[], evidence_quotes[], proposed_claims[], risks[], required_disclaimers[], confidence, source_suggestions[]
    If docs are empty/weak, the model must still produce a best-effort summary with lower confidence and suggestions to verify.
    """
    docs_json = [{"source": d["url"], "excerpt": d["text"]} for d in docs if d.get("text")]

    prompt = f"""
You are a compliance-aware product researcher. Use the provided documents if present; if not, infer cautiously from general knowledge and clearly lower confidence.

Return JSON ONLY with this exact shape:
{{
  "product": {{"brand":"", "name":"", "aliases":[]}},
  "key_features": [],
  "differentiators": [],
  "evidence_quotes": [{{"text":"", "source":""}}],
  "proposed_claims": [],
  "risks": [],
  "required_disclaimers": [],
  "confidence": "high|medium|low",
  "source_suggestions": []
}}

Rules:
- Prefer factual claims directly supported by the documents.
- When a claim is based on general knowledge (because documents are missing), keep it conservative and mark overall "confidence" accordingly.
- Phrase proposed_claims in simple, ad-safe language; avoid superlatives unless proven.
- If battery/charging/safety are mentioned, include short required_disclaimers (e.g., "Battery life varies with use").
- Include 2–5 short evidence_quotes with associated "source" URLs when available.
- Keep lists concise (3–8 bullets each).

Brand: {brand}
Product: {product}
Documents (JSON, may be empty): {docs_json}
"""
    raw = gemini_json(prompt, model=model, temperature=0.1)
    return safe_parse_json(raw)


def safe_parse_json(s: str) -> Dict:
    import json
    try:
        return json.loads(s)
    except Exception:
        return {}
