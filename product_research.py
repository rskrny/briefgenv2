# product_research.py
# Auto-search + scrape product pages → summarize to proposed claims (Gemini)

from __future__ import annotations
from typing import List, Dict, Tuple
import re
import time
import urllib.parse
import html
import requests
from readability import Document
from bs4 import BeautifulSoup

from llm import gemini_json


# ---------------------------
# DuckDuckGo HTML search (no API key)
# ---------------------------
def _ddg_search_once(query: str, k: int = 6, timeout: int = 12) -> List[Dict]:
    """
    Returns a list of {url, title} objects from DuckDuckGo's HTML endpoint.
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
    results: List[Dict] = []

    # DDG HTML has <a class="result__a"> for titles/links
    for a in soup.select("a.result__a"):
        href = a.get("href") or ""
        text = a.get_text(strip=True) or ""
        if not href:
            continue

        # Decode DDG redirect wrapper /l/?uddg=
        if "/l/?" in href and "uddg=" in href:
            try:
                qs = urllib.parse.urlparse(href).query
                uddg = urllib.parse.parse_qs(qs).get("uddg", [""])[0]
                if uddg:
                    href = urllib.parse.unquote(uddg)
            except Exception:
                pass

        if href.startswith("http"):
            results.append({"url": href, "title": text})

    # Fallback: some themes use .result__url
    if not results:
        for a in soup.select("a.result__url"):
            href = a.get("href") or ""
            text = a.get_text(strip=True) or ""
            if href.startswith("http"):
                results.append({"url": href, "title": text})

    # De-dupe by hostname+path (ignore scheme/www)
    seen = set()
    uniq: List[Dict] = []
    for r_ in results:
        key = re.sub(r"^https?://(www\.)?", "", r_["url"]).rstrip("/")
        if key not in seen:
            seen.add(key)
            uniq.append(r_)
    return uniq[:k]


def ddg_multi_search(queries: List[str], k_each: int = 6) -> List[Dict]:
    bag: List[Dict] = []
    seen = set()
    for q in queries:
        hits = _ddg_search_once(q, k=k_each)
        for h in hits:
            key = re.sub(r"^https?://(www\.)?", "", h["url"]).rstrip("/")
            if key not in seen:
                seen.add(key)
                bag.append(h)
        time.sleep(0.35)  # be polite
    return bag


# ---------------------------
# Candidate ranking
# ---------------------------
_SITEMAP_HINTS = ("sitemap", "html-sitemap", "category", "tag", "/pages/", "/blog/page/")

def rank_candidates(cands: List[Dict], product_token: str) -> List[Dict]:
    """
    Score URLs: prefer those that include the product token in URL or title,
    demote sitemaps/index pages.
    """
    token = product_token.lower()
    ranked = []
    for c in cands:
        url = c["url"]
        title = (c.get("title") or "").lower()
        path = re.sub(r"^https?://", "", url).lower()

        score = 0
        if token and (token in path or token in title):
            score += 4
        if any(h in path for h in _SITEMAP_HINTS):
            score -= 3
        if any(x in path for x in ("amazon.", "bestbuy.", "walmart.", "aliexpress.", "alibaba.", "bhphotovideo.")):
            score += 2
        if any(x in path for x in ("review", "spec", "features", "product")):
            score += 1
        c["__score"] = score
        ranked.append(c)

    ranked.sort(key=lambda x: x["__score"], reverse=True)
    return ranked


# ---------------------------
# Fetch & clean pages
# ---------------------------
def fetch_pages(urls: List[str], per_page_char_cap: int = 14000) -> List[Dict]:
    docs: List[Dict] = []
    headers = {"User-Agent": "Mozilla/5.0"}
    for url in urls:
        if not url or not url.strip():
            continue
        try:
            r = requests.get(url, timeout=15, headers=headers)
            r.raise_for_status()
            html_raw = r.text
            doc = Document(html_raw)
            main_html = doc.summary() or html_raw
            soup = BeautifulSoup(main_html, "lxml")
            # Remove scripts/nav/footers quickly
            for bad in soup(["script", "style", "noscript", "header", "footer", "nav"]):
                bad.extract()
            text = soup.get_text("\n", strip=True)
            text = re.sub(r"\n{2,}", "\n", text)
            # Discard super­short pages or pure sitemaps
            if len(text) < 600 or "sitemap" in url.lower():
                continue
            docs.append({"url": url, "text": text[:per_page_char_cap]})
        except Exception:
            pass
        time.sleep(0.35)
    return docs


# ---------------------------
# Product research orchestrator
# ---------------------------
def auto_collect_product_docs(brand: str, product: str, extra_urls: List[str] | None = None) -> Tuple[List[Dict], List[str]]:
    """
    Build several queries, merge, rank, fetch → return (docs, sources_used).
    """
    brand = (brand or "").strip()
    product = (product or "").strip()
    extra_urls = [u.strip() for u in (extra_urls or []) if u.strip()]

    # 1) Multi-query search
    brand_domain = re.sub(r"[^a-z0-9\-]", "", brand.lower()) + ".com" if brand else ""
    queries = [
        f'"{brand} {product}"',
        f'"{brand} {product}" site:{brand_domain}' if brand_domain != ".com" else f'"{brand} {product}"',
        f'{brand} {product} specifications',
        f'{brand} {product} review',
        f'{product} {brand} buy',
    ]
    raw_hits = ddg_multi_search(queries, k_each=6)

    # 2) Merge with user URLs
    for u in extra_urls:
        raw_hits.append({"url": u, "title": u})

    # 3) Rank
    ranked = rank_candidates(raw_hits, product_token=product)
    top_urls = [r["url"] for r in ranked[:10]]

    # 4) Fetch & clean
    docs = fetch_pages(top_urls)
    sources = [d["url"] for d in docs]
    return docs, sources


# ---------------------------
# Summarize to claims via Gemini
# ---------------------------
def summarize_to_claims(brand: str, product: str, docs: List[Dict], model: str = "gemini-1.5-pro") -> Dict:
    """
    Returns JSON with fields:
      product {brand,name,aliases}, key_features[], differentiators[], evidence_quotes[], proposed_claims[], risks[], required_disclaimers[], confidence, source_suggestions[]
    Must always return at least 3 proposed_claims; if documents are weak, set confidence="low" and keep language conservative.
    """
    docs_json = [{"source": d["url"], "excerpt": d["text"]} for d in docs if d.get("text")]

    prompt = f"""
You are a compliance-aware product researcher. Use the provided documents if present; if not, infer cautiously from general knowledge and lower confidence.

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

Strict rules:
- If documents are strong, base claims on them and add 2–5 short evidence_quotes (with source URLs).
- If documents are weak/empty: still return 3–6 conservative proposed_claims typical for this product category, but set confidence="low" and include concrete source_suggestions to verify (e.g., official product page, major retailers).
- Avoid superlatives unless quotes clearly support them.
- Prefer short, ad-safe phrasing (no medical/health implications).
- Keep each list 3–8 bullets.

Brand: {brand}
Product: {product}
Documents (JSON, may be empty): {docs_json}
"""
    raw = gemini_json(prompt, model=model, temperature=0.1)
    return _safe_parse_json(raw)


def _safe_parse_json(s: str) -> Dict:
    import json
    try:
        return json.loads(s)
    except Exception:
        return {}
