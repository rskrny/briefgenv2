# product_research.py
"""
Lightweight product research:
- Searches the web for the brand/product.
- Fetches a handful of pages and extracts likely feature claims & disclaimers.
- Returns a bundle for downstream use (UI, script prompt, PDF).

Notes:
- Uses duckduckgo-search (no API key). On hosts without egress, it will gracefully return empty results.
- Heuristics are intentionally conservative: short, non-comparative, non-metric-heavy claims only.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import re

try:
    from duckduckgo_search import DDGS  # pip install duckduckgo-search
except Exception:
    DDGS = None

try:
    import requests
    from bs4 import BeautifulSoup
except Exception:
    requests, BeautifulSoup = None, None


SAFE_WORDS = {
    # general feature-y terms we allow
    "lightweight", "portable", "folding", "foldable", "packable", "compact",
    "durable", "sturdy", "stable", "breathable", "water-resistant", "waterproof",
    "adjustable", "reclining", "recliner", "comfortable", "comfort", "support",
    "cup holder", "carry bag", "quick setup", "easy setup", "easy to clean",
}

RISKY_PATTERNS = [
    r"\b(safest|best|#1|guarantee|cure|miracle)\b",
    r"\b(\d{2,}% better|\d{2,}% more|\d+x)\b",
    r"\b(FDA|CE|UL|ISO)\b",  # certifications we don't want to invent
]

DISCLAIMER_PATTERNS = [
    r"\b(do not exceed|max(?:imum)? weight|weight capacity)\b",
    r"\b(read (the )?instructions\b|\bfollow (all )?warnings\b)",
    r"\b(not (a|intended) (medical|safety) device)\b",
]

HEADERS = {"User-Agent": "briefgenv2/1.0 (+https://github.com/)"}


def _clean_text(t: str) -> str:
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _fetch_html(url: str, timeout: int = 12) -> Optional[str]:
    if not requests:
        return None
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        if "text/html" not in r.headers.get("Content-Type", ""):
            return None
        return r.text
    except Exception:
        return None


def _extract_sentences(text: str) -> List[str]:
    # very light sentence splitter
    bits = re.split(r"(?<=[.!?])\s+", text)
    lines = []
    for b in bits:
        b = _clean_text(b)
        if 6 <= len(b) <= 180:
            lines.append(b)
    return lines


def _looks_like_claim(line: str) -> bool:
    low = line.lower()
    if any(re.search(p, low) for p in RISKY_PATTERNS):
        return False
    # avoid comparative "better than" etc.
    if " than " in low or " vs " in low:
        return False
    # prefer short, feature-like sentences
    if len(line.split()) > 20:
        return False
    # must have at least one safe-ish word OR typical feature verbs
    if any(w in low for w in SAFE_WORDS) or any(
        v in low for v in ["features", "includes", "designed", "built-in", "comes with"]
    ):
        return True
    return False


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for it in items:
        key = it.lower()
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out


def _extract_claims_and_disclaimers(html: str) -> Dict[str, List[str]]:
    if not BeautifulSoup:
        return {"claims": [], "disclaimers": [], "features": []}
    soup = BeautifulSoup(html, "html.parser")
    # try to pull list items and short paragraphs
    texts = []
    for li in soup.select("li"):
        s = _clean_text(li.get_text(" "))
        if 3 <= len(s.split()) <= 16:
            texts.append(s)
    for p in soup.select("p"):
        s = _clean_text(p.get_text(" "))
        if 4 <= len(s.split()) <= 20:
            texts.append(s)
    # also h2/h3 sections can carry features
    for h in soup.select("h2, h3, h4"):
        s = _clean_text(h.get_text(" "))
        if 2 <= len(s.split()) <= 12:
            texts.append(s)

    # last resort: short sentences from whole text
    if len(texts) < 10:
        texts.extend(_extract_sentences(_clean_text(soup.get_text(" ")))[:30])

    claims: List[str] = []
    discls: List[str] = []
    features: List[str] = []

    for line in texts:
        low = line.lower()
        if any(re.search(p, low) for p in DISCLAIMER_PATTERNS):
            discls.append(line)
            continue
        if _looks_like_claim(line):
            claims.append(line)
            # treat some as visible features, too
            for w in SAFE_WORDS:
                if w in low:
                    features.append(w)
                    break

    return {
        "claims": _dedupe_keep_order(claims)[:12],
        "disclaimers": _dedupe_keep_order(discls)[:6],
        "features": _dedupe_keep_order(features)[:12],
    }


def research_product(brand: str, product: str, *, max_results: int = 5, broaden_query: bool = True) -> Dict[str, Any]:
    """
    Returns:
      {
        "query": "...",
        "sources": [{"title": "...", "url": "..."}, ...],
        "claims": [...],
        "disclaimers": [...],
        "features": [...]
      }
    """
    bundle: Dict[str, Any] = {
        "query": "",
        "sources": [],
        "claims": [],
        "disclaimers": [],
        "features": [],
    }
    if not DDGS or not requests or not BeautifulSoup:
        return bundle  # offline / restricted environment

    q = f"{brand} {product} features"
    if broaden_query:
        q += " site:official OR site:shop OR review"
    bundle["query"] = q

    try:
        hits = []
        with DDGS() as ddgs:
            for r in ddgs.text(q, max_results=max_results):
                url = r.get("href") or r.get("url")
                title = r.get("title") or r.get("body") or url
                if not url:
                    continue
                hits.append({"title": title, "url": url})
    except Exception:
        hits = []

    bundle["sources"] = hits
    all_claims, all_disc, all_feats = [], [], []

    for h in hits:
        html = _fetch_html(h["url"])
        if not html:
            continue
        ext = _extract_claims_and_disclaimers(html)
        all_claims.extend(ext["claims"])
        all_disc.extend(ext["disclaimers"])
        all_feats.extend(ext["features"])

    bundle["claims"] = _dedupe_keep_order(all_claims)[:20]
    bundle["disclaimers"] = _dedupe_keep_order(all_disc)[:12]
    bundle["features"] = _dedupe_keep_order(all_feats)[:20]
    return bundle
