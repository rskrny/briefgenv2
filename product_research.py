# product_research.py
"""
Product-agnostic research pipeline.

1) Web search (DuckDuckGo) across ANY domain.
2) Fetch & parse pages:
   - JSON-LD Product blocks (@type Product, Offers, additionalProperty)
   - Tables, lists, headings, short paragraphs
3) Extract generic specs & features with regex + lightweight rules
4) If still thin, use LLM on the aggregated page text (JSON-only)
5) If web fails, fall back to vision/OCR hints to guarantee ≥4 features

Returns a normalized bundle the rest of the app can use safely.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import re, json

# Optional deps (we degrade gracefully when missing)
try:
    from duckduckgo_search import DDGS  # pip install duckduckgo-search
except Exception:
    DDGS = None

try:
    import requests
    from bs4 import BeautifulSoup
except Exception:
    requests, BeautifulSoup = None, None

try:
    from llm import gemini_json  # for text-only fallback extraction
except Exception:
    gemini_json = None

HEADERS = {"User-Agent": "briefgenv2/1.0 (+https://github.com/)"}

# --------------------------
# Utilities
# --------------------------
def _dedupe(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for s in seq:
        s = (s or "").strip()
        k = s.lower()
        if s and k not in seen:
            seen.add(k)
            out.append(s)
    return out

def _short_enough(s: str, max_words: int = 22) -> bool:
    w = s.split()
    return 3 <= len(w) <= max_words

def _looks_safe(line: str) -> bool:
    low = line.lower()
    # avoid obvious marketing superlatives & forbidden certifications
    risky = [
        r"\b(best|#1|guarantee|miracle|cure|world[-\s]?class)\b",
        r"\b(FDA|CE|UL|ISO|EPA)\b",
    ]
    if any(re.search(p, low) for p in risky):
        return False
    return True

# --------------------------
# Web search + fetch
# --------------------------
def _ddg_search(query: str, max_results: int = 8) -> List[Dict[str, str]]:
    if not DDGS:
        return []
    hits = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                url = r.get("href") or r.get("url")
                title = r.get("title") or r.get("body") or url
                if not url:
                    continue
                hits.append({"title": title, "url": url})
    except Exception:
        pass
    return hits

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

# --------------------------
# HTML parsing
# --------------------------
def _jsonld_blocks(html: str) -> List[dict]:
    if not BeautifulSoup:
        return []
    soup = BeautifulSoup(html, "html.parser")
    data = []
    for tag in soup.find_all("script", {"type": "application/ld+json"}):
        try:
            txt = tag.string or tag.text or ""
            if not txt.strip():
                continue
            obj = json.loads(txt)
            if isinstance(obj, dict):
                data.append(obj)
            elif isinstance(obj, list):
                data.extend([x for x in obj if isinstance(x, dict)])
        except Exception:
            continue
    return data

def _texts_from_dom(html: str) -> List[str]:
    if not BeautifulSoup:
        return []
    soup = BeautifulSoup(html, "html.parser")
    texts = []

    # Lists, table rows, definition lists
    for li in soup.select("li"):
        t = " ".join(li.get_text(" ").split())
        if _short_enough(t):
            texts.append(t)
    for tr in soup.select("table tr"):
        t = " ".join(tr.get_text(" ").split())
        if _short_enough(t):
            texts.append(t)
    for d in soup.select("dl > *"):
        t = " ".join(d.get_text(" ").split())
        if _short_enough(t):
            texts.append(t)

    # Short paragraphs & headings
    for p in soup.select("p"):
        t = " ".join(p.get_text(" ").split())
        if _short_enough(t, 28):
            texts.append(t)
    for h in soup.select("h1, h2, h3, h4"):
        t = " ".join(h.get_text(" ").split())
        if 2 <= len(t.split()) <= 14:
            texts.append(t)

    return texts[:250]

# --------------------------
# Structured (JSON-LD) → specs/features
# --------------------------
def _from_jsonld(jsonlds: List[dict]) -> Tuple[List[str], List[str], Dict[str, str]]:
    features, specs = [], {}
    for block in jsonlds:
        typ = block.get("@type") or block.get("type")
        if isinstance(typ, list):
            typ = ",".join(typ)
        if typ and "Product" not in str(typ):
            continue

        # description / featureList
        d = block.get("description") or ""
        if isinstance(d, str) and _short_enough(d, 40):
            features.append(d.strip())

        for key in ("featureList", "features"):
            fl = block.get(key)
            if isinstance(fl, list):
                for x in fl:
                    if isinstance(x, str) and _short_enough(x, 16):
                        features.append(x.strip())

        # additionalProperty (common in schema.org Product)
        addp = block.get("additionalProperty")
        if isinstance(addp, list):
            for prop in addp:
                name = (prop.get("name") or "").strip()
                val = prop.get("value")
                if isinstance(val, dict):
                    val = val.get("value") or val.get("name") or ""
                val = (val or "").strip()
                if name and val:
                    specs[name] = val
    return _dedupe(features), specs

# --------------------------
# Generic regex-driven spec extraction from text
# --------------------------
SPEC_PATTERNS = [
    ("weight", r"(?:^|\b)(?:item\s*)?weight[:\s-]*([\d\.]+)\s*(kg|g|lbs|lb|pounds|oz)\b"),
    ("dimensions", r"(?:^|\b)(?:dimensions|size)[:\s-]*([\d\.\sx×x]+)\s*(cm|mm|in|\"|')?"),
    ("capacity", r"(?:^|\b)(?:capacity|volume)[:\s-]*([\d\.]+)\s*(l|liters|ml|oz|mah|gb|tb)\b"),
    ("battery_life", r"(?:^|\b)(?:battery\s*(?:life|runtime|playtime))[:\s-]*([\d\.]+)\s*(h|hr|hrs|hours|minutes|min)\b"),
    ("power", r"(?:^|\b)(?:power|wattage|output)[:\s-]*([\d\.]+)\s*(w|kw|v|volts|amps|a)\b"),
    ("screen", r"(?:^|\b)(?:screen|display|resolution)[:\s-]*([\d]{3,4}\s?[x×]\s?[\d]{3,4}|[\d\.]+\s?(?:in|inch|\"))"),
    ("ip_rating", r"\bip\s?([0-9]{2})\b"),
]

def _extract_specs_from_lines(lines: List[str]) -> Dict[str, str]:
    specs = {}
    for ln in lines:
        low = ln.lower()
        for key, pat in SPEC_PATTERNS:
            m = re.search(pat, low)
            if m and key not in specs:
                specs[key] = " ".join([x for x in m.groups() if x]).strip()
    return specs

def _extract_features_from_lines(lines: List[str]) -> List[str]:
    feats = []
    for ln in lines:
        if not _looks_safe(ln):
            continue
        if _short_enough(ln, 14):
            # favor typical bullet/short phrases
            feats.append(ln.strip("•- ").strip())
    return _dedupe(feats)

def _extract_disclaimers(lines: List[str]) -> List[str]:
    out = []
    for ln in lines:
        low = ln.lower()
        if any(k in low for k in ["do not", "caution", "warning", "not intended", "read instructions", "follow warnings"]):
            if _short_enough(ln, 26):
                out.append(ln.strip())
    return _dedupe(out)

# --------------------------
# LLM fallback (text-only)
# --------------------------
def _llm_extract_json(texts: List[str]) -> Dict[str, Any]:
    if not gemini_json:
        return {}
    joined = " ".join(texts)[:8000]  # cap tokens
    prompt = f"""
You are extracting product information from messy web copy.
Return JSON ONLY with:
{{
  "features": ["≤12 short bullets, product-agnostic, no superlatives"],
  "specs": {{"weight":"","dimensions":"","capacity":"","battery_life":"","power":"","screen":"","ip_rating":""}},
  "disclaimers": ["0–6 short safety/usage disclaimers"]
}}
Constraints:
- Keep features factual and non-comparative (no "#1", "best", etc.).
- Prefer concise phrases (≤10 words each).
- Only fill specs you find (leave others empty strings).
TEXT:
{joined}
"""
    out = gemini_json(prompt)
    if not isinstance(out, dict):
        return {}
    out.setdefault("features", [])
    out.setdefault("specs", {})
    out.setdefault("disclaimers", [])
    # normalize
    out["features"] = [s for s in out["features"] if isinstance(s, str) and s.strip()]
    out["disclaimers"] = [s for s in out["disclaimers"] if isinstance(s, str) and s.strip()]
    if not isinstance(out["specs"], dict):
        out["specs"] = {}
    return out

# --------------------------
# Public entry
# --------------------------
def research_product(
    brand: str,
    product: str,
    *,
    product_url_override: str = "",
    ocr_hints: List[str] | None = None,
    vision_hints: List[str] | None = None,
    max_results: int = 8,
) -> Dict[str, Any]:
    """
    Returns
    {
      "query": "...",
      "sources": [{"title","url"}, ...],
      "features": [...],            # bullets, safe language
      "specs": {"weight": "...", ...},
      "disclaimers": [...],
      "claims": [...]               # extra clean sentences (optional)
    }
    """
    bundle = {
        "query": "",
        "sources": [],
        "features": [],
        "specs": {},
        "disclaimers": [],
        "claims": [],
    }

    # 0) Build broad query; include hints
    hints = " ".join((ocr_hints or [])[:8] + (vision_hints or [])[:8])
    q = f"{brand} {product} features specs {hints}".strip()
    bundle["query"] = q

    # 1) Candidate URLs
    urls: List[Dict[str, str]] = []
    if product_url_override:
        urls = [{"title": "Provided URL", "url": product_url_override}]
    else:
        urls = _ddg_search(q, max_results=max_results)

    # 2) Parse pages
    all_lines: List[str] = []
    jsonld_features: List[str] = []
    jsonld_specs: Dict[str, str] = {}

    for h in urls[:max_results]:
        html = _fetch_html(h["url"])
        if not html:
            continue
        jsonlds = _jsonld_blocks(html)
        f2, s2 = _from_jsonld(jsonlds)
        if f2:
            jsonld_features.extend(f2)
        if s2:
            jsonld_specs.update({k.lower(): v for k, v in s2.items()})

        lines = _texts_from_dom(html)
        all_lines.extend(lines)

    # 3) Heuristic extraction from lines
    specs = _extract_specs_from_lines(all_lines)
    specs.update({k: v for k, v in jsonld_specs.items() if k not in specs})
    feats = _extract_features_from_lines(all_lines) + jsonld_features
    discls = _extract_disclaimers(all_lines)

    # 4) If still thin, ask LLM to structure from raw text
    if len(feats) < 4 or not specs:
        llm_out = _llm_extract_json(all_lines)
        feats = _dedupe(feats + llm_out.get("features", []))
        for k, v in (llm_out.get("specs") or {}).items():
            if v and k not in specs:
                specs[k] = v
        discls = _dedupe(discls + (llm_out.get("disclaimers") or []))

    # 5) If still thin, use vision/OCR hints to guarantee ≥4 bullets
    floor_feats = _dedupe((feats or []) + (vision_hints or []) + (ocr_hints or []))
    while len(floor_feats) < 4:
        floor_feats.append("notable design feature")  # last-resort neutral filler
    feats = floor_feats[:20]

    # 6) Claims (optional, keep extra clean sentences)
    claims = [ln for ln in all_lines if _looks_safe(ln) and _short_enough(ln, 16)][:20]

    bundle["sources"] = urls
    bundle["features"] = feats
    bundle["specs"] = specs
    bundle["disclaimers"] = discls[:10]
    bundle["claims"] = claims
    return bundle
