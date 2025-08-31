# product_research.py
"""
Robust, product-agnostic research with provenance & consensus.

Steps:
  1) Candidate discovery (multi-query: site, specs, manual, datasheet, filetype:pdf)
  2) Fetch HTML (requests; optional Playwright render via fetcher.get_html)
  3) Rank pages with product heuristics; parse JSON-LD + main content only
  4) Extract candidate claims (features/specs/disclaimers) from product-like sections
  5) Parse PDF manuals/datasheets for hard specs (pdf_spec_extractor)
  6) Consolidate claims with consensus (manufacturer > majority; unit-normalized numbers)
  7) Floor with vision/OCR hints only for neutral visual features (no numbers)

Returns (normalized):
{
  "query": str,
  "sources": [{"title","url"}],
  "features": [{"text","sources":[{"url","snippet"}], "confidence":0..1}],
  "specs": [{"key","value","sources":[{"url","snippet"}], "confidence":0..1}],
  "disclaimers": [{"text","sources":[...] , "confidence":0..1}],
  "claims": [],               # (kept small; not used to script)
  "notes":  []
}
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import re, json, os

from consensus import Claim, consolidate_claims
from fetcher import get_html, link_density
from pdf_spec_extractor import discover_spec_pdfs, extract_pdf_specs

# Optional deps – degrade gracefully offline
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
    from readability import Document
except Exception:
    Document = None

HEADERS = {"User-Agent": "briefgenv2/1.0 (+https://github.com/)"}

PRODUCT_SECTIONS_RE = re.compile(r"(specs?|specifications?|features|details|technical|dimensions?|materials?|what'?s in the box)", re.I)
PRODUCT_URL_HINTS = ("/product/", "/products/", "/p/", "/dp/", "/sku/", "/item/")

DROP_HTML_HINTS = r"(privacy|terms|subscribe|newsletter|sitewide|homepage|careers|press|returns|shipping|gift card|promo|sale)"

# -------- search --------
def _ddg_search_multi(queries: List[str], max_results: int = 12) -> List[Dict[str, str]]:
    if not DDGS:
        return []
    hits = []
    try:
        with DDGS() as ddgs:
            per = max(2, max_results // max(1, len(queries)))
            for q in queries:
                for r in ddgs.text(q, max_results=per):
                    url = r.get("href") or r.get("url")
                    title = r.get("title") or r.get("body") or url
                    if not url:
                        continue
                    hits.append({"title": title, "url": url})
    except Exception:
        pass
    # dedupe by URL
    seen = set()
    out = []
    for h in hits:
        k = h["url"]
        if k not in seen:
            seen.add(k)
            out.append(h)
    return out[:max_results]

# -------- utils --------
def _lang_ok(text: str) -> bool:
    return bool(re.search(r"\b(the|and|with|for|of|in|to)\b", text.lower()))

def _is_manufacturer(url: str, brand: str) -> bool:
    # crude check: brand token in domain
    try:
        from urllib.parse import urlparse
        host = urlparse(url).hostname or ""
        brand_token = re.sub(r"[^a-z0-9]", "", brand.lower())
        return brand_token and brand_token in re.sub(r"[^a-z0-9]", "", host.lower())
    except Exception:
        return False

def _score_page(url: str, title: str, html: str, product_name: str) -> float:
    s = 0.0
    u = (url or "").lower()
    t = (title or "").lower()
    h = (html or "").lower()
    if any(x in u for x in PRODUCT_URL_HINTS): s += 2.0
    tokens = [tok for tok in product_name.lower().split() if len(tok) >= 3]
    s += sum(0.4 for tok in tokens if tok in t)
    if "schema.org/product" in h: s += 2.0
    if re.search(r"(add\s+to\s+cart|buy\s+now|sku|model\s*:|specifications?)", h): s += 1.2
    if link_density(h) > 0.25: s -= 1.0
    if re.search(DROP_HTML_HINTS, h): s -= 2.0
    return s

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

def _productish_blocks(html: str) -> List[str]:
    if not BeautifulSoup:
        return []
    soup = BeautifulSoup(html, "html.parser")
    # Remove obvious non-content
    for tag in soup.find_all(["header", "footer", "nav", "aside"]):
        tag.decompose()
    texts: List[str] = []
    # Heuristic: collect text near product headings/lists/tables
    for sec in soup.find_all(["section", "div", "article"]):
        heading = sec.find(re.compile("^h[1-4]$"))
        head_txt = " ".join((heading.get_text(" ") if heading else "").split())
        if head_txt and PRODUCT_SECTIONS_RE.search(head_txt):
            # Gather lists/rows/short paragraphs in this section
            for li in sec.find_all("li"):
                t = " ".join(li.get_text(" ").split())
                if 3 <= len(t.split()) <= 22:
                    texts.append(t)
            for tr in sec.find_all("tr"):
                t = " ".join(tr.get_text(" ").split())
                if 3 <= len(t.split()) <= 22:
                    texts.append(t)
            for p in sec.find_all("p"):
                t = " ".join(p.get_text(" ").split())
                if 3 <= len(t.split()) <= 22:
                    texts.append(t)
    # fallback: Readability main content
    if Document:
        try:
            doc = Document(html)
            main_html = doc.summary()
            main = BeautifulSoup(main_html, "html.parser").get_text(" ")
            main = " ".join(main.split())
            if _lang_ok(main):
                for seg in re.split(r"(?<=[.!?])\s+", main):
                    seg = seg.strip()
                    if 3 <= len(seg.split()) <= 22:
                        texts.append(seg)
        except Exception:
            pass
    # Keep short and English-ish
    texts = [x for x in texts if _lang_ok(x)]
    return texts[:400]

# ---------- Regex specs ----------
SPEC_PATTERNS = [
    ("weight", r"(?:^|\b)(?:item\s*)?weight[:\s-]*([\d\.]+)\s*(kg|g|lbs|lb|pounds|oz)\b"),
    ("dimensions", r"(?:^|\b)(?:dimensions|size|measurements)[:\s-]*([\d\.\sx×x]+)\s*(cm|mm|in|\"|')?"),
    ("capacity", r"(?:^|\b)(?:capacity|volume)[:\s-]*([\d\.]+)\s*(l|liters|ml|oz|mah|gb|tb)\b"),
    ("battery_life", r"(?:^|\b)(?:battery\s*(?:life|runtime|playtime))[:\s-]*([\d\.]+)\s*(h|hr|hrs|hours|minutes|min)\b"),
    ("power", r"(?:^|\b)(?:power|wattage|output)[:\s-]*([\d\.]+)\s*(w|kw|v|volts|amps|a)\b"),
    ("screen", r"(?:^|\b)(?:screen|display|resolution)[:\s-]*([\d]{3,4}\s?[x×]\s?[\d]{3,4}|[\d\.]+\s?(?:in|inch|\"))"),
    ("ip_rating", r"\bip\s?([0-9]{2})\b"),
    # Outdoor / hard-goods extras
    ("load_capacity", r"(?:max(?:imum)?\s+)?(?:load|weight)\s+capacit(?:y|ies)[:\s-]*([\d\.]+)\s*(lb|lbs|pounds|kg)"),
    ("packed_size", r"(?:packed\s*(?:size|dimensions))[:\s-]*([\d\.\sx×x]+)\s*(cm|mm|in|\"|')?"),
    ("seat_height", r"(?:seat\s*height)[:\s-]*([\d\.]+)\s*(cm|mm|in|\"|')"),
    ("materials", r"(?:materials?)[:\s-]*([a-z0-9,\s\-\/\+]+)"),
]

DROP_PATTERNS = [
    r"\b(terms of service|privacy policy|care(?:\s*&\s*| and )?repair|trade[-\s]?in|gift card)\b",
    r"\b(newsletter|subscribe|promotions?|inventory alerts|homepage|careers|press|retailers?)\b",
    r"\b(adventure anywhere|adventure for anyone|adventure forever|beginner to expert)\b",
    r"^where to next\??$",
    r"^purchase with purpose$",
]

PRODUCT_TOKENS = [
    "frame","seat","fabric","mesh","strap","pocket","zipper","leg","foot","arm",
    "hinge","case","bag","port","button","switch","sensor","battery","blade",
    "panel","lens","mount","rail","hood","pouch","cup","holder","stand","grip",
    "sole","midsole","outsole","drawcord","vent","nozzle","filters","filter"
]

def _looks_like_feature(line: str) -> bool:
    low = line.lower().strip("•-: ").strip()
    if "®" in low or "™" in low:
        return False
    if any(re.search(p, low) for p in DROP_PATTERNS):
        return False
    if not (3 <= len(low.split()) <= 12):
        return False
    if not any(t in low for t in PRODUCT_TOKENS):
        return False
    return True

def _extract_specs_from_lines(lines: List[str]) -> List[Claim]:
    claims: List[Claim] = []
    for ln in lines:
        low = ln.lower()
        for key, pat in SPEC_PATTERNS:
            m = re.search(pat, low)
            if m:
                val = " ".join([x for x in m.groups() if x]).strip()
                claims.append(Claim(key=key, value=val, source="", snippet=ln, kind="spec"))
    return claims

def _extract_features_from_lines(lines: List[str]) -> List[Claim]:
    out: List[Claim] = []
    for ln in lines:
        if _looks_like_feature(ln):
            out.append(Claim(key="feature", value=ln.strip("•-: ").strip(), source="", snippet=ln, kind="feature"))
    return out

def _extract_disclaimers(lines: List[str]) -> List[Claim]:
    out=[]
    for ln in lines:
        low=ln.lower()
        if any(k in low for k in ["do not", "caution", "warning", "not intended", "read instructions", "follow warnings","keep out of reach"]):
            if 3 <= len(ln.split()) <= 26:
                out.append(Claim(key="disclaimer", value=ln.strip(), source="", snippet=ln, kind="disclaimer"))
    return out

# -------- main entry --------
def research_product(
    brand: str,
    product: str,
    *,
    product_url_override: str = "",
    ocr_hints: List[str] | None = None,
    vision_hints: List[str] | None = None,
    max_results: int = 10,
    min_confidence: float = 0.6,
) -> Dict[str, Any]:
    bundle = {
        "query": "",
        "sources": [],
        "features": [],
        "specs": [],
        "disclaimers": [],
        "claims": [],
        "notes": [],
    }

    hints = " ".join((ocr_hints or [])[:5] + (vision_hints or [])[:5])
    exact = f"\"{brand} {product}\""
    queries = [
        f"{exact} specs {hints}".strip(),
        f"{exact} manual".strip(),
        f"{exact} datasheet".strip(),
        f"{brand} {product} specifications".strip(),
        f"{product} specifications".strip(),
        f"{exact} filetype:pdf".strip(),
    ]
    bundle["query"] = queries[0]

    # Candidate URLs
    urls: List[Dict[str, str]] = []
    if product_url_override:
        urls = [{"title": "Provided URL", "url": product_url_override}]
    else:
        urls = _ddg_search_multi(queries, max_results=max_results)

    # Fetch + score
    candidates = []
    for h in urls:
        html = get_html(h["url"]) or ""
        if not html:
            continue
        s = _score_page(h["url"], h.get("title",""), html, f"{brand} {product}")
        candidates.append((s, h, html))
    candidates.sort(reverse=True, key=lambda x: x[0])

    # Extract claims from HTML pages
    html_claims: List[Claim] = []
    seen_sources = []
    for score, meta, html in candidates[:max_results]:
        url = meta["url"]
        seen_sources.append({"title": meta.get("title") or url, "url": url})
        jsonlds = _jsonld_blocks(html)
        # JSON-LD Product additionalProperty
        for block in jsonlds:
            typ = block.get("@type") or block.get("type")
            if isinstance(typ, list): typ = ",".join(typ)
            if typ and "Product" not in str(typ):
                continue
            addp = block.get("additionalProperty")
            if isinstance(addp, list):
                for prop in addp:
                    name = (prop.get("name") or "").strip().lower()
                    val = prop.get("value")
                    if isinstance(val, dict):
                        val = val.get("value") or val.get("name") or ""
                    val = (val or "").strip()
                    if name and val:
                        html_claims.append(Claim(key=name, value=val, source=url, snippet=f"[JSON-LD] {name}: {val}", kind="spec"))
            desc = block.get("description")
            if isinstance(desc, str) and _looks_like_feature(desc):
                html_claims.append(Claim(key="feature", value=desc.strip(), source=url, snippet="[JSON-LD] description", kind="feature"))

        lines = _productish_blocks(html)
        # Add source to extracted lines
        for c in _extract_specs_from_lines(lines):
            c.source = url
            html_claims.append(c)
        for c in _extract_features_from_lines(lines):
            c.source = url
            html_claims.append(c)
        for c in _extract_disclaimers(lines):
            c.source = url
            html_claims.append(c)

    # PDF manuals / datasheets
    pdf_urls = discover_spec_pdfs(brand, product, max_results=4)
    for u in pdf_urls:
        try:
            pdf_specs, pdf_feats = extract_pdf_specs(u)
            for k, v, snip in pdf_specs:
                html_claims.append(Claim(key=k, value=v, source=u, snippet=snip, kind="spec"))
            for ft, snip in pdf_feats:
                if _looks_like_feature(ft):
                    html_claims.append(Claim(key="feature", value=ft, source=u, snippet=snip, kind="feature"))
            seen_sources.append({"title": "PDF", "url": u})
        except Exception:
            continue

    # Consolidate (consensus with unit normalization)
    consolidated = consolidate_claims(
        claims=html_claims,
        brand=brand,
        min_confidence=min_confidence,
        is_manufacturer=_is_manufacturer
    )

    # Vision/OCR floor: only if features < 4 after consensus
    if len(consolidated["features"]) < 4:
        neutral_fallbacks = []
        for hint in (vision_hints or []):
            if _looks_like_feature(hint):
                neutral_fallbacks.append({"text": hint, "sources": [], "confidence": 0.55})
        for hint in (ocr_hints or []):
            if _looks_like_feature(hint):
                neutral_fallbacks.append({"text": hint, "sources": [], "confidence": 0.55})
        # Fill up to 4
        needed = 4 - len(consolidated["features"])
        consolidated["features"].extend(neutral_fallbacks[:max(0, needed)])

    bundle.update({
        "sources": seen_sources,
        "features": consolidated["features"],
        "specs": consolidated["specs"],
        "disclaimers": consolidated["disclaimers"],
        "claims": consolidated["claims"],
        "notes": consolidated.get("notes", []),
    })
    return bundle
