# product_research.py
"""
Robust, product-agnostic research pipeline with:
- Multi-query resolver (site + manual + pdf/manual paths)
- Page rendering (Playwright if available) with scoring to prefer product pages
- Schema.org Product + DOM sections parsing
- PDF/manual spec extraction
- Anti-slogan firewall & noun-based feature filter
- Unit normalization (pint if available)
- Consensus aggregator (manufacturer first, else ≥2 sources)
- Vision/OCR hints as last-resort feature floor
- Provenance & confidence in outputs; also returns legacy fields for compatibility
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import re, json, math
from urllib.parse import urlparse

# Optional deps – all gracefully optional
try:
    from duckduckgo_search import DDGS
except Exception:
    DDGS = None

try:
    import requests
except Exception:
    requests = None

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

try:
    from readability import Document
except Exception:
    Document = None

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

# Local helper modules
try:
    from fetcher import get_html  # headless rendering if available
except Exception:
    get_html = None

from consensus import consolidate_claims, normalize_units

HEADERS = {"User-Agent": "briefgenv2/1.0 (+https://github.com/)"}

# -------------------- Claim dataclass --------------------
@dataclass
class Claim:
    key: str                 # "weight" | "dimensions" | "feature" | ...
    value: str               # "0.95 kg" | "mesh seat"
    source: str              # URL
    snippet: str             # nearby text
    kind: str                # "spec" | "feature" | "disclaimer"
    score: float = 0.0       # page-level score (helps tie-break)
    manufacturer: bool = False

# -------------------- Utils --------------------
def _dedupe(seq: List[str]) -> List[str]:
    seen=set(); out=[]
    for s in seq:
        s=(s or "").strip()
        k=s.lower()
        if s and k not in seen:
            seen.add(k); out.append(s)
    return out

def _short_enough(s: str, max_words: int = 22) -> bool:
    w = s.split()
    return 3 <= len(w) <= max_words

def _lang_ok(text: str) -> bool:
    return bool(re.search(r"\b(the|and|with|for|of|in|to|is|it|this|a|an)\b", (text or "").lower()))

def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

def link_density(html: str) -> float:
    if not BeautifulSoup:
        return 0.0
    soup = BeautifulSoup(html, "html.parser")
    text_len = len(" ".join(soup.get_text(" ").split()))
    link_text = " ".join([a.get_text(" ") for a in soup.find_all("a")])
    link_len = len(link_text)
    if text_len == 0:
        return 0.0
    return min(1.0, link_len / text_len)

# -------------------- Search --------------------
def ddg_search_multi(queries: List[str], max_results: int = 10) -> List[Dict[str, str]]:
    if not DDGS:
        return []
    hits=[]
    try:
        with DDGS() as ddgs:
            per = max(2, max_results // max(1, len(queries)))
            for q in queries:
                for r in ddgs.text(q, max_results=per, region="wt-wt", safesearch="moderate"):
                    url=r.get("href") or r.get("url")
                    title=r.get("title") or r.get("body") or url
                    if not url: 
                        continue
                    hits.append({"title": title, "url": url})
    except Exception:
        pass
    # Deduplicate by URL
    seen=set(); out=[]
    for h in hits:
        k=h["url"]
        if k not in seen:
            seen.add(k); out.append(h)
    return out[:max_results]

# -------------------- Fetch --------------------
def fetch_html(url: str, timeout: int = 15) -> Optional[str]:
    # prefer headless renderer
    if get_html:
        try:
            html = get_html(url, timeout=timeout)
            if html and len(html) > 500:
                return html
        except Exception:
            pass
    if not requests:
        return None
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        ct = r.headers.get("Content-Type","")
        if "text/html" not in ct and "application/xhtml" not in ct:
            return None
        return r.text
    except Exception:
        return None

# -------------------- Page scoring --------------------
PRODUCT_URL_HINTS = ("/product/", "/products/", "/p/", "/dp/", "/sku/", "/item/", "/buy/")
DROP_HTML_HINTS = r"(privacy|terms|subscribe|newsletter|sitewide|homepage|careers|press|login|account|tracking)"

def _score_page(url: str, title: str, html: str, product_tokens: List[str]) -> float:
    s = 0.0
    u = url.lower()
    t = (title or "").lower()
    h = (html or "").lower()

    # URL & title signals
    if any(x in u for x in PRODUCT_URL_HINTS): s += 2.0
    s += sum(0.4 for tok in product_tokens if tok and tok in t)

    # DOM signals
    if "schema.org/product" in h: s += 2.0
    if re.search(r"(add\s+to\s+cart|buy\s+now|sku|model\s*:)", h): s += 1.2
    if re.search(DROP_HTML_HINTS, h): s -= 1.5

    # Low link density → more product-like
    try:
        if link_density(html) < 0.18: s += 0.6
    except Exception:
        pass
    return s

# -------------------- HTML Parsing --------------------
def jsonld_blocks(html: str) -> List[dict]:
    if not BeautifulSoup:
        return []
    soup = BeautifulSoup(html, "html.parser")
    data=[]
    for tag in soup.find_all("script", {"type": "application/ld+json"}):
        try:
            txt = tag.string or tag.text or ""
            if not txt.strip(): continue
            obj = json.loads(txt)
            if isinstance(obj, dict): data.append(obj)
            elif isinstance(obj, list): data.extend([x for x in obj if isinstance(x, dict)])
        except Exception:
            continue
    return data

def texts_from_dom(html: str) -> List[str]:
    if not BeautifulSoup:
        return []
    soup = BeautifulSoup(html, "html.parser")

    # Remove obvious boilerplate containers
    for sel in ["header","footer","nav","aside",".site-header",".site-footer",".newsletter",".subscribe",".promo",".banner"]:
        for tag in soup.select(sel):
            tag.decompose()

    texts=[]
    # Section headings & lists/rows under them
    for sec in soup.find_all(["section","div","article"]):
        heading = sec.find(re.compile("^h[1-4]$"))
        if heading and not re.search(r"(spec|feature|detail|tech|specification|dimension|material|what.?s in the box)", heading.get_text(" "), re.I):
            continue
        # capture lists/rows
        for li in sec.select("li"):
            t=" ".join(li.get_text(" ").split())
            if _short_enough(t): texts.append(t)
        for tr in sec.select("table tr"):
            t=" ".join(tr.get_text(" ").split())
            if _short_enough(t): texts.append(t)
        for dd in sec.select("dl > *"):
            t=" ".join(dd.get_text(" ").split())
            if _short_enough(t): texts.append(t)

    # Short paragraphs & headings anywhere
    for p in soup.select("p"):
        t=" ".join(p.get_text(" ").split())
        if _short_enough(t, 28): texts.append(t)
    for h in soup.select("h1,h2,h3,h4"):
        t=" ".join(h.get_text(" ").split())
        if 2 <= len(t.split()) <= 14: texts.append(t)

    # Readability main text
    if Document:
        try:
            doc = Document(html)
            main = BeautifulSoup(doc.summary(), "html.parser").get_text(" ")
            main = " ".join(main.split())
            if _lang_ok(main):
                for seg in re.split(r"(?<=[.!?])\s+", main):
                    seg=seg.strip()
                    if _short_enough(seg, 22): texts.append(seg)
        except Exception:
            pass
    return texts[:400]

# -------------------- Pattern extraction --------------------
SPEC_PATTERNS = [
    ("weight", r"(?:^|\b)(?:item\s*)?weight[:\s-]*([\d\.]+)\s*(kg|g|lbs|lb|pounds|oz)\b"),
    ("dimensions", r"(?:^|\b)(?:dimensions|size)[:\s-]*([\d\.\sx×x]+)\s*(cm|mm|in|\"|')?"),
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
    r"\b(terms of service|privacy policy|warranty|care and repair|gear trade in)\b",
    r"\b(newsletter|subscribe|promotions|inventory alerts|join|homepage|careers|press|shipping|returns|sale|promo)\b",
    r"\b(adventure anywhere|adventure for anyone|adventure forever|beginner to expert)\b",
    r"^where to next\??$",
    r"^purchase with purpose$",
]
PRODUCT_TOKENS = [
    "frame","seat","fabric","mesh","strap","pocket","zipper","leg","foot","arm",
    "hinge","case","bag","port","button","switch","sensor","battery","blade",
    "panel","lens","mount","rail","hood","pouch","cup","holder","stand","grip",
    "sole","midsole","outsole","drawcord","vent","nozzle","filter","filters","buckle",
]

def looks_like_feature(line: str) -> bool:
    low = (line or "").lower().strip("•-: ").strip()
    if "®" in low or "™" in low:
        return False
    if any(re.search(p, low) for p in DROP_PATTERNS):
        return False
    if not (3 <= len(low.split()) <= 10):
        return False
    if not any(t in low for t in PRODUCT_TOKENS):
        return False
    return _lang_ok(low)

def extract_specs_from_lines(lines: List[str]) -> Dict[str,str]:
    specs={}
    for ln in lines:
        low=ln.lower()
        for key, pat in SPEC_PATTERNS:
            m=re.search(pat, low)
            if m and key not in specs:
                specs[key]=" ".join([x for x in m.groups() if x]).strip()
    return specs

def extract_features_from_lines(lines: List[str]) -> List[str]:
    feats=[]
    for ln in lines:
        if looks_like_feature(ln):
            feats.append(ln.strip("•-: ").strip())
    return _dedupe(feats)

def extract_disclaimers(lines: List[str]) -> List[str]:
    out=[]
    for ln in lines:
        low=ln.lower()
        if any(k in low for k in ["do not", "caution", "warning", "not intended", "read instructions", "follow warnings","keep out of reach"]):
            if _short_enough(ln, 26): out.append(ln.strip())
    return _dedupe(out)

# -------------------- PDF/manual parser --------------------
def pdf_text_from_url(url: str, timeout: int = 20) -> List[str]:
    if not (requests and fitz):
        return []
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        if "application/pdf" not in r.headers.get("Content-Type","").lower():
            return []
        with fitz.open(stream=r.content, filetype="pdf") as doc:
            lines=[]
            for page in doc:
                text = page.get_text("text")
                if text:
                    for ln in text.splitlines():
                        ln = " ".join(ln.split())
                        if ln: lines.append(ln)
            return lines[:2000]
    except Exception:
        return []

def extract_specs_from_pdf_lines(lines: List[str]) -> Dict[str, str]:
    if not lines:
        return {}
    specs={}
    blob = " ".join(lines)
    for key, pat in SPEC_PATTERNS:
        m = re.search(pat, blob.lower())
        if m and key not in specs:
            specs[key]=" ".join([x for x in m.groups() if x]).strip()
    # Try table-like lines for k: v pairs
    for ln in lines:
        if ":" in ln and len(ln) < 120:
            k,v = ln.split(":",1)
            k=k.strip().lower(); v=v.strip()
            if any(t in k for t in ["weight","dimensions","capacity","battery","power","screen","ip","load","packed","seat","material"]):
                if k not in specs and len(v) <= 40:
                    specs[k]=v
    return specs

# -------------------- Main research --------------------
def research_product(
    brand: str,
    product: str,
    *,
    product_url_override: str = "",
    ocr_hints: List[str] | None = None,
    vision_hints: List[str] | None = None,
    max_results: int = 10,
) -> Dict[str, Any]:

    ocr_hints = ocr_hints or []
    vision_hints = vision_hints or []

    bundle = {
        "query": "",
        "sources": [],
        "features": [],
        "specs": {},
        "disclaimers": [],
        "claims": [],
        "features_detailed": [],
        "specs_detailed": [],
    }

    # Queries
    hints=" ".join((ocr_hints[:6] + vision_hints[:6]))
    q0=f'"{brand} {product}" specs features {hints}'.strip()
    q1=f'"{product}" manual OR datasheet'.strip()
    q2=f'site:{brand.lower().split()[0]}.com "{product}" specifications'.strip() if brand else f'"{product}" specifications'
    q3=f'"{brand} {product}" filetype:pdf'.strip()
    bundle["query"]=q0

    # Candidate URLs
    urls=[]
    if product_url_override:
        urls=[{"title":"Provided URL","url":product_url_override}]
    else:
        urls=ddg_search_multi([q0,q2], max_results=max_results)

    # Fetch + score HTML pages
    product_tokens=[t for t in f"{brand} {product}".lower().split() if len(t)>=3]
    candidates=[]
    for h in urls:
        html=fetch_html(h["url"])
        if not html:
            continue
        s = _score_page(h["url"], h.get("title",""), html, product_tokens)
        candidates.append((s, h, html))

    # Add PDF/manual hits
    pdf_hits = ddg_search_multi([q1,q3], max_results=6)
    pdf_claims: List[Claim] = []
    for hit in pdf_hits:
        u = hit.get("url","")
        if not u.lower().endswith(".pdf"):
            continue
        lines = pdf_text_from_url(u)
        if not lines:
            continue
        specs_pdf = extract_specs_from_pdf_lines(lines)
        specs_pdf = normalize_units(specs_pdf)  # optional unit normalization
        for k,v in specs_pdf.items():
            pdf_claims.append(Claim(key=k, value=v, source=u, snippet="", kind="spec", score=3.0, manufacturer=False))

    # Rank HTML candidates
    candidates.sort(reverse=True, key=lambda x: x[0])

    # Parse HTML → raw claims
    raw_claims: List[Claim] = []
    src_list: List[Dict[str,str]] = []
    for s, h, html in candidates[:max_results]:
        url = h["url"]
        src_list.append({"title": h.get("title") or url, "url": url})
        jsonlds = jsonld_blocks(html)
        # JSON-LD Product
        for block in jsonlds:
            typ = block.get("@type") or block.get("type")
            if isinstance(typ, list): typ=",".join(typ)
            if typ and "Product" not in str(typ): 
                continue
            addp = block.get("additionalProperty")
            if isinstance(addp, list):
                for prop in addp:
                    name=(prop.get("name") or "").strip().lower()
                    val=prop.get("value")
                    if isinstance(val, dict): val=val.get("value") or val.get("name") or ""
                    val=(val or "").strip()
                    if name and val:
                        raw_claims.append(Claim(key=name, value=val, source=url, snippet="jsonld", kind="spec", score=s))
        # DOM texts
        lines = texts_from_dom(html)
        lines=[ln for ln in lines if _lang_ok(ln)]
        # Specs
        specs = extract_specs_from_lines(lines)
        specs = normalize_units(specs)
        for k,v in specs.items():
            raw_claims.append(Claim(key=k, value=v, source=url, snippet="", kind="spec", score=s))
        # Features
        feats = extract_features_from_lines(lines)
        for f in feats:
            raw_claims.append(Claim(key="feature", value=f, source=url, snippet="", kind="feature", score=s))
        # Disclaimers
        dis = extract_disclaimers(lines)
        for d in dis:
            raw_claims.append(Claim(key="disclaimer", value=d, source=url, snippet="", kind="disclaimer", score=s))

    # Mark manufacturer pages (simple heuristic: domain contains brand token)
    brand_token = (brand or "").split()[0].lower() if brand else ""
    for c in raw_claims:
        dom = domain_of(c.source)
        c.manufacturer = bool(brand_token and brand_token in dom)

    # Merge PDF claims
    raw_claims.extend(pdf_claims)

    # Vision/OCR fallback → non-numeric feature hints if still thin later
    fallback_feats = _dedupe(vision_hints + ocr_hints)

    # Consensus aggregation
    consensus = consolidate_claims(raw_claims, brand_token=brand_token)

    # Build outputs
    features_detailed = consensus.get("features", [])
    specs_detailed = consensus.get("specs", [])
    disclaimers = _dedupe([c.value for c in raw_claims if c.kind == "disclaimer"])[:10]

    # If features still thin, top up with vision/ocr hints (neutral)
    if len(features_detailed) < 4:
        for f in fallback_feats:
            features_detailed.append({"text": f, "sources": [], "confidence": 0.55})
            if len(features_detailed) >= 6:
                break

    # Legacy/simple fields for backward compatibility
    simple_features = [f["text"] for f in features_detailed][:20]
    simple_specs: Dict[str,str] = {}
    for s in specs_detailed:
        if s.get("key") and s.get("value") and s["key"] not in simple_specs:
            simple_specs[s["key"]] = s["value"]

    bundle.update({
        "sources": src_list,
        "features_detailed": features_detailed,
        "specs_detailed": specs_detailed,
        "features": simple_features,
        "specs": simple_specs,
        "disclaimers": disclaimers,
        "claims": [],  # not used for scripting (avoid slogans)
    })
    return bundle
