# product_research.py — v4.1
from __future__ import annotations
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re, json
from urllib.parse import urlparse

# Optional deps
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
# readability is optional
try:
    from readability import Document
except Exception:
    Document = None
# PDF parsers (optional)
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None
try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
except Exception:
    pdfminer_extract_text = None

# optional headless fetcher
try:
    from fetcher import get_html
except Exception:
    get_html = None

from consensus import consolidate_claims, normalize_units

HEADERS = {"User-Agent": "briefgenv2/1.0 (+https://github.com/)"}

# ---------------- Data ----------------
RETAIL_HOST_HINTS = [
    "amazon.", "bestbuy.", "bhphotovideo.", "walmart.", "target.", "costco.", "rei.",
    "backcountry.", "homedepot.", "lowes.", "microcenter.", "newegg.", "apple.", "samsung.",
    "sony.", "bose.", "sennheiser.", "jbl.", "anker.", "beats.", "logitech.", "lenovo.", "dell.",
]

DENY_HOST_HINTS = [
    "microsoft.com/answers", "support.microsoft.com", "learn.microsoft.com",
    "stackoverflow.", "stackexchange.", "github.", "reddit.", "docs.", "w3.",
]

PRODUCT_URL_HINTS = ("/product/", "/products/", "/p/", "/dp/", "/sku/", "/item/", "/buy/", "/shop/")
DROP_HTML_HINTS = r"(privacy|terms|subscribe|newsletter|sitewide|homepage|careers|press|login|account|tracking)"

SPEC_PATTERNS = [
    ("weight", r"(?:^|\b)(?:item\s*)?weight[:\s-]*([\d\.]+)\s*(kg|g|lbs|lb|pounds|oz)\b"),
    ("dimensions", r"(?:^|\b)(?:dimensions|size)[:\s-]*([\d\.\sx×x]+)\s*(cm|mm|in|\"|')?"),
    ("capacity", r"(?:^|\b)(?:capacity|volume)[:\s-]*([\d\.]+)\s*(l|liters|ml|oz|mah|gb|tb)\b"),
    ("battery_life", r"(?:^|\b)(?:battery\s*(?:life|runtime|playtime))[:\s-]*([\d\.]+)\s*(h|hr|hrs|hours|minutes|min)\b"),
    ("power", r"(?:^|\b)(?:power|wattage|output)[:\s-]*([\d\.]+)\s*(w|kw|v|volts|amps|a)\b"),
    ("screen", r"(?:^|\b)(?:screen|display|resolution)[:\s-]*([\d]{3,4}\s?[x×]\s?[\d]{3,4}|[\d\.]+\s?(?:in|inch|\"))"),
    ("ip_rating", r"\bip\s?([0-9]{2})\b"),
    # extra
    ("load_capacity", r"(?:max(?:imum)?\s+)?(?:load|weight)\s+capacit(?:y|ies)[:\s-]*([\d\.]+)\s*(lb|lbs|pounds|kg)"),
    ("packed_size", r"(?:packed\s*(?:size|dimensions))[:\s-]*([\d\.\sx×x]+)\s*(cm|mm|in|\"|')?"),
    ("seat_height", r"(?:seat\s*height)[:\s-]*([\d\.]+)\s*(cm|mm|in|\"|')"),
    ("materials", r"(?:materials?)[:\s-]*([a-z0-9,\s\-\/\+]+)"),
]

DROP_PATTERNS = [
    r"\b(terms of service|privacy policy|warranty|gear trade in)\b",
    r"\b(newsletter|subscribe|promotions|inventory alerts|join|homepage|careers|press|shipping|returns|sale|promo)\b",
    r"\b(adventure anywhere|adventure for anyone|adventure forever)\b",
]

PRODUCT_TOKENS = [
    "frame","seat","mesh","strap","pocket","zipper","leg","arm","hinge","bag",
    "button","switch","sensor","battery","driver","earcup","headband","cushion","microphone","anc","bluetooth","usb-c",
]

CATEGORY_SYNONYMS = {
    "headphones": {"headphone","headphones","earbuds","earphones","audio","anc","bt","bluetooth"},
    "chair": {"chair","camp chair","folding chair","lawn chair","double chair","recliner"},
    "camera": {"camera","dslr","mirrorless","action cam"},
    "phone": {"phone","smartphone","android","iphone"},
}

# ---------------- Models ----------------
@dataclass
class Claim:
    key: str
    value: str
    source: str
    snippet: str
    kind: str            # "spec" | "feature" | "disclaimer"
    score: float = 0.0
    manufacturer: bool = False

# ---------------- Utils ----------------
def _dedupe(seq: List[str]) -> List[str]:
    seen=set(); out=[]
    for s in seq:
        s=(s or "").strip(); k=s.lower()
        if s and k not in seen:
            seen.add(k); out.append(s)
    return out

def domain_of(url: str) -> str:
    try: return urlparse(url).netloc.lower()
    except Exception: return ""

def _short_enough(s: str, max_words: int = 22) -> bool:
    w = s.split(); return 3 <= len(w) <= max_words

def _lang_ok(text: str) -> bool:
    return bool(re.search(r"\b(the|and|with|for|of|in|to|is|it|this|a|an)\b", (text or "").lower()))

def link_density(html: str) -> float:
    if not BeautifulSoup: return 0.0
    soup = BeautifulSoup(html, "html.parser")
    text_len = len(" ".join(soup.get_text(" ").split()))
    link_text = " ".join([a.get_text(" ") for a in soup.find_all("a")])
    if text_len == 0: return 0.0
    return min(1.0, len(link_text)/text_len)

def category_from_text(text: str) -> set:
    low = (text or "").lower()
    tags=set()
    for cat, syns in CATEGORY_SYNONYMS.items():
        if any(s in low for s in syns): tags.add(cat)
    return tags

def categories_intersect(a: set, b: set) -> bool:
    return bool(a and b and (a & b))

# ---------------- Search ----------------
def ddg_search_multi(queries: List[str], max_results: int = 10) -> List[Dict[str,str]]:
    if not DDGS: return []
    hits=[]
    try:
        with DDGS() as ddgs:
            per = max(2, max_results//max(1,len(queries)))
            for q in queries:
                for r in ddgs.text(q, max_results=per, region="wt-wt", safesearch="moderate"):
                    url=r.get("href") or r.get("url"); title=r.get("title") or r.get("body") or url
                    if url: hits.append({"title":title,"url":url})
    except Exception:
        pass
    # dedupe
    seen=set(); out=[]
    for h in hits:
        if h["url"] not in seen:
            seen.add(h["url"]); out.append(h)
    return out[:max_results]

# ---------------- Fetch ----------------
def fetch_html(url: str, timeout: int = 15) -> Optional[str]:
    if get_html:
        try:
            html = get_html(url, timeout=timeout)
            if html and len(html)>500: return html
        except Exception: pass
    if not requests: return None
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        if "text/html" not in r.headers.get("Content-Type",""):
            return None
        return r.text
    except Exception:
        return None

# ---------------- Parse ----------------
def jsonld_blocks(html: str) -> List[dict]:
    if not BeautifulSoup: return []
    soup = BeautifulSoup(html, "html.parser")
    data=[]
    for tag in soup.find_all("script", {"type":"application/ld+json"}):
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
    if not BeautifulSoup: return []
    soup = BeautifulSoup(html, "html.parser")

    for sel in ["header","footer","nav","aside",".site-header",".site-footer",".newsletter",".subscribe",".promo",".banner"]:
        for tag in soup.select(sel):
            try: tag.decompose()
            except Exception: pass

    texts=[]
    # sections likely to contain specs/features
    for sec in soup.find_all(["section","div","article"]):
        heading = sec.find(re.compile("^h[1-4]$"))
        if heading and not re.search(r"(spec|feature|detail|tech|specification|dimension|material|what.?s in the box)", heading.get_text(" "), re.I):
            continue
        for li in sec.select("li"):
            t=" ".join(li.get_text(" ").split())
            if _short_enough(t): texts.append(t)
        for tr in sec.select("table tr"):
            t=" ".join(tr.get_text(" ").split())
            if _short_enough(t): texts.append(t)
        for dd in sec.select("dl > *"):
            t=" ".join(dd.get_text(" ").split())
            if _short_enough(t): texts.append(t)
    for p in soup.select("p"):
        t=" ".join(p.get_text(" ").split())
        if _short_enough(t,28): texts.append(t)
    for h in soup.select("h1,h2,h3,h4"):
        t=" ".join(h.get_text(" ").split())
        if 2 <= len(t.split()) <= 14: texts.append(t)

    if Document:
        try:
            doc = Document(html)
            main = BeautifulSoup(doc.summary(), "html.parser").get_text(" ")
            main = " ".join(main.split())
            if _lang_ok(main):
                for seg in re.split(r"(?<=[.!?])\s+", main):
                    if _short_enough(seg,22): texts.append(seg.strip())
        except Exception:
            pass
    return texts[:400]

def extract_specs_from_lines(lines: List[str]) -> Dict[str,str]:
    out={}
    for ln in lines:
        low=ln.lower()
        for key,pat in SPEC_PATTERNS:
            m=re.search(pat, low)
            if m and key not in out:
                out[key]=" ".join([x for x in m.groups() if x]).strip()
    return out

def looks_like_feature(line: str) -> bool:
    low=(line or "").lower().strip("•-: ").strip()
    if "®" in low or "™" in low: return False
    if any(re.search(p, low) for p in DROP_PATTERNS): return False
    if not (3 <= len(low.split()) <= 10): return False
    if not any(t in low for t in PRODUCT_TOKENS): return False
    return _lang_ok(low)

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
        if any(k in low for k in ["do not","caution","warning","not intended","read instructions","keep out of reach"]):
            if _short_enough(ln,26): out.append(ln.strip())
    return _dedupe(out)

# ---------------- PDF ----------------
def pdf_text_from_url(url: str, timeout: int = 20) -> List[str]:
    if not requests: return []
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        if "application/pdf" not in r.headers.get("Content-Type","").lower(): return []
        content = r.content
    except Exception:
        return []
    if fitz:
        try:
            lines=[]
            with fitz.open(stream=content, filetype="pdf") as doc:
                for page in doc:
                    text = page.get_text("text") or ""
                    for ln in text.splitlines():
                        ln=" ".join(ln.split())
                        if ln: lines.append(ln)
            return lines[:2000]
        except Exception:
            pass
    if pdfminer_extract_text:
        try:
            from io import BytesIO
            text = pdfminer_extract_text(BytesIO(content)) or ""
            lines=[]
            for ln in text.splitlines():
                ln=" ".join(ln.split())
                if ln: lines.append(ln)
            return lines[:2000]
        except Exception:
            pass
    return []

def extract_specs_from_pdf_lines(lines: List[str]) -> Dict[str,str]:
    if not lines: return {}
    specs={}
    blob=" ".join(lines)
    for key,pat in SPEC_PATTERNS:
        m=re.search(pat, blob.lower())
        if m and key not in specs:
            specs[key]=" ".join([x for x in m.groups() if x]).strip()
    for ln in lines:
        if ":" in ln and len(ln) < 120:
            k,v = ln.split(":",1)
            k=k.strip().lower(); v=v.strip()
            if any(t in k for t in ["weight","dimensions","capacity","battery","power","screen","ip","load","packed","seat","material"]):
                if k not in specs and len(v) <= 40:
                    specs[k]=v
    return specs

# ---------------- Core ----------------
def _is_allowed_host(host: str) -> bool:
    if any(h in host for h in DENY_HOST_HINTS): return False
    return True

def _is_productish(url: str, title: str, html: str, product_tokens: List[str]) -> bool:
    u=url.lower(); t=(title or "").lower(); h=(html or "").lower()
    # must look like a product page
    url_hit = any(x in u for x in PRODUCT_URL_HINTS)
    schema_hit = "schema.org/product" in h
    buy_hit = re.search(r"(add\s+to\s+cart|buy\s+now|sku|model\s*:|price)", h) is not None
    token_hit = sum(1 for tok in product_tokens if tok and (tok in t or tok in h)) >= max(1, min(2, len(product_tokens)))
    return (schema_hit or buy_hit or url_hit) and token_hit

def research_product(
    brand: str, product: str, *,
    product_url_override: str = "",
    vision_category_tags: List[str] | None = None,
    max_results: int = 10,
) -> Dict[str, Any]:

    warnings=[]
    typed_cats = category_from_text(f"{brand} {product}")
    vision_cats = set(x.lower() for x in (vision_category_tags or []))

    # base queries
    q0=f'"{brand} {product}" specs features'
    q1=f'"{product}" manual OR datasheet'
    q2=f'site:{brand.lower().split()[0]}.com "{product}" specifications' if brand else f'"{product}" specifications'
    q3=f'"{brand} {product}" filetype:pdf'
    if not brand or not product:
        warnings.append("Missing brand/product name — research is likely to be weak.")

    urls=[]
    if product_url_override:
        urls=[{"title":"Provided URL","url":product_url_override}]
    else:
        urls = ddg_search_multi([q0,q2], max_results=max_results)

    # fetch+filter
    product_tokens = [t for t in f"{brand} {product}".replace("-", " ").split() if len(t)>=3]
    candidates=[]
    for h in urls:
        url = h["url"]; host = domain_of(url)
        if not _is_allowed_host(host): continue
        html = fetch_html(url)
        if not html: continue
        if not _is_productish(url, h.get("title",""), html, product_tokens):
            # allow if manufacturer domain
            manu = bool(brand and brand.split()[0].lower() in host)
            if not manu: 
                continue
        # prefer retailer/manufacturer
        bias = 2.0 if (any(x in host for x in RETAIL_HOST_HINTS) or (brand and brand.split()[0].lower() in host)) else 0.0
        score = 2.0 + bias - (link_density(html) or 0.0)
        candidates.append((score, h, html))

    candidates.sort(reverse=True, key=lambda x: x[0])

    # PDFs
    pdf_hits = ddg_search_multi([q1,q3], max_results=6)
    pdf_claims=[]
    for hit in pdf_hits:
        u = hit.get("url","")
        if not u.lower().endswith(".pdf"): continue
        lines = pdf_text_from_url(u)
        if not lines: continue
        sp = normalize_units(extract_specs_from_pdf_lines(lines))
        for k,v in sp.items():
            pdf_claims.append(Claim(key=k, value=v, source=u, snippet="", kind="spec", score=3.0, manufacturer=False))

    # parse HTML -> claims
    raw_claims=[]; used_sources=set()
    for s, h, html in candidates[:max_results]:
        url=h["url"]; host=domain_of(url)
        used_sources.add(url)
        # JSON-LD
        soup_blocks = jsonld_blocks(html)
        for block in soup_blocks:
            typ = block.get("@type") or block.get("type")
            if isinstance(typ, list): typ=",".join(typ)
            if typ and "Product" not in str(typ): continue
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
        specs = normalize_units(extract_specs_from_lines(lines))
        for k,v in specs.items():
            raw_claims.append(Claim(key=k, value=v, source=url, snippet="", kind="spec", score=s))
        feats = extract_features_from_lines(lines)
        for f in feats:
            raw_claims.append(Claim(key="feature", value=f, source=url, snippet="", kind="feature", score=s))
        dis = extract_disclaimers(lines)
        for d in dis:
            raw_claims.append(Claim(key="disclaimer", value=d, source=url, snippet="", kind="disclaimer", score=s))

    brand_token = (brand or "").split()[0].lower() if brand else ""
    for c in raw_claims:
        c.manufacturer = bool(brand_token and brand_token in domain_of(c.source))

    raw_claims.extend(pdf_claims)

    # CONSENSUS
    consensus = consolidate_claims(raw_claims, brand_token=brand_token)
    specs_d = [s for s in (consensus.get("specs") or []) if len(s.get("sources",[]))>0]
    feats_d = [f for f in (consensus.get("features") or []) if len(f.get("sources",[]))>0]

    # Visual hints (category-gated and never "verified")
    visual_hints = []
    if vision_cats := set(vision_category_tags or []):
        # if typed category exists and doesn't match visuals, we refuse to mix
        if typed_cats and not categories_intersect(typed_cats, vision_cats):
            warnings.append("Reference video category does not match product category; only using visual hints for storyboard (not claims).")
        else:
            # ok to include hints as separate, non-verified info
            pass

    # Only list sources that actually contributed to accepted claims
    used_urls = set([u["url"] for s in specs_d for u in s.get("sources",[])]+
                    [u["url"] for f in feats_d for u in f.get("sources",[])])
    source_cards=[]
    for u in used_urls:
        source_cards.append({"title": u, "url": u})

    return {
        "query": q0,
        "sources": source_cards,              # ONLY used sources
        "features": [f["text"] for f in feats_d],
        "specs": {s["key"]: s["value"] for s in specs_d},
        "features_detailed": feats_d,
        "specs_detailed": specs_d,
        "disclaimers": _dedupe([c.value for c in raw_claims if c.kind=="disclaimer"])[:10],
        "visual_hints": visual_hints,         # ALWAYS non-verified
        "warnings": warnings,
    }
