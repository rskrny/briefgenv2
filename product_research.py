# product_research.py
"""
Product-agnostic research pipeline (robust).

Search any domain, parse pages generically, and guarantee at least 4 safe feature bullets.
Priority:
  1) Direct URL override (if provided)
  2) Web search (multiple queries)
  3) Parse: JSON-LD Product, tables/lists/headings, readability main text
  4) Regex/spec extraction + conservative feature bullets
  5) LLM structuring (optional if google-generativeai key present)
  6) Vision/OCR hints as final floor (still product-agnostic)

Returns normalized bundle:
{
  "query": "...",
  "sources": [{"title","url"}],
  "features": [...],            # short bullets (≤10 words)
  "specs": {"weight":"", "dimensions":"", ...},  # may be partial
  "disclaimers": [...],
  "claims": [...]               # extra short sentences (optional)
}
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import re, json

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

try:
    from llm import gemini_json  # optional LLM structuring
except Exception:
    gemini_json = None

HEADERS = {"User-Agent": "briefgenv2/1.0 (+https://github.com/)"}

# ------------------------ Utils ------------------------
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

def _looks_safe(line: str) -> bool:
    low=line.lower()
    risky=[r"\b(best|#1|guarantee|miracle|cure|world[-\s]?class)\b", r"\b(FDA|CE|UL|ISO|EPA)\b"]
    if any(re.search(p, low) for p in risky): return False
    if " better than " in low or " vs " in low or " versus " in low: return False
    return True

def _lang_ok(text: str) -> bool:
    # crude english detection: presence of common words
    return bool(re.search(r"\b(the|and|with|for|of|in|to)\b", text.lower()))

# -------------------- Search & Fetch --------------------
def _ddg_search_multi(queries: List[str], max_results: int = 10) -> List[Dict[str,str]]:
    if not DDGS:
        return []
    hits=[]
    try:
        with DDGS() as ddgs:
            for q in queries:
                for r in ddgs.text(q, max_results=max_results//len(queries) or 3):
                    url=r.get("href") or r.get("url"); title=r.get("title") or r.get("body") or url
                    if not url: continue
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

# -------------------- HTML parsing --------------------
def _jsonld_blocks(html: str) -> List[dict]:
    if not BeautifulSoup: return []
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

def _texts_from_dom(html: str) -> List[str]:
    if not BeautifulSoup: return []
    soup = BeautifulSoup(html, "html.parser")
    texts=[]
    for li in soup.select("li"):
        t=" ".join(li.get_text(" ").split())
        if _short_enough(t): texts.append(t)
    for tr in soup.select("table tr"):
        t=" ".join(tr.get_text(" ").split())
        if _short_enough(t): texts.append(t)
    for dd in soup.select("dl > *"):
        t=" ".join(dd.get_text(" ").split())
        if _short_enough(t): texts.append(t)
    for p in soup.select("p"):
        t=" ".join(p.get_text(" ").split())
        if _short_enough(t, 28): texts.append(t)
    for h in soup.select("h1,h2,h3,h4"):
        t=" ".join(h.get_text(" ").split())
        if 2 <= len(t.split()) <= 14: texts.append(t)
    # readability main text (if available)
    if Document:
        try:
            doc = Document(html)
            main = BeautifulSoup(doc.summary(), "html.parser").get_text(" ")
            main = " ".join(main.split())
            if _lang_ok(main):
                # split into short sentences
                for seg in re.split(r"(?<=[.!?])\s+", main):
                    seg=seg.strip()
                    if _short_enough(seg, 22): texts.append(seg)
        except Exception:
            pass
    return texts[:300]

# ---------- JSON-LD → specs/features ----------
def _from_jsonld(jsonlds: List[dict]) -> Tuple[List[str], Dict[str,str]]:
    features, specs = [], {}
    for block in jsonlds:
        typ = block.get("@type") or block.get("type")
        if isinstance(typ, list): typ=",".join(typ)
        if typ and "Product" not in str(typ): continue
        d = block.get("description") or ""
        if isinstance(d, str) and _short_enough(d, 40):
            features.append(d.strip())
        for key in ("featureList","features","feature"):
            fl=block.get(key)
            if isinstance(fl, list):
                for x in fl:
                    if isinstance(x, str) and _short_enough(x, 16):
                        features.append(x.strip())
        addp = block.get("additionalProperty")
        if isinstance(addp, list):
            for prop in addp:
                name=(prop.get("name") or "").strip()
                val=prop.get("value")
                if isinstance(val, dict): val=val.get("value") or val.get("name") or ""
                val=(val or "").strip()
                if name and val: specs[name.lower()] = val
    return _dedupe(features), specs

# ---------- Regex specs from text ----------
SPEC_PATTERNS = [
    ("weight", r"(?:^|\b)(?:item\s*)?weight[:\s-]*([\d\.]+)\s*(kg|g|lbs|lb|pounds|oz)\b"),
    ("dimensions", r"(?:^|\b)(?:dimensions|size)[:\s-]*([\d\.\sx×x]+)\s*(cm|mm|in|\"|')?"),
    ("capacity", r"(?:^|\b)(?:capacity|volume)[:\s-]*([\d\.]+)\s*(l|liters|ml|oz|mah|gb|tb)\b"),
    ("battery_life", r"(?:^|\b)(?:battery\s*(?:life|runtime|playtime))[:\s-]*([\d\.]+)\s*(h|hr|hrs|hours|minutes|min)\b"),
    ("power", r"(?:^|\b)(?:power|wattage|output)[:\s-]*([\d\.]+)\s*(w|kw|v|volts|amps|a)\b"),
    ("screen", r"(?:^|\b)(?:screen|display|resolution)[:\s-]*([\d]{3,4}\s?[x×]\s?[\d]{3,4}|[\d\.]+\s?(?:in|inch|\"))"),
    ("ip_rating", r"\bip\s?([0-9]{2})\b"),
]

def _extract_specs_from_lines(lines: List[str]) -> Dict[str,str]:
    specs={}
    for ln in lines:
        low=ln.lower()
        for key, pat in SPEC_PATTERNS:
            m=re.search(pat, low)
            if m and key not in specs:
                specs[key]=" ".join([x for x in m.groups() if x]).strip()
    return specs

def _extract_features_from_lines(lines: List[str]) -> List[str]:
    feats=[]
    for ln in lines:
        if not _looks_safe(ln): continue
        # prefer bullets / short phrases
        if _short_enough(ln, 12):
            feats.append(ln.strip("•-: ").strip())
    return _dedupe(feats)

def _extract_disclaimers(lines: List[str]) -> List[str]:
    out=[]
    for ln in lines:
        low=ln.lower()
        if any(k in low for k in ["do not", "caution", "warning", "not intended", "read instructions", "follow warnings","keep out of reach"]):
            if _short_enough(ln, 26): out.append(ln.strip())
    return _dedupe(out)

# ---------- LLM structuring (optional) ----------
def _llm_struct(texts: List[str]) -> Dict[str, Any]:
    if not gemini_json:
        return {}
    joined=" ".join(texts)[:8000]
    prompt=f"""
You extract product information from noisy web copy.
Return JSON ONLY:
{{
  "features": ["≤12 short bullets, product-agnostic, no superlatives"],
  "specs": {{"weight":"","dimensions":"","capacity":"","battery_life":"","power":"","screen":"","ip_rating":""}},
  "disclaimers": ["0–6 short safety/usage disclaimers"]
}}
Constraints:
- Keep features factual & non-comparative (no "#1", "best").
- Phrases ≤10 words each.
TEXT:
{joined}
"""
    out=gemini_json(prompt)
    if not isinstance(out, dict): return {}
    out.setdefault("features", []); out.setdefault("specs", {}); out.setdefault("disclaimers", [])
    if not isinstance(out["specs"], dict): out["specs"]={}
    out["features"]=[s for s in out["features"] if isinstance(s,str) and s.strip()]
    out["disclaimers"]=[s for s in out["disclaimers"] if isinstance(s,str) and s.strip()]
    return out

# -------------------- Public API --------------------
def research_product(
    brand: str,
    product: str,
    *,
    product_url_override: str = "",
    ocr_hints: List[str] | None = None,
    vision_hints: List[str] | None = None,
    max_results: int = 10,
) -> Dict[str, Any]:
    """
    Returns normalized bundle usable everywhere in the app.
    """
    bundle={"query":"", "sources":[], "features":[], "specs":{}, "disclaimers":[], "claims":[]}

    # Queries (broad + exact)
    hints=" ".join((ocr_hints or [])[:6] + (vision_hints or [])[:6])
    q0=f'"{brand} {product}" specs features {hints}'.strip()
    q1=f'"{product}" specs features {hints}'.strip()
    q2=f'{brand} {product} details buy'.strip()
    bundle["query"]=q0

    # Candidate URLs
    urls=[]
    if product_url_override:
        urls=[{"title":"Provided URL","url":product_url_override}]
    else:
        urls=_ddg_search_multi([q0,q1,q2], max_results=max_results)

    # Parse pages
    all_lines=[]; jsonld_features=[]; jsonld_specs={}
    for h in urls:
        html=_fetch_html(h["url"])
        if not html: continue
        jsonlds=_jsonld_blocks(html)
        f2, s2 = _from_jsonld(jsonlds)
        jsonld_features.extend(f2)
        jsonld_specs.update({k.lower():v for k,v in s2.items()})
        lines=_texts_from_dom(html)
        # keep English-ish text only
        lines=[ln for ln in lines if _lang_ok(ln)]
        all_lines.extend(lines)

    # Heuristic extraction
    specs=_extract_specs_from_lines(all_lines)
    for k,v in jsonld_specs.items():
        if k not in specs: specs[k]=v
    feats=_extract_features_from_lines(all_lines) + jsonld_features
    discls=_extract_disclaimers(all_lines)

    # LLM structuring if still thin
    if (len(feats) < 4 or not specs) and all_lines:
        llm_out=_llm_struct(all_lines)
        feats=_dedupe(feats + llm_out.get("features", []))
        for k,v in (llm_out.get("specs") or {}).items():
            if v and k not in specs: specs[k]=v
        discls=_dedupe(discls + (llm_out.get("disclaimers") or []))

    # Final floor using hints
    floor_feats=_dedupe((feats or []) + (vision_hints or []) + (ocr_hints or []))
    while len(floor_feats) < 4:
        floor_feats.append("notable design feature")
    feats=floor_feats[:20]

    claims=[ln for ln in all_lines if _looks_safe(ln) and _short_enough(ln, 16)][:20]

    bundle.update({
        "sources": urls,
        "features": feats,
        "specs": specs,
        "disclaimers": discls[:10],
        "claims": claims,
    })
    return bundle
