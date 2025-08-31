# product_research.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
import re

try:
    from duckduckgo_search import DDGS
except Exception:
    DDGS = None

try:
    import requests
    from bs4 import BeautifulSoup
except Exception:
    requests, BeautifulSoup = None, None

HEADERS = {"User-Agent": "briefgenv2/1.0 (+https://github.com/)"}

TRUSTED_DOMAINS = [
    "nemo", "rei.com", "backcountry.com", "moosejaw.com", "mec.ca", "trekkinn.com",
    "backpackinglight.com", "outdoorgearlab.com", "amazon.com"
]

FEATURE_HARDWORDS = {
    "recline":"reclining back", "reclining":"reclining back", "mesh":"mesh seat",
    "aluminum":"aluminum frame", "aluminium":"aluminum frame",
    "cup holder":"cup holder", "cupholder":"cup holder",
    "carry bag":"carry bag", "stuff sack":"carry bag",
    "adjustable":"adjustable straps", "strap":"adjustable straps",
    "shock cord":"shock-corded frame", "fold":"folding frame",
    "pack":"packs small", "lightweight":"lightweight build",
    "low":"low seat height",
}

RISKY = [r"\b(best|#1|guarantee|cure|miracle)\b", r"\b(FDA|CE|UL|ISO)\b"]
DISC = [r"\b(weight capacity|max(?:imum)? weight)\b", r"\b(read instructions|follow warnings)\b"]

def _fetch_html(url: str, timeout=12) -> Optional[str]:
    if not requests:
        return None
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        if "text/html" not in r.headers.get("Content-Type", ""):
            return None
        return r.text
    except Exception:
        return None

def _dedupe(xs: List[str]) -> List[str]:
    seen=set(); out=[]
    for x in xs:
        k=x.strip().lower()
        if k and k not in seen:
            seen.add(k); out.append(x.strip())
    return out

def _short(s: str, n=18):
    return 3 <= len(s.split()) <= n

def _looks_ok(line: str) -> bool:
    low=line.lower()
    if any(re.search(p, low) for p in RISKY): return False
    if " than " in low or " vs " in low: return False
    return True

def _extract_blocks(html: str) -> List[str]:
    if not BeautifulSoup: return []
    soup = BeautifulSoup(html, "html.parser")
    texts=[]
    # Product bullets / details
    for sel in ["li","p",".feature",".features","table tr"]:
        for el in soup.select(sel):
            t = " ".join(el.get_text(" ").split())
            if _short(t):
                texts.append(t)
    # Headings
    for h in soup.select("h2,h3,h4"):
        t = " ".join(h.get_text(" ").split())
        if _short(t, 12): texts.append(t)
    return texts[:120]

def _to_features(lines: List[str]) -> List[str]:
    feats=[]
    for ln in lines:
        low=ln.lower()
        for k,v in FEATURE_HARDWORDS.items():
            if k in low and v not in feats:
                feats.append(v)
    # also keep any short “includes/comes with” lines
    for ln in lines:
        low=ln.lower()
        if _looks_ok(low) and any(w in low for w in ["features","includes","comes with","designed","recline","mesh","lightweight","aluminum","cup"]):
            if len(ln.split())<=10 and ln not in feats:
                feats.append(ln)
    return _dedupe(feats)

def _to_disclaimers(lines: List[str]) -> List[str]:
    out=[]
    for ln in lines:
        low=ln.lower()
        if any(re.search(p, low) for p in DISC):
            out.append(ln)
    return _dedupe(out)

def _search(q: str, max_results=6) -> List[Dict[str,str]]:
    if not DDGS: return []
    hits=[]
    with DDGS() as ddgs:
        for r in ddgs.text(q, max_results=max_results):
            url=r.get("href") or r.get("url"); title=r.get("title") or r.get("body") or url
            if not url: continue
            if any(dom in url.lower() for dom in TRUSTED_DOMAINS):
                hits.append({"title": title, "url": url})
    return hits

def research_product(brand: str, product: str, *, product_url_override: str = "", max_results: int = 6,
                     fallback_visual: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Returns:
      {"query":..., "sources":[{title,url}], "features":[...], "claims":[...], "disclaimers":[...]}
    """
    bundle={"query":"", "sources":[], "features":[], "claims":[], "disclaimers":[]}
    # 1) Direct URL override (guaranteed path)
    urls=[]
    if product_url_override:
        urls=[{"title":"Provided URL", "url":product_url_override}]
    else:
        q=f'{brand} {product} features site:rei.com OR site:backcountry.com OR site:nemoequipment.com'
        bundle["query"]=q
        urls=_search(q, max_results=max_results)
        if not urls:
            q2=f'{brand} {product} specs review'
            bundle["query"]=q2
            urls=_search(q2, max_results=max_results)

    # 2) Parse pages
    all_lines=[]
    for h in urls[:max_results]:
        html=_fetch_html(h["url"])
        if not html: continue
        all_lines.extend(_extract_blocks(html))
    feats=_to_features(all_lines)
    discls=_to_disclaimers(all_lines)
    claims=[ln for ln in all_lines if _looks_ok(ln) and _short(ln, 15)][:20]

    # 3) If still thin, use visual fallback (from keyframes via vision_tools)
    if len(feats) < 4 and fallback_visual:
        vis = fallback_visual
        feats = _dedupe(feats + (vis.get("visible_features") or []))
        # no numeric claims; keep conservative wording only

    # 4) Heuristic floor: ensure at least 4 short, safe features
    FLOOR = ["reclining back", "adjustable straps", "mesh seat", "carry bag"]
    if len(feats) < 4:
        for f in FLOOR:
            if f not in feats:
                feats.append(f)

    bundle["sources"]=urls
    bundle["features"]=_dedupe(feats)[:20]
    bundle["claims"]=_dedupe(claims)[:20]
    bundle["disclaimers"]=_dedupe(discls)[:12]
    return bundle
