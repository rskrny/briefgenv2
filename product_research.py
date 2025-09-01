# product_research.py — 2025-09-01 (brand-domain anchor; quoted-model SERP; strict category filters)

from __future__ import annotations
import re, logging
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple, Set, Optional

from rapidfuzz import fuzz
from pint import UnitRegistry
from bs4 import BeautifulSoup

from fetcher import search_serp, fetch_html, extract_structured_product, download_pdf

ureg = UnitRegistry(auto_reduce_dimensions=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Trusted domains: add brand site first if available
TRUSTED_DOMAINS = [
    "siawag.com",  # anchor for this brand
    "amazon.", "bestbuy.", "walmart.", "target.", "newegg.", "bhphotovideo.",
    "rei.", "backcountry.", "sony.", "sennheiser.", "bose.", "jabra.", "beats.",
    "apple.", "dell.", "lenovo.", "asus.", "acer.", "samsung.",
    "rtings.com", "notebookcheck.net", "techradar.", "wired.", "cnet.", "gsmarena.com",
]
PDF_RE = re.compile(r"\.pdf($|[?#])", re.I)
DENY_HINTS = ["support.", "help.", "community.", "forum.", "youtube.", "facebook.",
              "reddit.", "pinterest.", "twitter.", "instagram.", "linkedin."]

# Expanded category allow/deny markers
CATEGORY_ALLOW = {
    "headphones": ["headphone", "headset", "bluetooth", "anc", "noise cancelling",
                   "impedance", "frequency response", "codec"],
    "earbuds": ["earbuds", "tws", "charging case", "enc", "bluetooth", "anc",
                "ipx", "playtime", "fast charging"],
}
CATEGORY_DENY = {
    "headphones": ["generator", "wh", "inverter", "lifepo4", "solar", "symbian", "android"],
    "earbuds": ["generator", "wh", "inverter", "lifepo4", "symbian", "android"],
}

# Generic spec regex
SPEC_RE: List[Tuple[str, re.Pattern]] = [
    ("bluetooth_version", re.compile(r"bluetooth\s*(\d(?:\.\d)?)", re.I)),
    ("battery_life_h", re.compile(r"battery\s*(life|playtime)[^\d]{0,15}(\d+(?:\.\d+)?)\s*(h|hr|hrs|hours?)", re.I)),
    ("weight_g", re.compile(r"\bweight[^\d]{0,12}(\d+(?:\.\d+)?)\s*(g|grams?)\b", re.I)),
    ("drivers_mm", re.compile(r"(\d+(?:\.\d+)?)\s*mm\s*(?:drivers?)", re.I)),
    ("impedance_ohm", re.compile(r"(\d+(?:\.\d+)?)\s*ohm", re.I)),
    ("freq_response_hz", re.compile(r"(\d+(?:\.\d+)?)\s*hz\s*[-–]\s*(\d+(?:\.\d+)?)\s*hz", re.I)),
]

FEATURE_HINTS = ("feature", "key", "highlight", "benefit", "overview", "description",
                 "feature-bullets", "detail-bullets")

# ---------------- Helpers ----------------
def _norm_val(k: str, v: str) -> str:
    try:
        if k in {"battery_life_h", "drivers_mm"}:
            return str(float(re.sub(r"[^0-9.]+", "", v)))
        if k == "weight_g":
            return f"{float(re.sub(r'[^0-9.]+', '', v)):.2f} g"
    except Exception:
        pass
    return (v or "").strip()

def _host(u: str) -> str:
    from urllib.parse import urlparse
    return (urlparse(u).hostname or "").lower()

def _url_ok(u: str) -> bool:
    h = _host(u)
    return not any(d in h for d in DENY_HINTS)

def _title_url_has_brand_model(title: str, url: str, brand: str, model: str) -> bool:
    b, m = brand.lower(), model.lower()
    t, u = (title or "").lower(), (url or "").lower()
    return b in (t + u) and m in (t + u)

def _category_ok(text: str, cat: str) -> bool:
    if not cat: return True
    t = (text or "").lower()
    allows, denies = CATEGORY_ALLOW.get(cat, []), CATEGORY_DENY.get(cat, [])
    if allows and not any(k in t for k in allows):
        return False
    if any(k in t for k in denies):
        return False
    return True

# ---------------- Data class ----------------
@dataclass
class SpecRecord:
    brand: str
    model: str
    category_hint: str = ""
    specs: Dict[str, str] = field(default_factory=dict)
    features: Dict[str, set] = field(default_factory=lambda: defaultdict(set))
    sources: Dict[str, Dict[str, str]] = field(default_factory=dict)
    specs_from_gemini: Set[str] = field(default_factory=set)
    confidence: float = 0.0

    def merge_specs(self, attrs: Dict[str, str], url: str):
        if not attrs: return
        self.sources[url] = self.sources.get(url, {})
        for k, v in attrs.items():
            nk = k.lower()
            self.sources[url][nk] = _norm_val(nk, v)
            if nk not in self.specs:
                self.specs[nk] = self.sources[url][nk]

    def merge_features(self, feats: List[str], url: str):
        for f in feats:
            if f and isinstance(f, str):
                self.features[f].add(url)

# ---------------- Extraction ----------------
def _extract_features_from_html(html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for ul in soup.find_all(["ul", "ol"]):
        cls = " ".join(ul.get("class") or []) + " " + (ul.get("id") or "")
        if any(h in cls.lower() for h in FEATURE_HINTS):
            for li in ul.find_all("li"):
                txt = li.get_text(" ", strip=True)
                if 20 <= len(txt) <= 160:
                    out.append(txt)
    deduped = []
    for b in out:
        if not any(fuzz.token_set_ratio(b, o) > 90 for o in deduped):
            deduped.append(b)
    return deduped[:12]

def _extract_from_html(html: str, url: str) -> Tuple[Dict[str, str], List[str], str]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    h1 = soup.find("h1")
    h1t = h1.get_text(" ", strip=True) if h1 else ""
    plain = soup.get_text(" ", strip=True)
    attrs = dict(extract_structured_product(html, url).get("attributes", {}))
    for k, pat in SPEC_RE:
        m = pat.search(plain)
        if m and k not in attrs:
            attrs[k] = m.group(1) if k != "freq_response_hz" else f"{m.group(1)}-{m.group(2)}"
    feats = _extract_features_from_html(html)
    return attrs, feats, (title + " " + h1t)

# ---------------- Scraper ----------------
def get_product_record(brand: str, model: str, category_hint: str = "", *, max_urls: int = 12) -> SpecRecord:
    rec = SpecRecord(brand, model, category_hint)
    queries = [
        f'{brand} "{model}" {category_hint} specifications',
        f'{brand} {model} {category_hint} product page',
    ]
    seen: Set[str] = set()
    for q in queries:
        for url in search_serp(q, max_results=max_urls):
            if url in seen: continue
            seen.add(url)
            if not _url_ok(url): continue
            h = _host(url)
            if not any(d in h for d in TRUSTED_DOMAINS) and not PDF_RE.search(url):
                continue
            try:
                html = fetch_html(url, timeout=12) if not PDF_RE.search(url) else None
                if not html: continue
                attrs, feats, t = _extract_from_html(html, url)
            except Exception:
                continue
            # Relaxed title rule if brand domain
            if _host(url).endswith(brand.lower()+".com"):
                title_ok = brand.lower() in (t+url).lower() or model.lower() in (t+url).lower()
            else:
                title_ok = _title_url_has_brand_model(t, url, brand, model)
            if not title_ok: continue
            if not _category_ok(t + " ".join(feats[:3]), category_hint): continue
            if attrs: rec.merge_specs(attrs, url)
            if feats: rec.merge_features(feats, url)
    return rec

# ---------------- Public wrapper ----------------
def research_product(
    brand: str, product: str,
    *, product_url_override: str = "",
    vision_category_tags: List[str] | None = None,
    max_results: int = 12,
) -> Dict[str, Any]:
    cat = ""
    if vision_category_tags:
        try: cat = str(list(vision_category_tags)[0]).lower()
        except Exception: pass
    rec = get_product_record(brand, product, category_hint=cat, max_urls=max_results)
    if not rec.specs and not rec.features and not product_url_override:
        from gemini_fetcher import gemini_product_info
        data = gemini_product_info(brand, product, category_hint=cat)
        if data.get("status") == "OK":
            for k,v in data.get("specs",{}).items():
                rec.specs[k.lower()] = _norm_val(k,v)
                rec.specs_from_gemini.add(k.lower())
            for f in data.get("features",[]): rec.features.setdefault(f,set()).add("gemini")
    # Flatten for app.py
    detailed_feats = [{"text":f,"confidence":0.7,"sources":[{"url":u} for u in urls]}
                      for f,urls in rec.features.items()]
    detailed_specs = [{"key":k,"value":v,"confidence":0.7,
                       "sources":[{"url":u} for u,attrs in rec.sources.items() if k in attrs]}
                      for k,v in rec.specs.items()]
    return {
        "query": f"{brand} {product}",
        "specs": rec.specs,
        "features": list(rec.features.keys()),
        "features_detailed": detailed_feats,
        "specs_detailed": detailed_specs,
        "sources": [{"title": _host(u), "url": u} for u in rec.sources],
        "warnings": [] if rec.confidence >= 0.75 else ["Low confidence"],
    }
