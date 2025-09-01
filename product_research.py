# product_research.py — 2025-09-01 (brand+model title/URL guard; category profiles; schema aligned)
"""
Category-aware product-info engine for ANY product type.

Core changes:
- Hard match: brand AND model must appear as exact tokens in page title or URL.
- Category profiles (allow/deny markers) to avoid cross-category mix-ups.
- Second SERP with quoted model and category to sharpen recall.
- Gemini fallback uses the exact same constraints.
- Output matches app.py (features_detailed/specs_detailed w/ confidence & sources).

This file intentionally keeps the spec-regex layer generic. Deep site-specific
parsers can be added later behind the same guards.
"""

from __future__ import annotations
import re, logging
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple, Set, Optional

from rapidfuzz import fuzz
from pint import UnitRegistry
from bs4 import BeautifulSoup

from fetcher import (
    search_serp,
    fetch_html,
    extract_structured_product,
    download_pdf,
)

ureg = UnitRegistry(auto_reduce_dimensions=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Domain rules ----------------
TRUSTED_DOMAINS = [
    "amazon.", "bestbuy.", "walmart.", "target.", "newegg.", "bhphotovideo.",
    "rei.", "backcountry.", "sony.", "sennheiser.", "bose.", "jabra.", "beats.",
    "apple.", "dell.", "lenovo.", "asus.", "acer.", "samsung.", "nike.", "adidas.",
    "rtings.com", "notebookcheck.net", "techradar.", "wired.", "cnet.", "gsmarena.com",
]
PDF_RE = re.compile(r"\.pdf($|[?#])", re.I)
DENY_HINTS = ["support.", "help.", "community.", "forum.", "youtube.", "facebook.",
              "reddit.", "pinterest.", "twitter.", "instagram.", "linkedin."]

# ---------------- Category profiles (same spirit as gemini_fetcher) ----------------
CATEGORY_ALLOW = {
    "headphones": ["headphone", "earbuds", "headset", "bluetooth", "anc", "codec", "drivers", "impedance"],
    "earbuds":    ["earbuds", "bluetooth", "anc", "codec", "ipx"],
    "smartphone": ["smartphone", "android", "ios", "display", "battery", "camera"],
    "laptop":     ["laptop", "notebook", "intel", "amd", "ryzen", "ram", "ssd"],
    "camera":     ["camera", "sensor", "lens", "aperture", "iso", "shutter"],
    "speaker":    ["speaker", "soundbar", "bluetooth", "watt", "woofer", "dolby"],
    "vacuum":     ["vacuum", "suction", "pa", "dustbin", "hepa"],
    "coffee maker": ["coffee", "espresso", "bar", "brew"],
    "portable power station": ["wh", "kwh", "inverter", "ac outlet", "lifepo4", "solar input", "cycle life"],
}
CATEGORY_DENY = {
    "headphones": ["wh", "portable power station", "generator", "inverter", "lifepo4", "ac outlet", "solar"],
    "earbuds":    ["wh", "generator", "inverter", "lifepo4"],
    "smartphone": ["portable power station", "generator", "sofa", "detergent"],
    "laptop":     ["generator", "sofa", "toothbrush"],
    "camera":     ["generator", "chair", "shampoo"],
    "speaker":    ["generator", "lifepo4", "ac outlet"],
    "vacuum":     ["smartphone", "headphones", "generator"],
    "coffee maker": ["smartphone", "headphones", "generator"],
    "portable power station": ["headphones", "earbuds", "anc", "drivers", "impedance"],
}

# ---------------- Generic spec regex (lightweight, cross-category) ----------------
SPEC_RE: List[Tuple[str, re.Pattern]] = [
    ("bluetooth_version", re.compile(r"bluetooth\s*(\d(?:\.\d)?)", re.I)),
    ("battery_life_h", re.compile(r"battery\s*(life|playtime|runtime)[^\d]{0,15}(\d+(?:\.\d+)?)\s*(h|hr|hrs|hours?)", re.I)),
    ("weight_g", re.compile(r"\bweight[^\d]{0,12}(\d+(?:\.\d+)?)\s*(g|grams?)\b", re.I)),
    ("weight_kg", re.compile(r"\bweight[^\d]{0,12}(\d+(?:\.\d+)?)\s*(kg|kilograms?)\b", re.I)),
    ("capacity_wh", re.compile(r"(\d+(?:\.\d+)?)\s*wh\b", re.I)),
    ("drivers_mm", re.compile(r"(\d+(?:\.\d+)?)\s*mm\s*(?:drivers?)", re.I)),
    ("impedance_ohm", re.compile(r"(\d+(?:\.\d+)?)\s*ohm", re.I)),
    ("freq_response_hz", re.compile(r"(\d+(?:\.\d+)?)\s*hz\s*[-–]\s*(\d+(?:\.\d+)?)\s*hz", re.I)),
    ("dimensions_in", re.compile(r"(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)(?:\s*(in|inch|inches))", re.I)),
    ("dimensions_cm", re.compile(r"(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)(?:\s*(cm|centimeters?))", re.I)),
]

FEATURE_HINTS = ("feature", "key", "highlight", "benefit", "overview", "description",
                 "feature-bullets", "detail-bullets")
JUNK_HINTS = ("breadcrumb", "menu", "nav", "navbar", "toolbar", "shortcut", "hotkey", "category", "department")

_SHORTCUT_RE = re.compile(r"\b(shift|ctrl|control|alt|cmd|command|option)\b\s*(\+|plus)", re.I)
_BREADCRUMB_RE = re.compile(r"^\s*(home|orders|cart|search|account|sign in|sign out)\b", re.I)
_CATEGORY_SINGLETON_RE = re.compile(r"^\s*(chairs|camping|camping & hiking|outdoor recreation)\s*$", re.I)

# ---------------- Helpers ----------------
def _unit(qty):
    try: return qty.to_base_units().magnitude
    except Exception: return qty

def _norm_val(k: str, v: str) -> str:
    try:
        if k in {"battery_life_h", "drivers_mm"}: return str(float(re.sub(r"[^0-9.]+", "", v)))
        if k in {"weight_g"}: return f"{float(re.sub(r'[^0-9.]+', '', v)):.2f} g"
        if k in {"weight_kg"}: return f"{float(re.sub(r'[^0-9.]+', '', v)):.2f} kg"
    except Exception:
        pass
    return (v or "").strip()

def _host(u: str) -> str:
    from urllib.parse import urlparse
    return (urlparse(u).hostname or "").lower()

def _url_ok(u: str) -> bool:
    h = _host(u)
    return not any(d in h for d in DENY_HINTS)

def _brand_in_domain(brand: str, url: str) -> bool:
    return brand.lower() in _host(url)

def _title_url_has_brand_model(title: str, url: str, brand: str, model: str) -> bool:
    b = brand.lower().strip()
    m = model.lower().strip()
    # token-ish match in title or URL
    t = (title or "").lower()
    u = (url or "").lower()
    def has_token(s: str, tok: str) -> bool:
        return re.search(rf"(?:^|[^a-z0-9]){re.escape(tok)}(?:[^a-z0-9]|$)", s) is not None
    return (has_token(t, b) or has_token(u, b)) and (has_token(t, m) or has_token(u, m))

def _category_ok(text: str, cat: str) -> bool:
    if not cat: return True
    t = (text or "").lower()
    allows = CATEGORY_ALLOW.get(cat, [])
    denies = CATEGORY_DENY.get(cat, [])
    if allows and not any(k in t for k in allows):
        return False
    if any(k in t for k in denies):
        return False
    return True

# ---------------- Data classes ----------------
@dataclass
class SpecRecord:
    brand: str
    model: str
    category_hint: str = ""
    specs: Dict[str, str] = field(default_factory=dict)
    features: Dict[str, set] = field(default_factory=lambda: defaultdict(set))  # feature → {urls or "gemini"}
    sources: Dict[str, Dict[str, str]] = field(default_factory=dict)           # url → attrs
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

    def finalise(self):
        votes = Counter()
        for attrs in self.sources.values():
            for k in attrs:
                votes[k.lower()] += 1
        diversity = len({_host(u) for u in self.sources})
        self.confidence = min(1.0, (len(votes) / 7.0) * (1 + 0.1 * max(diversity - 1, 0)))

# ---------------- Extraction ----------------
def _is_junk_bullet(li: Any) -> bool:
    txt = li.get_text(" ", strip=True)
    if _SHORTCUT_RE.search(txt): return True
    if _BREADCRUMB_RE.search(txt): return True
    if _CATEGORY_SINGLETON_RE.match(txt): return True
    ptr = li
    for _ in range(4):
        ptr = ptr.parent or ptr
        classes = " ".join(ptr.get("class") or [])
        id_ = ptr.get("id") or ""
        if any(h in (classes + " " + id_).lower() for h in JUNK_HINTS):
            return True
    return False

def _extract_features_from_html(html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    bullets: List[str] = []
    for ul in soup.find_all(["ul", "ol"]):
        cls = " ".join(ul.get("class") or []) + " " + (ul.get("id") or "")
        if any(hint in cls.lower() for hint in FEATURE_HINTS):
            for li in ul.find_all("li"):
                if not _is_junk_bullet(li):
                    t = li.get_text(" ", strip=True)
                    if 20 <= len(t) <= 160:
                        bullets.append(t)
    if len(bullets) < 4:
        for li in soup.find_all("li")[:40]:
            if not _is_junk_bullet(li):
                t = li.get_text(" ", strip=True)
                if 20 <= len(t) <= 160:
                    bullets.append(t)
    out: List[str] = []
    for b in bullets:
        if not any(fuzz.token_set_ratio(b, o) > 90 for o in out):
            out.append(b)
    return out[:12]

def _extract_specs_generic(text: str) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    for k, pat in SPEC_RE:
        m = pat.search(text)
        if not m: continue
        if k == "freq_response_hz":
            attrs[k] = f"{m.group(1)}-{m.group(2)}"
        elif k in ("weight_g", "weight_kg"):
            attrs[k] = f"{m.group(1)} {m.group(2) if m.lastindex and m.lastindex>=2 else ''}".strip()
        else:
            attrs[k] = m.group(1)
    return attrs

def _extract_from_html(html: str, url: str) -> Tuple[Dict[str, str], List[str], str]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    h1 = soup.find("h1")
    h1t = h1.get_text(" ", strip=True) if h1 else ""
    plain = soup.get_text(" ", strip=True)
    attrs = dict(extract_structured_product(html, url).get("attributes", {}))
    attrs.update(_extract_specs_generic(plain))
    feats = _extract_features_from_html(html)
    return attrs, feats, (title + " " + h1t)

def _extract_from_pdf(data: bytes) -> Dict[str, str]:
    try:
        import pdfplumber, io
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            txt = "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception:
        return {}
    return _extract_specs_generic(txt)

# ---------------- Scraper (with guards) ----------------
def get_product_record(brand: str, model: str, category_hint: str = "", *, max_urls: int = 12) -> 'SpecRecord':
    rec = SpecRecord(brand, model, category_hint=category_hint)
    queries = [
        f'{brand} "{model}" {category_hint} specifications'.strip(),
        f'{brand} {model} {category_hint} specs'.strip(),
    ]
    seen_urls: Set[str] = set()

    for q in queries:
        for url in search_serp(q, max_results=max_urls):
            if url in seen_urls: continue
            seen_urls.add(url)
            if not _url_ok(url): continue
            h = _host(url)
            if not any(d in h for d in TRUSTED_DOMAINS) and not PDF_RE.search(url):
                continue
            try:
                if PDF_RE.search(url):
                    pdf = download_pdf(url)
                    attrs = _extract_from_pdf(pdf) if pdf else {}
                    feats: List[str] = []
                    t = ""
                else:
                    html = fetch_html(url, timeout=12)
                    if not html: continue
                    attrs, feats, t = _extract_from_html(html, url)
            except Exception:
                attrs, feats, t = {}, [], ""

            # brand+model title/URL exact match
            if not _title_url_has_brand_model(t, url, brand, model):
                continue
            # category allow/deny
            if not _category_ok((t + " " + " ".join(feats[:3])), category_hint):
                continue

            if attrs: rec.merge_specs(attrs, url)
            if feats: rec.merge_features(feats, url)

            if rec.confidence > 0.85 and len(rec.features) >= 4:
                break

    rec.finalise()
    return rec

# ---------------- Public wrapper (used by app.py) ----------------
def research_product(
    brand: str,
    product: str,
    *,
    product_url_override: str = "",
    vision_category_tags: List[str] | None = None,   # we use this if provided
    max_results: int = 12,
) -> Dict[str, Any]:
    # choose category hint (first tag, or parse from product field keywords)
    category_hint = ""
    if vision_category_tags:
        try: category_hint = str(list(vision_category_tags)[0]).lower()
        except Exception: category_hint = ""
    for k in CATEGORY_ALLOW.keys():
        if k in (product or "").lower():
            category_hint = k
            break

    # 1) scraper with guards
    rec = get_product_record(brand, product, category_hint=category_hint, max_urls=max_results)

    # 2) Gemini fallback (same constraints)
    if not rec.specs and not rec.features and not product_url_override:
        from gemini_fetcher import gemini_product_info
        data = gemini_product_info(brand, product, category_hint=category_hint)
        if data.get("status") == "OK":
            for k, v in data.get("specs", {}).items():
                nk = k.lower()
                rec.specs[nk] = _norm_val(nk, v)
                rec.specs_from_gemini.add(nk)
            for feat in data.get("features", []):
                rec.features.setdefault(feat, set()).add("gemini")
            for cit in data.get("citations", []):
                attr = (cit.get("attr") or "").lower(); url = cit.get("url")
                if attr and url:
                    rec.sources.setdefault(url, {})
                    if attr in rec.specs:
                        rec.sources[url][attr] = rec.specs[attr]
            rec.confidence = 0.9

    # 3) Flatten for app.py
    simple_feats = list(rec.features.keys())

    # features_detailed
    detailed_feats: List[Dict[str, Any]] = []
    for ftxt, urls in rec.features.items():
        n = len([u for u in urls if u != "gemini"])
        used_gemini = ("gemini" in urls)
        conf = 0.55 + 0.15 * min(n, 3) + (0.15 if used_gemini else 0.0)
        conf = max(0.5, min(0.95, conf))
        detailed_feats.append({
            "text": ftxt,
            "confidence": round(conf, 3),
            "sources": [{"url": u} for u in urls if u != "gemini"] or [{"url": "gemini"}],
        })

    # specs_detailed
    specs_detailed: List[Dict[str, Any]] = []
    for k, v in rec.specs.items():
        urls_with_attr = [u for u, attrs in rec.sources.items() if k in attrs]
        n = len(urls_with_attr)
        came_from_gemini = (k in rec.specs_from_gemini)
        bonus = 0.0
        for u in urls_with_attr:
            if "amazon." in _host(u): bonus = max(bonus, 0.1)
            if _brand_in_domain(brand, u): bonus = max(bonus, 0.1)
        conf = 0.55 + 0.15 * min(n, 3) + bonus + (0.05 if came_from_gemini else 0.0)
        conf = max(0.5, min(0.98, conf))
        specs_detailed.append({
            "key": k,
            "value": v,
            "confidence": round(conf, 3),
            "sources": [{"url": u} for u in urls_with_attr] or ([{"url": "gemini"}] if came_from_gemini else []),
        })

    return {
        "query": f"{brand} {product}",
        "sources": [{"title": _host(u), "url": u} for u in rec.sources],
        "specs": rec.specs,
        "features": simple_feats,
        "features_detailed": detailed_feats,
        "specs_detailed": specs_detailed,
        "disclaimers": [],
        "visual_hints": [],
        "warnings": [] if rec.confidence >= 0.75 else ["Low-confidence extraction"],
    }
