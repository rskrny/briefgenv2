# product_research.py — 2025-09-01
"""
Product-info engine for briefgenv2.

• First tries open-web scraping (Google SERP + HTML/PDF parsing).
• If nothing verified, falls back to Gemini-Pro via gemini_fetcher.py.
• Returns dict that app.py already expects.
"""

from __future__ import annotations
import re, logging
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple

from rapidfuzz import fuzz
from pint import UnitRegistry

# ── local deps ──
from fetcher import (
    search_serp,
    fetch_html,
    extract_structured_product,
    download_pdf,
)

ureg = UnitRegistry(auto_reduce_dimensions=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ────────────────── static config ──────────────────
WHITELIST_HINTS = [
    "amazon.", "bestbuy.", "walmart.", "target.", "homedepot.", "newegg.",
    "bhphotovideo.", "lenovo.", "dell.", "asus.", "acer.", "apple.",
]
REVIEW_SITES = ["rtings.com", "notebookcheck.net", "gsmarena.com", "techradar."]
PDF_RE = re.compile(r"\.pdf($|[?#])", re.I)
DENY_HINTS = [
    "support.", "help.", "community.", "forum.", "youtube.", "facebook.",
    "reddit.", "pinterest.", "twitter.", "instagram.", "linkedin.",
]

SPEC_RE: List[Tuple[str, re.Pattern]] = [
    ("battery_life_h", re.compile(r"battery\s*(life|playtime|runtime)[^\d]{0,15}(\d+(?:\.\d+)?)\s*(hours?|hrs?|h)", re.I)),
    ("capacity_mah",  re.compile(r"(\d+(?:\.\d+)?)\s*mah", re.I)),
    ("weight_g",      re.compile(r"weight[^\d]{0,10}(\d+(?:\.\d+)?)\s*(g|grams?)", re.I)),
    ("weight_lb",     re.compile(r"weight[^\d]{0,10}(\d+(?:\.\d+)?)\s*(lb|pounds?)", re.I)),
    ("bluetooth_version", re.compile(r"bluetooth\s*(\d(?:\.\d)?)", re.I)),
    ("ip_rating",     re.compile(r"ip\s*([0-9]{2})", re.I)),
]

FEATURE_H1NTS = ("feature", "key", "highlight", "benefit", "overview", "description")

# ────────────────── helpers ──────────────────
def _unit(qty):
    try:
        return qty.to_base_units().magnitude
    except Exception:
        return qty

def _norm_val(k: str, v: str) -> str:
    try:
        if k.startswith("weight"):
            return f"{_unit(ureg(v.strip())):.2f} g"
        if k.startswith("capacity"):
            return f"{float(v):g} mAh"
    except Exception:
        pass
    return v.strip()

def _host(u: str) -> str:
    from urllib.parse import urlparse
    return (urlparse(u).hostname or "").lower()

def _url_ok(u: str) -> bool:
    h = _host(u)
    return not any(d in h for d in DENY_HINTS)

# ────────────────── data classes ──────────────────
@dataclass
class SpecRecord:
    brand: str
    model: str
    specs: Dict[str, str] = field(default_factory=dict)
    features: Dict[str, set] = field(default_factory=lambda: defaultdict(set))  # feature → {urls}
    sources: Dict[str, Dict[str, str]] = field(default_factory=dict)           # url → attrs
    confidence: float = 0.0

    def merge_specs(self, attrs: Dict[str, str], url: str):
        self.sources[url] = attrs
        for k, v in attrs.items():
            nk = k.lower()
            if nk not in self.specs:
                self.specs[nk] = _norm_val(nk, v)

    def merge_features(self, feats: List[str], url: str):
        for f in feats:
            self.features[f].add(url)

    def finalise(self):
        votes = Counter()
        for attrs in self.sources.values():
            for k in attrs:
                votes[k.lower()] += 1
        diversity = len({ _host(u) for u in self.sources })
        self.confidence = min(1.0, (len(votes) / 7.0) * (1 + 0.1 * max(diversity - 1, 0)))

# ────────────────── HTML/PDF extraction ──────────────────
def _extract_features_from_html(html: str) -> List[str]:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []
    soup = BeautifulSoup(html, "html.parser")
    bullets: List[str] = []
    for tag in soup.find_all(["ul", "ol"]):
        wrapper = tag
        for _ in range(3):
            cls = " ".join(wrapper.get("class") or [])
            if any(hint in cls.lower() for hint in FEATURE_H1NTS):
                bullets.extend(li.get_text(" ", strip=True) for li in tag.find_all("li"))
                break
            wrapper = wrapper.parent or wrapper
    if len(bullets) < 4:
        for li in soup.find_all("li")[:30]:
            txt = li.get_text(" ", strip=True)
            if 20 < len(txt) < 120:
                bullets.append(txt)
    out: List[str] = []
    for b in bullets:
        if not any(fuzz.token_set_ratio(b, o) > 90 for o in out):
            out.append(b)
    return out[:12]

def _extract_from_html(html: str, url: str) -> Tuple[Dict[str, str], List[str]]:
    attrs = dict(extract_structured_product(html, url).get("attributes", {}))
    plain = re.sub(r"<[^>]+>", " ", html)
    for k, pat in SPEC_RE:
        m = pat.search(plain)
        if m:
            if "mah" in k:
                attrs[k] = m.group(1)
            elif k in ("weight_lb", "weight_g"):
                attrs[k] = f"{m.group(1)} {m.group(2)}"
            else:
                attrs[k] = m.group(1)
    feats = _extract_features_from_html(html)
    return attrs, feats

def _extract_from_pdf(data: bytes) -> Dict[str, str]:
    try:
        import pdfplumber, io
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            txt = "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception:
        return {}
    attrs = {}
    for k, pat in SPEC_RE:
        m = pat.search(txt)
        if m:
            if "mah" in k:
                attrs[k] = m.group(1)
            elif k in ("weight_lb", "weight_g"):
                attrs[k] = f"{m.group(1)} {m.group(2)}"
            else:
                attrs[k] = m.group(1)
    return attrs

# ────────────────── main function ──────────────────
def get_product_record(brand: str, model: str, *, max_urls: int = 12) -> SpecRecord:
    rec = SpecRecord(brand, model)
    query = f"{brand} {model} specifications"
    for url in search_serp(query, max_results=max_urls):
        if not _url_ok(url):
            continue
        h = _host(url)
        if not any(w in h for w in WHITELIST_HINTS + REVIEW_SITES) and not PDF_RE.search(url):
            continue
        try:
            if PDF_RE.search(url):
                pdf = download_pdf(url)
                attrs = _extract_from_pdf(pdf) if pdf else {}
                feats: List[str] = []
            else:
                html = fetch_html(url, timeout=12)
                attrs, feats = _extract_from_html(html, url) if html else ({}, [])
        except Exception:
            attrs, feats = {}, []
        if attrs:
            rec.merge_specs(attrs, url)
        if feats:
            rec.merge_features(feats, url)
        if rec.confidence > 0.8 and len(rec.features) >= 4:
            break
    rec.finalise()
    return rec

# ────────────────── public wrapper (used by app.py) ──────────────────
def research_product(
    brand: str,
    product: str,
    *,
    product_url_override: str = "",
    max_results: int = 12,
) -> Dict[str, Any]:
    # 1) Try scraper first
    rec = get_product_record(brand, product, max_urls=max_results)

    # 2) Fallback to Gemini if nothing verified and no manual URL
    if not rec.specs and not rec.features and not product_url_override:
        from gemini_fetcher import gemini_product_info
        data = gemini_product_info(brand, product)
        if data.get("status") == "OK":
            for k, v in data.get("specs", {}).items():
                rec.specs[k] = v
            for idx, feat in enumerate(data.get("features", [])):
                rec.features.setdefault(feat, set()).add("gemini")
            for cit in data.get("citations", []):
                attr = cit.get("attr")
                url  = cit.get("url")
                if attr and url:
                    rec.sources.setdefault(url, {})
                    if attr in rec.specs:
                        rec.sources[url][attr] = rec.specs[attr]
            rec.confidence = 0.9  # Gemini self-validated

    # 3) Flatten return format
    simple_feats = list(rec.features.keys())
    detailed_feats = [
        {"feature": f, "sources": [{"url": u} for u in urls]}
        for f, urls in rec.features.items()
    ]

    return {
        "query": f"{brand} {product}",
        "sources": [{"title": _host(u), "url": u} for u in rec.sources],
        "specs": rec.specs,
        "features": simple_feats,
        "features_detailed": detailed_feats,
        "specs_detailed": [
            {"key": k, "value": v, "sources": [{"url": u} for u in rec.sources]}
            for k, v in rec.specs.items()
        ],
        "disclaimers": [],
        "visual_hints": [],
        "warnings": [] if rec.confidence >= 0.75 else ["Low-confidence extraction"],
    }
