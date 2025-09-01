# product_research.py — v5.0  (2025-09-01)
"""
Open-web product-info engine (no paid APIs).

Workflow
┌ search_serp      – Google HTML / DuckDuckGo fallback
├ fetch_html       – Playwright→requests
├ extract_structured_product
├ regex heuristics – fills gaps
└ merge            – SpecRecord with confidence score

Public helpers
• get_product_record(brand, model)
• research_product(...)  # legacy wrapper used by app.py
"""
from __future__ import annotations
import re, logging, os
from dataclasses import dataclass, field
from collections import Counter
from typing import Dict, List, Any, Tuple

from rapidfuzz import fuzz
from pint import UnitRegistry
ureg = UnitRegistry(auto_reduce_dimensions=True)

from fetcher import (
    search_serp,
    fetch_html,
    extract_structured_product,
    download_pdf,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ────────────────── config ──────────────────
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
    ("battery_life_h", re.compile(r"battery\\s*(life|playtime|runtime)[^\\d]{0,15}(\\d+(?:\\.\\d+)?)\\s*(hours?|hrs?|h)", re.I)),
    ("capacity_mah",  re.compile(r"(\\d+(?:\\.\\d+)?)\\s*mah", re.I)),
    ("weight_g",      re.compile(r"weight[^\\d]{0,10}(\\d+(?:\\.\\d+)?)\\s*(g|grams?)", re.I)),
    ("weight_lb",     re.compile(r"weight[^\\d]{0,10}(\\d+(?:\\.\\d+)?)\\s*(lb|pounds?)", re.I)),
    ("bluetooth_version", re.compile(r"bluetooth\\s*(\\d(?:\\.\\d)?)", re.I)),
    ("ip_rating",     re.compile(r"ip\\s*([0-9]{2})", re.I)),
]

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
    try:
        return (urlparse(u).hostname or "").lower()
    except Exception:
        return u.lower()

def _url_ok(u: str) -> bool:
    h = _host(u)
    if any(d in h for d in DENY_HINTS):
        return False
    return True

# ────────────────── core dataclass ──────────────────
@dataclass
class SpecRecord:
    brand: str
    model: str
    specs: Dict[str, str] = field(default_factory=dict)
    sources: Dict[str, Dict[str, str]] = field(default_factory=dict)  # url → attrs
    confidence: float = 0.0

    def merge(self, attrs: Dict[str, str], url: str):
        self.sources[url] = attrs
        for k, v in attrs.items():
            nk = k.lower()
            if nk not in self.specs:
                self.specs[nk] = _norm_val(nk, v)

    def finalise(self):
        votes = Counter()
        for attrs in self.sources.values():
            for k in attrs:
                votes[k.lower()] += 1
        diversity = len({ _host(u) for u in self.sources })
        self.confidence = min(1.0, (len(votes) / 7.0) * (1 + 0.1 * max(diversity - 1, 0)))

# ────────────────── extraction helpers ──────────────────
def _extract_from_html(html: str, url: str) -> Dict[str, str]:
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
    return attrs

def _extract_from_pdf(data: bytes) -> Dict[str, str]:
    try:
        import pdfplumber, io
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            txt = "\\n".join(page.extract_text() or "" for page in pdf.pages)
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

# ────────────────── public API ──────────────────
def get_product_record(brand: str, model: str, *, max_urls: int = 12) -> SpecRecord:
    rec = SpecRecord(brand, model)
    query = f"{brand} {model} specifications"
    for url in search_serp(query, max_results=max_urls):
        if not _url_ok(url):
            continue
        h = _host(url)
        if not any(w in h for w in WHITELIST_HINTS + REVIEW_SITES) and not PDF_RE.search(url):
            continue  # unknown site → skip
        try:
            if PDF_RE.search(url):
                pdf = download_pdf(url)
                attrs = _extract_from_pdf(pdf) if pdf else {}
            else:
                html = fetch_html(url, timeout=12)
                attrs = _extract_from_html(html, url) if html else {}
        except Exception:
            attrs = {}
        if attrs:
            rec.merge(attrs, url)
        if rec.confidence > 0.8:
            break  # early exit
    rec.finalise()
    return rec

# ────────── legacy wrapper (app.py still calls this) ──────────
def research_product(
    brand: str,
    product: str,
    *,
    product_url_override: str = "",
    vision_category_tags: List[str] | None = None,
    max_results: int = 12,
) -> Dict[str, Any]:
    rec = get_product_record(brand, product, max_urls=max_results)
    return {
        "query": f"{brand} {product}",
        "sources": [{"title": _host(u), "url": u} for u in rec.sources],
        "specs": rec.specs,
        "features": [],
        "features_detailed": [],
        "specs_detailed": [
            {"key": k, "value": v, "sources": [{"url": u} for u in rec.sources]}
            for k, v in rec.specs.items()
        ],
        "disclaimers": [],
        "visual_hints": [],
        "warnings": [] if rec.confidence >= 0.75 else ["Low-confidence extraction"],
    }
