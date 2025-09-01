# product_research.py — 2025-09-01 (category-aware; schema aligned)
"""
Category-aware product-info engine.

Flow:
1) Scraper tries with a category-biased query (brand + model + category).
2) If empty/low-confidence, falls back to Gemini with a strict category constraint.
3) Returns dict expected by app.py (features_detailed/specs_detailed with confidence & sources).
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

# ───────────── domain rules ─────────────
WHITELIST_HINTS = [
    "amazon.", "bestbuy.", "walmart.", "target.", "homedepot.", "newegg.",
    "bhphotovideo.", "lenovo.", "dell.", "asus.", "acer.", "apple.",
    "rei.", "backcountry.", "sony.", "sennheiser.", "bose.", "jabra.", "beats."
]
REVIEW_SITES = ["rtings.com", "notebookcheck.net", "gsmarena.com", "techradar.", "wired.", "cnet."]
PDF_RE = re.compile(r"\.pdf($|[?#])", re.I)
DENY_HINTS = [
    "support.", "help.", "community.", "forum.", "youtube.", "facebook.",
    "reddit.", "pinterest.", "twitter.", "instagram.", "linkedin.",
]

# ───────────── regexes ─────────────
SPEC_RE: List[Tuple[str, re.Pattern]] = [
    # audio
    ("bluetooth_version", re.compile(r"bluetooth\s*(\d(?:\.\d)?)", re.I)),
    ("anc", re.compile(r"\bactive noise cancellation\b|\banc\b", re.I)),
    ("drivers_mm", re.compile(r"(\d+(?:\.\d+)?)\s*mm\s*(?:drivers?)", re.I)),
    ("impedance_ohm", re.compile(r"(\d+(?:\.\d+)?)\s*ohm", re.I)),
    ("freq_response_hz", re.compile(r"(\d+(?:\.\d+)?)\s*hz\s*[-–]\s*(\d+(?:\.\d+)?)\s*hz", re.I)),
    ("battery_life_h", re.compile(r"battery\s*(life|playtime|runtime)[^\d]{0,15}(\d+(?:\.\d+)?)\s*(h|hr|hrs|hours?)", re.I)),
    ("charge_time_h", re.compile(r"charge\s*time[^\d]{0,15}(\d+(?:\.\d+)?)\s*(h|hr|hrs|hours?)", re.I)),
    ("weight_g", re.compile(r"\bweight[^\d]{0,12}(\d+(?:\.\d+)?)\s*(g|grams?)\b", re.I)),
    # power station (used only if category is that)
    ("capacity_wh", re.compile(r"(\d+(?:\.\d+)?)\s*wh\b", re.I)),
    ("lifepo4", re.compile(r"\blife?po4\b", re.I)),
    ("ac_outlets", re.compile(r"\bac\s*outlets?\b", re.I)),
]

FEATURE_HINTS = ("feature", "key", "highlight", "benefit", "overview", "description", "feature-bullets", "detail-bullets")
JUNK_HINTS = ("breadcrumb", "menu", "nav", "navbar", "toolbar", "shortcut", "hotkey", "category", "department")

_SHORTCUT_RE = re.compile(r"\b(shift|ctrl|control|alt|cmd|command|option)\b\s*(\+|plus)", re.I)
_BREADCRUMB_RE = re.compile(r"^\s*(home|orders|cart|search|account|sign in|sign out)\b", re.I)
_CATEGORY_SINGLETON_RE = re.compile(r"^\s*(chairs|camping|camping & hiking|outdoor recreation)\s*$", re.I)

# ───────────── helpers ─────────────
def _unit(qty):
    try:
        return qty.to_base_units().magnitude
    except Exception:
        return qty

def _norm_val(k: str, v: str) -> str:
    try:
        if k.startswith("weight_") and v:
            if k.endswith("_g"):
                return f"{float(re.sub(r'[^0-9.]+', '', v)):g} g"
        if k in {"battery_life_h", "charge_time_h", "drivers_mm"}:
            return str(float(re.sub(r"[^0-9.]+", "", v)))
    except Exception:
        pass
    return (v or "").strip()

def _host(u: str) -> str:
    from urllib.parse import urlparse
    return (urlparse(u).hostname or "").lower()

def _url_ok(u: str) -> bool:
    h = _host(u)
    return not any(d in h for d in DENY_HINTS)

def _category_from_signals(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ["headphones", "headset", "earbuds"]):
        return "headphones"
    if any(w in t for w in ["power station", "generator", "inverter", "wh", "ac outlets"]):
        return "portable power station"
    return ""

# ───────────── data classes ─────────────
@dataclass
class SpecRecord:
    brand: str
    model: str
    category_hint: str = ""
    specs: Dict[str, str] = field(default_factory=dict)
    features: Dict[str, set] = field(default_factory=lambda: defaultdict(set))  # feature → {urls}
    sources: Dict[str, Dict[str, str]] = field(default_factory=dict)            # url → attrs
    specs_from_gemini: Set[str] = field(default_factory=set)
    confidence: float = 0.0

    def merge_specs(self, attrs: Dict[str, str], url: str):
        if not attrs:
            return
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

# ───────────── extraction ─────────────
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

def _extract_from_html_generic(html: str, url: str) -> Dict[str, str]:
    plain = re.sub(r"<[^>]+>", " ", html)
    attrs: Dict[str, str] = {}
    for k, pat in SPEC_RE:
        m = pat.search(plain)
        if not m:
            continue
        if k == "freq_response_hz":
            attrs[k] = f"{m.group(1)}-{m.group(2)}"
        elif k in ("weight_g",):
            attrs[k] = f"{m.group(1)} g"
        else:
            attrs[k] = m.group(1)
    return attrs

def _extract_from_html(html: str, url: str) -> Tuple[Dict[str, str], List[str]]:
    attrs = dict(extract_structured_product(html, url).get("attributes", {}))
    # generic regex sweep
    attrs.update(_extract_from_html_generic(html, url))
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
            if k == "freq_response_hz":
                attrs[k] = f"{m.group(1)}-{m.group(2)}"
            elif k in ("weight_g",):
                attrs[k] = f"{m.group(1)} g"
            else:
                attrs[k] = m.group(1)
    return attrs

# ───────────── category guard ─────────────
def _category_guard(candidate_text: str, desired: str) -> bool:
    """Return True if candidate_text matches desired category; False = reject."""
    desired = (desired or "").strip().lower()
    if not desired:
        return True
    cand = _category_from_signals(candidate_text)
    if not cand:
        # unknown → allow but not strong
        return True
    # strict match sets
    if desired in {"headphones", "earbuds", "headset"}:
        return cand == "headphones"
    if desired in {"portable power station", "generator"}:
        return cand == "portable power station"
    return True

# ───────────── scraper first ─────────────
def get_product_record(brand: str, model: str, category_hint: str = "", *, max_urls: int = 12) -> 'SpecRecord':
    rec = SpecRecord(brand, model, category_hint=category_hint)
    q = f"{brand} {model} {category_hint}".strip()
    for url in search_serp(f"{q} specifications", max_results=max_urls):
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
                text_for_guard = ""
            else:
                html = fetch_html(url, timeout=12)
                if not html:
                    continue
                attrs, feats = _extract_from_html(html, url)
                # use title + first h1 for category guard
                soup = BeautifulSoup(html, "html.parser")
                title = (soup.title.get_text(" ", strip=True) if soup.title else "") + " " + (soup.find("h1").get_text(" ", strip=True) if soup.find("h1") else "")
                text_for_guard = title + " " + " ".join(feats[:3])
        except Exception:
            attrs, feats, text_for_guard = {}, [], ""

        if text_for_guard and not _category_guard(text_for_guard, category_hint):
            # skip cross-category pages
            continue

        if attrs:
            rec.merge_specs(attrs, url)
        if feats:
            rec.merge_features(feats, url)

        if rec.confidence > 0.8 and len(rec.features) >= 4:
            break

    rec.finalise()
    return rec

# ───────────── public wrapper (used by app.py) ─────────────
def research_product(
    brand: str,
    product: str,
    *,
    product_url_override: str = "",
    vision_category_tags: List[str] | None = None,   # ← we will USE this now
    max_results: int = 12,
) -> Dict[str, Any]:
    # derive a sensible category hint
    category_hint = ""
    if vision_category_tags:
        # take the first tag as the category hint (app passes a set)
        try:
            category_hint = str(list(vision_category_tags)[0]).lower()
        except Exception:
            category_hint = ""
    # If the product string itself includes a likely category, use it
    for k in ["headphones", "earbuds", "headset", "portable power station", "generator"]:
        if k in (product or "").lower():
            category_hint = k
            break

    # 1) scraper (category-biased)
    rec = get_product_record(brand, product, category_hint=category_hint, max_urls=max_results)

    # 2) Gemini fallback (strict category)
    if not rec.specs and not rec.features and not product_url_override:
        from gemini_fetcher import gemini_product_info
        data = gemini_product_info(brand, product, category_hint=category_hint)
        if data.get("status") == "OK":
            for k, v in data.get("specs", {}).items():
                nk = k.lower()
                rec.specs[nk] = _norm_val(nk, v)
                rec.specs_from_gemini.add(nk)
            for feat in data.get("features", []):
                if feat and _category_guard(feat, category_hint):
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

    # Features detailed
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

    # Specs detailed
    specs_detailed: List[Dict[str, Any]] = []
    for k, v in rec.specs.items():
        urls_with_attr = [u for u, attrs in rec.sources.items() if k in attrs]
        n = len(urls_with_attr)
        came_from_gemini = (k in rec.specs_from_gemini)
        bonus = 0.0
        for u in urls_with_attr:
            h = _host(u)
            if "amazon." in h:
                bonus = max(bonus, 0.1)
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
