# product_research.py — 2025-09-01 (better PDP parsing + junk filters)
"""
Product-info engine for briefgenv2.

Improvements:
- Amazon PDP parser (tech-spec table + detail bullets)
- Filters out UI/shortcut/breadcrumb/category bullets
- Adds more spec patterns (dimensions, weight capacity, materials)
- Aligns schemas expected by app.py
"""

from __future__ import annotations
import re, logging
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple, Set

from rapidfuzz import fuzz
from pint import UnitRegistry
from bs4 import BeautifulSoup  # ensure installed

from fetcher import (
    search_serp,
    fetch_html,
    extract_structured_product,
    download_pdf,
)

ureg = UnitRegistry(auto_reduce_dimensions=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ────────────────── site/domain rules ──────────────────
WHITELIST_HINTS = [
    "amazon.", "bestbuy.", "walmart.", "target.", "homedepot.", "newegg.",
    "bhphotovideo.", "lenovo.", "dell.", "asus.", "acer.", "apple.", "rei.", "backcountry."
]
REVIEW_SITES = ["rtings.com", "notebookcheck.net", "gsmarena.com", "techradar.", "wired.", "cnet."]
PDF_RE = re.compile(r"\.pdf($|[?#])", re.I)
DENY_HINTS = [
    "support.", "help.", "community.", "forum.", "youtube.", "facebook.",
    "reddit.", "pinterest.", "twitter.", "instagram.", "linkedin.",
]

# ────────────────── spec regexes ──────────────────
SPEC_RE: List[Tuple[str, re.Pattern]] = [
    ("battery_life_h", re.compile(r"battery\s*(life|playtime|runtime)[^\d]{0,15}(\d+(?:\.\d+)?)\s*(h|hr|hrs|hours?)", re.I)),
    ("capacity_mah",  re.compile(r"(\d+(?:\.\d+)?)\s*mah", re.I)),
    ("bluetooth_version", re.compile(r"bluetooth\s*(\d(?:\.\d)?)", re.I)),
    ("ip_rating",     re.compile(r"\bip\s*([0-9]{2})\b", re.I)),
    # physicals
    ("weight_g",      re.compile(r"\bweight[^\d]{0,12}(\d+(?:\.\d+)?)\s*(g|grams?)\b", re.I)),
    ("weight_lb",     re.compile(r"\bweight[^\d]{0,12}(\d+(?:\.\d+)?)\s*(lb|lbs|pounds?)\b", re.I)),
    ("weight_capacity_lb", re.compile(r"(weight\s*capacity|max\s*load|supports\s*up\s*to)[^\d]{0,12}(\d+(?:\.\d+)?)\s*(lb|lbs|pounds?)", re.I)),
    ("dimensions_in", re.compile(r"(\d+(?:\.\d+)?)\s*(?:x|×)\s*(\d+(?:\.\d+)?)\s*(?:x|×)\s*(\d+(?:\.\d+)?)(?:\s*(in|inch|inches))", re.I)),
    ("dimensions_cm", re.compile(r"(\d+(?:\.\d+)?)\s*(?:x|×)\s*(\d+(?:\.\d+)?)\s*(?:x|×)\s*(\d+(?:\.\d+)?)(?:\s*(cm|centimeters?))", re.I)),
    ("material",      re.compile(r"\b(material|fabric)\b[^\n:]{0,15}:\s*([A-Za-z0-9 \-\/]+)", re.I)),
]

FEATURE_HINTS = ("feature", "key", "highlight", "benefit", "overview", "description", "feature-bullets", "detail-bullets")
JUNK_HINTS = ("breadcrumb", "menu", "nav", "navbar", "toolbar", "shortcut", "hotkey", "category", "department")

# ────────────────── helpers ──────────────────
def _unit(qty):
    try:
        return qty.to_base_units().magnitude
    except Exception:
        return qty

def _norm_val(k: str, v: str) -> str:
    try:
        if k.startswith("weight_") and v:
            # auto-convert to grams where possible
            if k.endswith("_lb"):
                g = _unit(ureg(f"{v.split()[0]} lb")) * 1000 / ureg.kg  # to grams
                return f"{g:.2f} g"
            if k.endswith("_g"):
                return f"{float(v.split()[0]):.2f} g"
        if k.startswith("capacity_mah"):
            return f"{float(v):g} mAh"
    except Exception:
        pass
    return (v or "").strip()

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
    # feature → set(urls). "gemini" may appear as a sentinel.
    features: Dict[str, set] = field(default_factory=lambda: defaultdict(set))
    # url → attrs extracted from that url
    sources: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # specs introduced by Gemini (no direct URL)
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

# ────────────────── extraction helpers ──────────────────
_SHORTCUT_RE = re.compile(r"\b(shift|ctrl|control|alt|cmd|command|option)\b\s*(\+|plus)", re.I)
_BREADCRUMB_RE = re.compile(r"^\s*(home|orders|cart|search|account|sign in|sign out)\b", re.I)
_CATEGORY_SINGLETON_RE = re.compile(r"^\s*(chairs|camping|camping & hiking|outdoor recreation)\s*$", re.I)

def _is_junk_bullet(li: Any) -> bool:
    txt = li.get_text(" ", strip=True)
    if _SHORTCUT_RE.search(txt):
        return True
    if _BREADCRUMB_RE.search(txt):
        return True
    if _CATEGORY_SINGLETON_RE.match(txt):
        return True
    # parent classes/ids suggest navigation/menus
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
    # Amazon & general: feature/detail bullets
    for ul in soup.find_all(["ul", "ol"]):
        cls = " ".join(ul.get("class") or []) + " " + (ul.get("id") or "")
        if any(hint in cls.lower() for hint in FEATURE_HINTS):
            for li in ul.find_all("li"):
                if not _is_junk_bullet(li):
                    t = li.get_text(" ", strip=True)
                    if 20 <= len(t) <= 160:
                        bullets.append(t)
    # Fallback: consider first 40 list items, but filter junk
    if len(bullets) < 4:
        for li in soup.find_all("li")[:40]:
            if not _is_junk_bullet(li):
                t = li.get_text(" ", strip=True)
                if 20 <= len(t) <= 160:
                    bullets.append(t)

    # Fuzzy dedupe
    out: List[str] = []
    for b in bullets:
        if not any(fuzz.token_set_ratio(b, o) > 90 for o in out):
            out.append(b)
    return out[:12]

def _extract_from_amazon(html: str) -> Dict[str, str]:
    """
    Parse Amazon PDP tech-specs + detail bullets table sections.
    """
    soup = BeautifulSoup(html, "html.parser")
    attrs: Dict[str, str] = {}

    # 1) Technical Details table(s)
    for table_id in ["productDetails_techSpec_section_1", "productDetails_techSpec_section_2"]:
        table = soup.find("table", id=table_id)
        if not table:
            continue
        for row in table.find_all("tr"):
            th = row.find("th")
            td = row.find("td")
            if not th or not td:
                continue
            key = th.get_text(" ", strip=True).lower()
            val = td.get_text(" ", strip=True)
            if "item weight" in key and val:
                attrs["weight_lb"] = re.sub(r"[^\d.]", " ", val).strip().split()[0] + " lb"
            if ("maximum weight" in key or "weight limit" in key) and val:
                attrs["weight_capacity_lb"] = re.sub(r"[^\d.]", " ", val).strip().split()[0] + " lb"
            if "product dimensions" in key and val:
                dims = val.lower()
                if "cm" in dims:
                    m = re.search(r"(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)", dims)
                    if m: attrs["dimensions_cm"] = f"{m.group(1)} x {m.group(2)} x {m.group(3)}"
                else:
                    m = re.search(r"(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)", dims)
                    if m: attrs["dimensions_in"] = f"{m.group(1)} x {m.group(2)} x {m.group(3)} in"
            if "material" in key and val:
                attrs["material"] = val

    # 2) Generic regex pass over page text
    plain = soup.get_text(" ", strip=True)
    for k, pat in SPEC_RE:
        if k in attrs:
            continue
        m = pat.search(plain)
        if m:
            if k in ("weight_lb", "weight_g"):
                num = m.group(1); unit = m.group(2)
                attrs[k] = f"{num} {unit}"
            elif k in ("dimensions_in", "dimensions_cm"):
                attrs[k] = f"{m.group(1)} x {m.group(2)} x {m.group(3)}"
            else:
                attrs[k] = m.group(1)
    return attrs

def _extract_from_html(html: str, url: str) -> Tuple[Dict[str, str], List[str]]:
    attrs = dict(extract_structured_product(html, url).get("attributes", {}))
    host = _host(url)
    if "amazon." in host:
        # enrich with Amazon-specific parsing
        try:
            a = _extract_from_amazon(html)
            attrs.update(a)
        except Exception:
            pass

    # Generic regex sweep for additional specs
    plain = re.sub(r"<[^>]+>", " ", html)
    for k, pat in SPEC_RE:
        if k in attrs:
            continue
        m = pat.search(plain)
        if m:
            if k in ("weight_lb", "weight_g"):
                attrs[k] = f"{m.group(1)} {m.group(2)}"
            elif k in ("dimensions_in", "dimensions_cm"):
                attrs[k] = f"{m.group(1)} x {m.group(2)} x {m.group(3)}"
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
            if k in ("weight_lb", "weight_g"):
                attrs[k] = f"{m.group(1)} {m.group(2)}"
            elif k in ("dimensions_in", "dimensions_cm"):
                attrs[k] = f"{m.group(1)} x {m.group(2)} x {m.group(3)}"
            else:
                attrs[k] = m.group(1)
    return attrs

# ────────────────── scraper first ──────────────────
def get_product_record(brand: str, model: str, *, max_urls: int = 12) -> 'SpecRecord':
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
    vision_category_tags: List[str] | None = None,   # accepted & ignored
    max_results: int = 12,
) -> Dict[str, Any]:
    # 1) scraper
    rec = get_product_record(brand, product, max_urls=max_results)

    # 2) Gemini fallback
    if not rec.specs and not rec.features and not product_url_override:
        from gemini_fetcher import gemini_product_info
        data = gemini_product_info(brand, product)
        if data.get("status") == "OK":
            for k, v in data.get("specs", {}).items():
                nk = k.lower()
                rec.specs[nk] = _norm_val(nk, v)
                rec.specs_from_gemini.add(nk)
            for feat in data.get("features", []):
                # filter out junk again defensively
                if not _SHORTCUT_RE.search(feat) and not _BREADCRUMB_RE.search(feat) and not _CATEGORY_SINGLETON_RE.match(feat):
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
        # small bonus if one of the sources is Amazon or brand domain
        bonus = 0.0
        for u in urls_with_attr:
            h = _host(u)
            if "amazon." in h or any(h.endswith(d.strip(".")) for d in ["apple.", "dell.", "lenovo.", "rei.", "backcountry."]):
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
