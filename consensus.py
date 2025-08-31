# consensus.py
from __future__ import annotations
from typing import List, Dict, Any, Callable, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter
import re

try:
    from pint import UnitRegistry  # pip install pint
    UREG = UnitRegistry()
except Exception:
    UREG = None

@dataclass
class Claim:
    key: str
    value: str
    source: str
    snippet: str
    kind: str  # "spec" | "feature" | "disclaimer"

NUMERIC_KEYS = {
    "weight","capacity","battery_life","power","screen","ip_rating",
    "dimensions","load_capacity","packed_size","seat_height",
}

# Normalize strings (for features)
def _norm_phrase(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\s×x\.\-:/]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def _is_numeric_like(key: str) -> bool:
    return key in NUMERIC_KEYS

def _normalize_units(key: str, value: str) -> str:
    if not UREG:
        return value
    try:
        v = value
        # Very light normalization cases:
        if key in ("weight","load_capacity"):
            # extract number + unit
            m = re.search(r"([\d\.]+)\s*(kg|g|lbs?|pounds|oz)", value.lower())
            if not m:
                return value
            qty = float(m.group(1))
            unit = m.group(2)
            q = qty * UREG(unit)
            # standardize to kg (one decimal)
            kg = q.to(UREG.kg).magnitude
            return f"{kg:.2f} kg"
        if key in ("dimensions","packed_size","screen","seat_height"):
            # Keep as-is; dimensions often "X × Y × Z in/cm"
            v = value.replace("x", "×")
            return v
        if key in ("capacity","battery_life","power","ip_rating","materials"):
            return value
        return value
    except Exception:
        return value

def _pick_numeric(claims: List[Claim], brand: str, is_manufacturer) -> Tuple[str, float, List[Dict[str,str]]]:
    # prefer manufacturer; else majority (normalized)
    by_norm: Dict[str, List[Claim]] = defaultdict(list)
    for c in claims:
        norm = _normalize_units(c.key, c.value)
        by_norm[norm].append(c)
    # manufacturer boost
    def score_group(vals: List[Claim]) -> float:
        score = len(vals)
        if any(is_manufacturer(v.source, brand) for v in vals):
            score += 2.5
        return score
    best_norm = max(by_norm.keys(), key=lambda k: score_group(by_norm[k]))
    chosen = by_norm[best_norm]
    conf = min(0.95, 0.45 + 0.15 * len(chosen) + (0.25 if any(is_manufacturer(v.source, brand) for v in chosen) else 0))
    sources = [{"url": c.source, "snippet": c.snippet} for c in chosen]
    return best_norm, conf, sources

def consolidate_claims(
    claims: List[Claim],
    brand: str,
    min_confidence: float = 0.6,
    is_manufacturer: Callable[[str, str], bool] = lambda url, brand: False,
) -> Dict[str, Any]:
    by_key: Dict[str, List[Claim]] = defaultdict(list)
    feats: List[Dict[str, Any]] = []
    discls: List[Dict[str, Any]] = []
    notes: List[str] = []

    for c in claims:
        by_key[c.key].append(c)

    specs_out: List[Dict[str, Any]] = []

    # Specs (numeric-like keys)
    for key, lst in by_key.items():
        if key == "feature" or key == "disclaimer":
            continue
        if _is_numeric_like(key):
            val, conf, srcs = _pick_numeric(lst, brand, is_manufacturer)
            if conf >= min_confidence:
                specs_out.append({"key": key, "value": val, "sources": srcs, "confidence": conf})
        else:
            # non-numeric spec (e.g., materials string)
            # group by normalized strings
            bucket: Dict[str, List[Claim]] = defaultdict(list)
            for c in lst:
                bucket[_norm_phrase(c.value)].append(c)
            best = max(bucket.keys(), key=lambda k: len(bucket[k]))
            chosen = bucket[best]
            conf = min(0.9, 0.4 + 0.2 * len(chosen) + (0.25 if any(is_manufacturer(v.source, brand) for v in chosen) else 0))
            if conf >= min_confidence:
                specs_out.append({"key": key, "value": chosen[0].value, "sources": [{"url": x.source, "snippet": x.snippet} for x in chosen], "confidence": conf})

    # Features
    feat_vals: Dict[str, List[Claim]] = defaultdict(list)
    for c in by_key.get("feature", []):
        feat_vals[_norm_phrase(c.value)].append(c)
    for val_norm, lst in feat_vals.items():
        conf = min(0.9, 0.35 + 0.15 * len(lst) + (0.25 if any(is_manufacturer(v.source, brand) for v in lst) else 0))
        if conf >= min_confidence or len(lst) >= 2:
            feats.append({"text": lst[0].value, "sources": [{"url": x.source, "snippet": x.snippet} for x in lst], "confidence": conf})

    # Disclaimers
    disc_vals: Dict[str, List[Claim]] = defaultdict(list)
    for c in by_key.get("disclaimer", []):
        disc_vals[_norm_phrase(c.value)].append(c)
    for _, lst in disc_vals.items():
        conf = min(0.9, 0.35 + 0.15 * len(lst) + (0.25 if any(is_manufacturer(v.source, brand) for v in lst) else 0))
        if conf >= min_confidence or len(lst) >= 2:
            discls.append({"text": lst[0].value, "sources": [{"url": x.source, "snippet": x.snippet} for x in lst], "confidence": conf})

    # Keep some raw short claims for debugging (not used to script)
    claims_out = []
    for c in claims[:30]:
        claims_out.append({"key": c.key, "value": c.value, "source": c.source})

    return {
        "specs": specs_out,
        "features": feats[:20],
        "disclaimers": discls[:10],
        "claims": claims_out,
        "notes": notes,
    }
