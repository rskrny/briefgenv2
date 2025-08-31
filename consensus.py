# consensus.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter
import re

# Optional unit normalization
try:
    from pint import UnitRegistry
    _ureg = UnitRegistry()
except Exception:
    _ureg = None

NUMERIC_KEYS = {
    "weight", "capacity", "battery_life", "power", "screen", "ip_rating",
    "load_capacity", "seat_height"
}

DIMENSION_KEYS = {"dimensions", "packed_size"}

def _norm_number_text(val: str) -> str:
    # strip commas/spaces
    return re.sub(r"\s+", " ", (val or "")).strip()

def normalize_units(specs: Dict[str,str]) -> Dict[str,str]:
    """Normalize common units when pint is available; otherwise return as-is."""
    if not _ureg:
        return specs
    out = dict(specs)
    # weight
    if "weight" in out:
        m = re.search(r"([\d\.]+)\s*(kg|g|lb|lbs|pounds|oz)", out["weight"], re.I)
        if m:
            qty = float(m.group(1))
            unit = m.group(2).lower().replace("pounds","lb").replace("lbs","lb")
            try:
                q = (qty * _ureg(unit)).to(_ureg.kilogram)
                out["weight"] = f"{q.magnitude:.2f} kg"
            except Exception:
                pass
    # seat_height as cm
    if "seat_height" in out:
        m = re.search(r"([\d\.]+)\s*(cm|mm|in|\"|')", out["seat_height"], re.I)
        if m:
            qty = float(m.group(1)); unit = m.group(2).lower().replace("\"","in").replace("'","ft")
            try:
                q = (qty * _ureg(unit)).to(_ureg.cm)
                out["seat_height"] = f"{q.magnitude:.1f} cm"
            except Exception:
                pass
    # dimensions/packed_size — standardize separator to ×
    for key in ("dimensions","packed_size"):
        if key in out:
            out[key] = out[key].replace("x","×").replace("X","×")
    return out

def _conf_from_sources(count: int, manufacturer_present: bool) -> float:
    if manufacturer_present:
        return 0.92 if count >= 1 else 0.7
    # non-manufacturer: raise with more independent sources
    return min(0.9, 0.55 + 0.15 * max(0, count-1))

def consolidate_claims(raw_claims: List[Any], brand_token: str) -> Dict[str, Any]:
    """
    raw_claims: list of product_research.Claim
    Returns {"features":[...], "specs":[...]}
    """
    # bucket by key/value (normalized)
    spec_bucket: Dict[str, Dict[str, List[Any]]] = defaultdict(lambda: defaultdict(list))
    feat_bucket: Dict[str, List[Any]] = defaultdict(list)

    for c in raw_claims:
        if c.kind == "spec":
            k = c.key.lower().strip()
            v = _norm_number_text(c.value)
            spec_bucket[k][v].append(c)
        elif c.kind == "feature":
            v = c.value.strip()
            feat_bucket[v].append(c)

    # specs: choose manufacturer > majority (unit-normalized when possible)
    specs_out = []
    for key, valmap in spec_bucket.items():
        # Flatten to count sources and manufacturer presence
        best_v, best_score = None, -1.0
        for v, lst in valmap.items():
            manu = any(x.manufacturer for x in lst)
            conf = _conf_from_sources(len(lst), manu)
            # prefer manufacturer, else more sources, else higher page score sum
            tie = (1 if manu else 0, len(lst), sum(x.score for x in lst))
            score = conf * 10 + tie[0]*5 + tie[1]*0.2 + tie[2]*0.01
            if score > best_score:
                best_score = score
                best_v = (v, lst, manu, conf)
        if best_v:
            v, lst, manu, conf = best_v
            specs_out.append({
                "key": key,
                "value": v,
                "sources": [{"url": x.source, "manufacturer": x.manufacturer} for x in lst],
                "confidence": float(f"{conf:.3f}")
            })

    # features: accept if manufacturer or ≥2 sources
    feats_out = []
    for text, lst in feat_bucket.items():
        manu = any(x.manufacturer for x in lst)
        if manu or len(lst) >= 2:
            conf = _conf_from_sources(len(lst), manu)
            feats_out.append({
                "text": text,
                "sources": [{"url": x.source, "manufacturer": x.manufacturer} for x in lst],
                "confidence": float(f"{conf:.3f}")
            })

    # sort by confidence descending
    specs_out.sort(key=lambda x: x["confidence"], reverse=True)
    feats_out.sort(key=lambda x: x["confidence"], reverse=True)
    return {"specs": specs_out, "features": feats_out}
