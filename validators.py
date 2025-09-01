# validators.py — v5.0  (2025-09-01)
"""
• JSON-structure validators used by app.py
• Confidence-gate decorator added for the new product-info engine
"""

from __future__ import annotations
from typing import Dict, List, Any, Callable, TypeVar
from functools import wraps

# ─────────────────── generic helpers ───────────────────
def _float(x) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _non_empty(s) -> bool:
    return isinstance(s, str) and len(s.strip()) > 0


# ─────────────────── Analyzer JSON validator ───────────────────
def validate_analyzer_json(j: Dict[str, Any], target_runtime_s: float) -> List[str]:
    """
    Very lightweight placeholder (keeps app.py happy).
    Returns a list of error strings; empty == OK.
    TODO: re-implement full schema checks if needed.
    """
    errs: List[str] = []

    # Require top-level keys the app needs later
    if "video_metadata" not in j:
        errs.append("Missing video_metadata.")
    if "global_signals" not in j:
        errs.append("Missing global_signals.")
    if "scenes" not in j:
        errs.append("Missing scenes.")

    # Basic runtime sanity
    t = _float(j.get("video_metadata", {}).get("duration_s"))
    if t is None or t < 0:
        errs.append("Invalid duration_s.")
    elif abs(t - target_runtime_s) > 120:  # 2-min tolerance
        errs.append("duration_s far from target.")

    return errs


# ─────────────────── Script JSON validator ───────────────────
def validate_script_json(j: Dict[str, Any], target_runtime_s: float) -> List[str]:
    """
    Placeholder version that just checks presence of keys & timing.
    """
    errs: List[str] = []
    script = j.get("script")
    if not isinstance(script, dict):
        errs.append("Missing script dict.")
        return errs

    scenes = script.get("scenes", [])
    if not scenes:
        errs.append("No scenes in script.")
        return errs

    # Ensure final scene roughly matches target runtime
    end_last = _float(scenes[-1].get("end_s"))
    if end_last is None or abs(end_last - target_runtime_s) > 3.0:
        errs.append("Last scene end_s not near target_runtime_s.")

    return errs


# ─────────────────── confidence-gate decorator ───────────────────
T = TypeVar("T")


def require_confidence(min_conf: float = 0.75):
    """
    Decorator to stop the pipeline early if SpecRecord.confidence is too low.

        @require_confidence(0.8)
        def step_after_research(spec: SpecRecord): ...
    """

    def deco(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            spec = kwargs.get("spec") or (args[0] if args else None)
            conf = getattr(spec, "confidence", 0.0)
            if conf < min_conf:
                raise ValueError(
                    f"Spec confidence {conf:.2f} below threshold {min_conf}. "
                    "Double-check brand/model or supply a product URL."
                )
            return fn(*args, **kwargs)

        return wrapper

    return deco
