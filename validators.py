# validators.py — v5.0.1  (2025-09-01)
"""
Lightweight JSON validators + confidence-gate decorator
"""

from __future__ import annotations
from typing import Dict, List, Any, Callable, TypeVar
from functools import wraps

# ─────────────────── util helpers ───────────────────
def _float(x) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


# ─────────────────── Analyzer JSON validator ───────────────────
def validate_analyzer_json(
    j: Dict[str, Any],
    target_runtime_s: float | None = None,       # ← now optional
) -> List[str]:
    """
    Returns list of issues; empty == OK.
    If target_runtime_s is None, runtime checks are skipped.
    """
    errs: List[str] = []
    if "video_metadata" not in j:
        errs.append("Missing video_metadata.")
    if "global_signals" not in j:
        errs.append("Missing global_signals.")
    if "scenes" not in j:
        errs.append("Missing scenes.")

    if target_runtime_s is not None:
        t = _float(j.get("video_metadata", {}).get("duration_s"))
        if t is None or t < 0:
            errs.append("Invalid duration_s.")
        elif abs(t - target_runtime_s) > 120:
            errs.append("duration_s far from target.")
    return errs


# ─────────────────── Script JSON validator ───────────────────
def validate_script_json(
    j: Dict[str, Any],
    target_runtime_s: float | None = None,       # ← also optional
) -> List[str]:
    errs: List[str] = []
    script = j.get("script")
    if not isinstance(script, dict):
        errs.append("Missing script dict.")
        return errs

    scenes = script.get("scenes", [])
    if not scenes:
        errs.append("No scenes in script.")
        return errs

    if target_runtime_s is not None:
        end_last = _float(scenes[-1].get("end_s"))
        if end_last is None or abs(end_last - target_runtime_s) > 3.0:
            errs.append("Last scene end_s not near target_runtime_s.")
    return errs


# ─────────────────── confidence-gate decorator ───────────────────
T = TypeVar("T")


def require_confidence(min_conf: float = 0.75):
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
