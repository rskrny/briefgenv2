# ─── confidence gate decorator ───
from functools import wraps
from typing_extensions import ParamSpec, Any, Callable
P = ParamSpec("P")

def require_confidence(min_conf: float = 0.75):
    """
    Decorator to halt the pipeline if SpecRecord.confidence is too low.
    Usage:
        @require_confidence(0.8)
        def step_after_research(spec: SpecRecord): ...
    """
    def deco(fn: Callable[P, Any]) -> Callable[P, Any]:
        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs):
            spec = kwargs.get("spec") or (args[0] if args else None)
            conf = getattr(spec, "confidence", 0.0)
            if conf < min_conf:
                raise ValueError(
                    f"Spec confidence {conf:.2f} below threshold {min_conf}. "
                    "Please double-check brand / model or provide a product URL."
                )
            return fn(*args, **kwargs)
        return wrapper
    return deco
