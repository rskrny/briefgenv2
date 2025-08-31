# validators.py
# Archetype-aware validators to enforce structure without forcing narrative phases.

from typing import Dict, List, Any, Optional

def _f(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def _non_empty_str(s) -> bool:
    return isinstance(s, str) and len(s.strip()) > 0

# -------------------------------
# Analyzer JSON validator
# -------------------------------
def validate_analyzer_json(j: Dict[str, Any]) -> List[str]:
    errs: List[str] = []

    # Basics
    vm = j.get("video_metadata", {})
    if not isinstance(vm, dict):
        errs.append("Missing 'video_metadata'.")
    else:
        dur = _f(vm.get("duration_s"))
        if dur is None or dur < 0:
            errs.append("Invalid 'video_metadata.duration_s'.")
        ar = vm.get("aspect_ratio")
        if not _non_empty_str(ar):
            errs.append("Missing 'video_metadata.aspect_ratio'.")

    gs = j.get("global_signals", {})
    if not isinstance(gs, dict):
        errs.append("Missing 'global_signals'.")
    else:
        sp = gs.get("speech_presence")
        if sp not in {"none", "low", "medium", "high"}:
            errs.append("global_signals.speech_presence must be one of none|low|medium|high.")

    arch = j.get("archetype")
    if arch not in {
        "SHOWCASE","NARRATIVE","TUTORIAL","COMPARISON","TEST_DEMO","TESTIMONIAL_UGC","TIMELAPSE","ANNOUNCEMENT"
    }:
        errs.append("Invalid or missing 'archetype'.")

    conf = j.get("confidence")
    if conf is None or not (0.0 <= float(conf) <= 1.0):
        errs.append("Missing/invalid 'confidence' in [0,1].")

    # Phases
    phases = j.get("phases", [])
    if not isinstance(phases, list):
        errs.append("Missing 'phases' (list).")
        phases = []

    # Archetype-specific checks (no forced narrative for showcase)
    if arch == "SHOWCASE":
        # Allow empty if the video is too short, but warn if totally empty
        if not phases:
            errs.append("SHOWCASE: no phases found (expected Unbox/Handle/Features/Demo/Outro if present).")
    else:
        if not phases:
            errs.append(f"{arch}: 'phases' should not be empty.")

    # Phase time sanity & ordering
    prev_end = -1.0
    for i, p in enumerate(phases, start=1):
        st = _f(p.get("start_s"))
        en = _f(p.get("end_s"))
        if st is not None and en is not None:
            if en <= st:
                errs.append(f"Phase {i} has invalid start/end (end <= start).")
            if prev_end >= 0 and st < prev_end - 0.25:
                errs.append(f"Phase {i} starts before previous phase ends (overlap).")
            prev_end = en if en is not None else prev_end

    # Visible features (optional but should be list if present)
    vf = j.get("visible_product_features", [])
    if vf and not isinstance(vf, list):
        errs.append("'visible_product_features' must be a list.")

    return errs


# -------------------------------
# Script JSON validator
# -------------------------------
def validate_script_json(j: Dict[str, Any], target_runtime_s: float) -> List[str]:
    errs: List[str] = []

    script = j.get("script")
    if not isinstance(script, dict):
        errs.append("Missing 'script' object.")
        return errs

    scenes = script.get("scenes", [])
    if not isinstance(scenes, list) or not scenes:
        errs.append("Missing 'script.scenes' (non-empty list).")
        return errs

    # Timing sanity and approximate coverage
    first = scenes[0]
    st0 = _f(first.get("start_s"))
    if st0 is None or st0 > 0.75:
        errs.append("First scene should start near 0.0s (≤0.75s).")

    prev_end = None
    for i, s in enumerate(scenes, start=1):
        st = _f(s.get("start_s")); en = _f(s.get("end_s"))
        if st is None or en is None or en <= st:
            errs.append(f"Scene {i} invalid start/end.")
        if prev_end is not None and (st - prev_end) > 1.0:
            errs.append(f"Scene {i} has a large gap from previous scene (>1s).")
        prev_end = en

        # Basic content checks
        if not _non_empty_str(s.get("action", "")):
            errs.append(f"Scene {i} missing 'action'.")
        # OSD ≤ 2 lines if present
        osd = s.get("on_screen_text", [])
        if osd and len(osd) > 2:
            errs.append(f"Scene {i} has more than 2 on_screen_text lines.")

    # End near target
    end_last = _f(scenes[-1].get("end_s"))
    if end_last is not None and abs(end_last - float(target_runtime_s)) > 2.0:
        errs.append("Last scene end_s should be close to target_runtime_s (±2s).")

    return errs
