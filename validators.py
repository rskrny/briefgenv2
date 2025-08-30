# validators.py
# Lightweight validators to catch obvious schema issues.

from typing import Dict, List, Any

def _num(x): 
    try: return float(x)
    except Exception: return None

def validate_analyzer_json(j: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    if "narrative" not in j or not isinstance(j["narrative"], list) or not j["narrative"]:
        errs.append("Missing or empty 'narrative' list.")
    if "scenes" not in j or not isinstance(j["scenes"], list) or not j["scenes"]:
        errs.append("Missing or empty 'scenes' list.")
    else:
        prev_end = None
        for s in j["scenes"]:
            st = _num(s.get("start_s")); en = _num(s.get("end_s"))
            if st is None or en is None or en <= st:
                errs.append(f"Scene {s.get('idx')} has invalid start/end.")
            if prev_end is not None and st is not None and st > prev_end + 0.01:
                errs.append(f"Gap before scene {s.get('idx')}.")
            prev_end = en
            txt = s.get("on_screen_text", [])
            if isinstance(txt, list) and len(txt) > 2:
                errs.append(f"Scene {s.get('idx')} has >2 on_screen_text lines.")
    return errs

def validate_script_json(j: Dict[str, Any], target_runtime_s: float) -> List[str]:
    errs: List[str] = []
    scr = j.get("script", {})
    scenes = scr.get("scenes", [])
    if not scenes:
        errs.append("Script scenes empty.")
        return errs
    # contiguous timing
    prev_end = 0.0
    for s in scenes:
        st = _num(s.get("start_s")); en = _num(s.get("end_s"))
        if st is None or en is None or en <= st:
            errs.append(f"Scene {s.get('idx')} invalid start/end.")
            continue
        if abs(st - prev_end) > 0.6:  # allow small drift
            errs.append(f"Non-contiguous timing at scene {s.get('idx')} (start {st}, prev_end {prev_end}).")
        prev_end = en
        txt = s.get("on_screen_text", [])
        if isinstance(txt, list) and len(txt) > 2:
            errs.append(f"Scene {s.get('idx')} has >2 on_screen_text lines.")
    # runtime check
    if abs(prev_end - float(target_runtime_s)) > 1.0:
        errs.append(f"Total runtime {prev_end:.2f}s differs from target {target_runtime_s:.2f}s by >1s.")
    return errs
