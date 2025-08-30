# validators.py
# Lightweight validators to enforce structure + timing + content constraints.

from typing import Dict, List, Any

REQUIRED_PHASES = ["hook", "pain_point", "solution", "proof", "cta"]

def _f(x):
    try: return float(x)
    except Exception: return None

def validate_analyzer_json(j: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    # Narrative phases
    phases = [p.get("phase") for p in j.get("narrative", []) if isinstance(p, dict)]
    if not phases:
        errs.append("Missing 'narrative'.")
    else:
        for req in REQUIRED_PHASES:
            if req not in phases:
                errs.append(f"Missing narrative phase: {req}.")
    # Scenes
    scenes = j.get("scenes", [])
    if not scenes:
        errs.append("Missing 'scenes'.")
    else:
        prev_end = 0.0
        start_seen = False
        for s in scenes:
            st = _f(s.get("start_s")); en = _f(s.get("end_s"))
            if st is None or en is None or en <= st:
                errs.append(f"Scene {s.get('idx')} invalid start/end.")
                continue
            if not start_seen and abs(st - 0.0) > 0.6:
                errs.append("Scenes do not start at 0.0s.")
            start_seen = True
            if abs(st - prev_end) > 0.6:
                errs.append(f"Non-contiguous timing at scene {s.get('idx')} (start {st}, prev_end {prev_end}).")
            prev_end = en
            # Must have transition and sfx
            if not s.get("transition_out"):
                errs.append(f"Scene {s.get('idx')} missing transition_out.")
            if not s.get("sfx_or_music"):
                errs.append(f"Scene {s.get('idx')} missing sfx_or_music.")
            # OSD max 2
            txt = s.get("on_screen_text", [])
            if isinstance(txt, list) and len(txt) > 2:
                errs.append(f"Scene {s.get('idx')} has >2 on_screen_text lines.")
    # Keyframes
    kfs = j.get("keyframes", [])
    if not kfs:
        errs.append("Missing 'keyframes'.")
    else:
        for k in kfs:
            if not k.get("image_ref"):
                errs.append("Keyframe missing image_ref.")
            if not k.get("why"):
                errs.append("Keyframe missing 'why'.")
    # Influencer DNA / edit grammar
    dna = j.get("influencer_DNA", {})
    if not dna or not dna.get("edit_grammar") or not dna.get("retention_devices"):
        errs.append("Missing influencer_DNA/edit_grammar/retention_devices.")
    return errs

def validate_script_json(j: Dict[str, Any], target_runtime_s: float) -> List[str]:
    errs: List[str] = []
    scr = j.get("script", {})
    scenes = scr.get("scenes", [])
    if not scenes:
        errs.append("Script scenes empty.")
        return errs
    # contiguous timing + runtime envelope
    prev_end = 0.0
    for s in scenes:
        st = _f(s.get("start_s")); en = _f(s.get("end_s"))
        if st is None or en is None or en <= st:
            errs.append(f"Scene {s.get('idx')} invalid start/end.")
            continue
        if abs(st - prev_end) > 0.6:  # small drift allowed
            errs.append(f"Non-contiguous timing at scene {s.get('idx')} (start {st}, prev_end {prev_end}).")
        prev_end = en
        # Require transition + sfx
        if not s.get("transition_out"):
            errs.append(f"Scene {s.get('idx')} missing transition_out.")
        if not s.get("sfx_or_music"):
            errs.append(f"Scene {s.get('idx')} missing sfx_or_music.")
        # OSD max 2
        txt = s.get("on_screen_text", [])
        if isinstance(txt, list) and len(txt) > 2:
            errs.append(f"Scene {s.get('idx')} has >2 on_screen_text lines.")
    if abs(prev_end - float(target_runtime_s)) > 1.0:
        errs.append(f"Total runtime {prev_end:.2f}s differs from target {target_runtime_s:.2f}s by >1s.")
    # Style-transfer mapping presence
    st_map = j.get("style_transfer", {}).get("affordance_map", [])
    if not st_map:
        errs.append("Missing style_transfer.affordance_map.")
    return errs
