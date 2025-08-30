# prompts.py
# Prompt builders (Analyzer + Style-Transfer Script) with strict schemas

import json
from typing import Any, Dict, List, Optional

# ===== Analyzer schema (model output we expect) =====
ANALYZER_SCHEMA_EXAMPLE = {
    "video_metadata": {
        "platform": "tiktok|reels|ytshorts",
        "duration_s": 19.8,
        "aspect_ratio": "9:16",
    },
    "narrative": [
        {"phase": "hook",        "start_s": 0.0,  "end_s": 1.5,  "purpose": "pattern interrupt"},
        {"phase": "pain_point",  "start_s": 1.5,  "end_s": 4.0,  "purpose": "relatable problem"},
        {"phase": "solution",    "start_s": 4.0,  "end_s": 11.0, "purpose": "introduce product/approach"},
        {"phase": "proof",       "start_s": 11.0, "end_s": 16.0, "purpose": "demo/social proof/benefit"},
        {"phase": "cta",         "start_s": 16.0, "end_s": 19.8, "purpose": "clear action or loop"}
    ],
    "influencer_DNA": {
        "persona": "relatable, dry humor",
        "energy": "medium-high",
        "tone": "conversational, confident",
        "camera_presence": "handheld, faces camera",
        "edit_grammar": ["hard cuts", "caption pops", "sound effect hits"],
        "retention_devices": ["quick reveal", "jump cuts", "countdown timer"],
    },
    "beats": [
        {"t_s": 0.0,  "type": "cut"},
        {"t_s": 1.6,  "type": "caption_pop"},
        {"t_s": 4.2,  "type": "product_reveal"},
        {"t_s": 7.5,  "type": "whoosh_hit"},
        {"t_s": 16.0, "type": "cta_overlay"}
    ],
    "keyframes": [
        {"t_s": 0.0,  "image_ref": "kf_01.jpg", "why": "hook text burst"},
        {"t_s": 4.5,  "image_ref": "kf_02.jpg", "why": "macro product demo"},
        {"t_s": 16.0, "image_ref": "kf_03.jpg", "why": "CTA overlay"}
    ],
    "scenes": [
        {
            "idx": 1,
            "start_s": 0.0,
            "end_s": 3.8,
            "camera": "front camera, chest-up",
            "action": "creator addresses camera",
            "dialogue": "…",
            "on_screen_text": ["Line 1", "Line 2"],  # <= 2 short lines
            "transition_out": "hard cut",
            "sfx_or_music": "whoosh at 0.2s",
            "retention_note": "promise result quickly"
        }
    ],
    "compliance": {
        "sensitive_claims_detected": [],
        "notes": "no medical claims"
    },
    "transferable_patterns": [
        "open with pattern interrupt text burst",
        "macro close-up timed to sound hit",
        "CTA as overlay, not spoken"
    ]
}


def build_analyzer_messages(
    *,
    platform: str,
    duration_s: Optional[float],
    keyframes_meta: List[Dict[str, Any]],
    ocr_frames: List[Dict[str, Any]],
    transcript_text: Optional[str] = None,
    aspect_ratio: str = "9:16",
) -> str:
    """
    Build a single text prompt for Gemini that requests JSON only.
    We pass a compact EVIDENCE JSON and show the exact schema we want back.
    """
    evidence = {
        "platform": platform,
        "duration_s": duration_s,
        "aspect_ratio": aspect_ratio,
        "keyframes_meta": keyframes_meta,      # [{t, path, image_ref}]
        "ocr_frames": {"frames": ocr_frames},  # [{t, text[], image_path}]
        "transcript_text": transcript_text or "",
    }

    prompt = f"""
You are a director-level video analyst. Given the EVIDENCE below from a short-form video,
return JSON ONLY matching the schema that follows. Do not include extra keys or commentary.

EVIDENCE (JSON):
{json.dumps(evidence, ensure_ascii=False)}

DESIRED_OUTPUT_SCHEMA (JSON EXAMPLE):
{json.dumps(ANALYZER_SCHEMA_EXAMPLE, ensure_ascii=False, indent=2)}

Hard requirements:
- Include ALL phases: hook, pain_point, solution, proof, cta — covering the FULL runtime (0s → duration).
- Produce a beat grid ('beats'): cut/impact times you can infer.
- Include influencer_DNA (persona, energy, tone, camera_presence) + edit_grammar + retention_devices.
- 'scenes' MUST be contiguous, no gaps/overlaps, and cover 0s → duration. Each scene includes transition_out and sfx_or_music.
- 'on_screen_text' per scene: ≤ 2 short lines (7–10 words max/line), based on overlays or inferred intent.
- 'keyframes' must reference provided keyframe file names ('image_ref' is the basename from paths) and include a brief 'why'.
- Keep language concise. Output JSON ONLY.
"""
    return prompt


# ===== Script schema (model output we expect) =====
SCRIPT_SCHEMA_EXAMPLE = {
    "product": {"brand": "Siawag", "name": "BTW73"},
    "target_runtime_s": 20.0,
    "style_transfer": {
        "preserve": ["hook cadence", "caption pops", "CTA overlay"],
        "adapt": ["replace beauty macro with earbud case macro"],
        "affordance_map": [
            {"from": "lipstick swatch", "to": "tap earbud to change ANC mode"}
        ],
    },
    "script": {
        "opening_hook": "Quick line that matches the style and sets the promise.",
        "scenes": [
            {
                "idx": 1,
                "start_s": 0.0,
                "end_s": 3.5,
                "camera": "front camera, chest-up",
                "action": "Creator raises case, pops open",
                "dialogue": "You ever open your earbuds and…",
                "on_screen_text": ["Instant connect", "ENC calls"],  # <= 2 lines
                "transition_out": "hard cut",
                "sfx_or_music": "whoosh at 0.2s",
                "retention_note": "promise payoff by 5s"
            }
        ],
        "cta_options": [
            {"variant": "hard", "line": "Tap to shop the BTW73s."},
            {"variant": "loop", "line": "Watch how fast they connect →"}
        ],
    },
    "notes_for_legal": [
        "No superlatives or competitor comparisons.",
        "Keep claims to whitelist only."
    ],
    "checklist": [
        "Safe area: on-screen text max 2 lines, high contrast",
        "Match cuts to SFX beats"
    ]
}


def build_script_messages(
    *,
    analyzer_json: Dict[str, Any],
    brand: str,
    product: str,
    approved_claims: List[str],
    required_disclaimers: List[str],
    target_runtime_s: float,
    platform: str,
    brand_voice: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build a script-generation prompt that performs style transfer using Analyzer output.
    """
    voice = brand_voice or {}
    scaffold = {
        "brand": brand,
        "product": product,
        "approved_claims": approved_claims,
        "required_disclaimers": required_disclaimers,
        "brand_voice": voice,
        "target_runtime_s": target_runtime_s,
        "platform": platform,
    }

    prompt = f"""
You are a short-form video creative director. Using the ANALYZER JSON of a reference video,
produce a NEW script that transfers the style to a different product while respecting claim safety.

Return JSON ONLY matching the SCRIPT schema below. Do not include commentary.

CONSTRAINTS:
- Use ONLY the provided approved_claims. Do not invent claims.
- Scenes MUST be contiguous from 0s to target_runtime_s (±0.5s), with transition_out and sfx_or_music for every scene.
- on_screen_text per scene: ≤ 2 lines, short, 7–10 words per line, safe-area friendly.
- Keep tone consistent with influencer_DNA from the Analyzer, but adapt to brand_voice if given.
- Include 'style_transfer.affordance_map' to show how original actions transform to this product category.
- Include two CTA variants: 'hard' and 'loop'.

ANALYZER (JSON):
{json.dumps(analyzer_json, ensure_ascii=False)}

BRAND_INPUT (JSON):
{json.dumps(scaffold, ensure_ascii=False)}

DESIRED_OUTPUT_SCHEMA (JSON EXAMPLE):
{json.dumps(SCRIPT_SCHEMA_EXAMPLE, ensure_ascii=False, indent=2)}
"""
    return prompt
