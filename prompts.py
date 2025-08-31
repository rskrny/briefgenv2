# prompts.py
# Prompt builders (Analyzer + Script) with archetype-aware schemas and JSON-only responses.

from __future__ import annotations
import json
from typing import Any, Dict, List, Optional

# =============================================================================
# Analyzer: schema example (what we expect the model to return)
# =============================================================================

ANALYZER_SCHEMA_EXAMPLE: Dict[str, Any] = {
    "video_metadata": {
        "platform": "tiktok|reels|ytshorts",
        "duration_s": 19.8,
        "aspect_ratio": "9:16",
    },
    "global_signals": {
        "speech_presence": "none|low|medium|high",
        "music_presence": False,
        "tempo": "calm|moderate|fast",
        "setting": "indoor|outdoor|store|desk|campsite|unknown"
    },
    "archetype": "SHOWCASE|NARRATIVE|TUTORIAL|COMPARISON|TEST_DEMO|TESTIMONIAL_UGC|TIMELAPSE|ANNOUNCEMENT",
    "confidence": 0.86,

    # Phases are dependent on archetype and should only include what truly exists.
    "phases": [
        {
            "phase": "Unbox|Handle/Features|Demo|Outro|Hook|Problem|Solution|Proof|CTA|Step 1|Step 2|Comparison Setup|Side-by-side Test|Result|Setup|Test|Takeaway|Context|Key Info|How to Act|Reveal|Progress",
            "start_s": 0.0,
            "end_s": 4.0,
            "what_happens": "concise, visible action",
            "camera_notes": "CU/MS/WS; moves if obvious",
            "audio_notes": "speech|silent|ambient|music",
            "on_screen_text": "short label if present else ''",
            "evidence": ["kf@0.0", "ocr@'Quick-fold latch'"]
        }
    ],

    "keyframes": [
        {"t_s": 0.0,  "image_ref": "kf_01.jpg", "why": "unbox moment"},
        {"t_s": 9.2,  "image_ref": "kf_02.jpg", "why": "latch close-up"},
        {"t_s": 16.0, "image_ref": "kf_03.jpg", "why": "stability demo"}
    ],

    "visible_product_features": [],

    "compliance": {
        "sensitive_claims_detected": [],
        "notes": ""
    },

    "influencer_DNA": {
        "persona": "",
        "pacing": "",
        "visual_motifs": [],
        "edit_habits": []
    }
}

# Default archetype menu + grammar (allowed phases for each)
ARCHETYPES: List[str] = [
    "SHOWCASE", "NARRATIVE", "TUTORIAL", "COMPARISON",
    "TEST_DEMO", "TESTIMONIAL_UGC", "TIMELAPSE", "ANNOUNCEMENT"
]

ARCHETYPE_PHASES: Dict[str, List[str]] = {
    "SHOWCASE": ["Unbox", "Handle/Features", "Demo", "Outro"],
    "NARRATIVE": ["Hook", "Problem", "Solution", "Proof", "CTA"],
    "TUTORIAL": ["Hook", "Step 1", "Step 2", "Step 3", "Result", "CTA"],
    "COMPARISON": ["Hook", "Comparison Setup", "Side-by-side Test", "Result", "CTA"],
    "TEST_DEMO": ["Setup", "Test", "Results", "Takeaway", "Outro"],
    "TESTIMONIAL_UGC": ["Problem", "Experience", "Benefit", "CTA"],
    "TIMELAPSE": ["Setup", "Progress", "Reveal", "Outro"],
    "ANNOUNCEMENT": ["Context", "Key Info", "How to Act", "CTA"],
}

def build_analyzer_prompt_with_fewshots(
    *,
    platform: str,
    duration_s: Optional[float],
    aspect_ratio: str,
    keyframes_meta: List[Dict[str, Any]],
    ocr_frames: List[Dict[str, Any]],
    transcript_text: str,
    archetype_menu: List[str] = ARCHETYPES,
    grammar_table: Dict[str, List[str]] = ARCHETYPE_PHASES,
    fewshots: List[Dict[str, Any]] = [],
    schema_example: Dict[str, Any] = ANALYZER_SCHEMA_EXAMPLE,
    scene_candidates: List[Dict[str, Any]] = None,
) -> str:
    """
    Build a *single* JSON-only prompt for the analyzer.
    - fewshots: a short list of exemplars (name, archetype, mini_analysis)
    - keyframes_meta: [{"t": 9.2, "ref": "kf@9.2"}, ...]
    - ocr_frames: [{"t": 9.2, "lines": ["...","..."]}, ...]
    - scene_candidates: [{"start_s":..., "end_s":..., "why":"motion jump"}, ...]
    """
    payload = {
        "platform": platform,
        "duration_s": duration_s,
        "aspect_ratio": aspect_ratio,
        "keyframes_meta": keyframes_meta,
        "ocr_frames": ocr_frames,
        "transcript_text": transcript_text or "",
        "archetype_menu": archetype_menu,
        "grammar": grammar_table,
        "fewshots": [
            {
                "name": fs.get("name",""),
                "archetype": fs.get("archetype",""),
                "mini_analysis": fs.get("mini_analysis", {}),
            } for fs in (fewshots or [])
        ],
        "scene_candidates": scene_candidates or []
    }

    return f"""
You are a director-level analyst for short-form video (TikTok/Reels/Shorts).
Return JSON ONLY that matches the provided schema exactly. Do not include commentary.

GUIDANCE:
- Infer 'speech_presence' from transcript length + OCR caption density.
- Choose exactly one 'archetype' from 'archetype_menu'.
- Only include phases valid for that archetype (see 'grammar').
- Use 'scene_candidates' as coarse cut suggestions. ALIGN your phase start/end to these when reasonable.
- SHOWCASE segmentation guidance (no speech):
  * If duration ≥ 10s, prefer 2–4 phases (e.g., Unbox → Handle/Features → Demo → Outro) rather than one giant phase.
  * Only output a single phase if the video truly has no discernible internal beats; if you do, lower 'confidence' to ≤ 0.6.
- For each phase, include specific 'evidence' refs (e.g., "kf@9.2", "ocr@'Quick-fold latch'"). Prefer omission over hallucination.

EVIDENCE (JSON):
{json.dumps(payload, ensure_ascii=False)}

SCHEMA (JSON):
{json.dumps(schema_example, ensure_ascii=False, indent=2)}
""".strip()

# =============================================================================
# Script generator: schema + builder
# =============================================================================

SCRIPT_SCHEMA_EXAMPLE: Dict[str, Any] = {
    "product": {"brand": "Brand", "name": "Product"},
    "target_runtime_s": 20.0,
    "style_transfer": {
        "preserve": ["cadence", "visual motifs"],
        "adapt": ["map actions to this product"],
        "affordance_map": [{"from": "orig action", "to": "new product action"}],
    },
    "script": {
        "opening_hook": "Optional quick line or empty for showcase.",
        "scenes": [
            {
                "idx": 1,
                "start_s": 0.0,
                "end_s": 3.5,
                "camera": "WS/MS/CU; movement if any",
                "action": "what we film",
                "on_screen_text": ["≤2 short lines"],
                "voiceover": "'' if none",
                "sfx_or_music": "if any",
                "transition_out": "hard cut|crossfade"
            }
        ]
    },
    "ctas": {
        "hard": "Check our TikTok Shop for details.",
        "loop": "See part 2 in my profile."
    },
    "checklist": [
        "Safe area: 2 short OSD lines max; high contrast",
        "Keep beats clear; no hype words if brand voice forbids"
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

    return f"""
You are a short-form video creative director. Using the ANALYZER JSON of the reference video,
produce a NEW script that transfers the style to a different product while respecting claim safety.

Return JSON ONLY matching the SCRIPT schema below. Do not include commentary.

ARCHETYPE LOGIC:
- If ANALYZER.archetype is SHOWCASE or global_signals.speech_presence is none/low:
  - Do NOT invent pain points or solutions.
  - Voiceover is optional (0–2 very short lines total); it’s okay to leave VO empty.
  - Focus on visual actions that mirror the reference phases (Unbox → Handle/Features → Demo → Outro).
- If NARRATIVE (or Tutorial/Comparison): Use phases appropriate to the analysis (Hook/Problem/Solution/Proof/CTA OR Steps/Tests),
  but only when the analyzer phases indicate such beats (or speech/captions support them).

HARD RULES:
- Use ONLY the provided approved_claims. Do not invent claims, certifications, or metrics.
- Scenes should roughly cover 0s→target_runtime_s with minimal gaps; keep times sensible (±0.5s tolerance ok).
- on_screen_text per scene ≤ 2 lines; 7–10 words each; safe-area friendly.
- Add 'style_transfer.affordance_map' to show how original actions translate to this product.

ANALYZER (JSON):
{json.dumps(analyzer_json, ensure_ascii=False)}

BRAND_INPUT (JSON):
{json.dumps(scaffold, ensure_ascii=False)}

DESIRED_OUTPUT_SCHEMA (JSON EXAMPLE):
{json.dumps(SCRIPT_SCHEMA_EXAMPLE, ensure_ascii=False, indent=2)}
""".strip()
