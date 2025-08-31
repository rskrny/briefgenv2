# prompts.py
# Prompt builders (Analyzer + Script) with archetype-aware schemas.

import json
from typing import Any, Dict, List, Optional

# ===== Archetype-aware Analyzer schema example =====
ANALYZER_SCHEMA_EXAMPLE = {
    "video_metadata": {
        "platform": "tiktok|reels|ytshorts",
        "duration_s": 19.8,
        "aspect_ratio": "9:16",
    },
    "global_signals": {
        "speech_presence": "none|low|medium|high",
        "music_presence": True,
        "tempo": "calm|moderate|fast",
        "setting": "indoor|outdoor|store|desk|campsite|unknown"
    },
    "archetype": "SHOWCASE|NARRATIVE|TUTORIAL|COMPARISON|TEST_DEMO|TESTIMONIAL_UGC|TIMELAPSE|ANNOUNCEMENT",
    "confidence": 0.86,

    # Lightweight phase list that depends on archetype.
    # SHOWCASE examples: Unbox, Handle/Features, Demo, Outro
    # NARRATIVE examples: Hook, Problem, Solution, Proof, CTA
    "phases": [
        {
            "phase": "Unbox|Handle/Features|Demo|Outro|Hook|Problem|Solution|Proof|CTA|Step 1|Step 2|Comparison Setup|Test|Result",
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

    "visible_product_features": ["folding frame","quick-fold latch","cup holder","carry bag"],

    "compliance": {
        "sensitive_claims_detected": [],
        "notes": "no medical claims; no certifications"
    },

    "influencer_DNA": {
        "persona": "hands-only; no dialogue",
        "pacing": "calm",
        "visual_motifs": ["macro texture", "slow pans"],
        "edit_habits": ["hard cuts", "minimal captions"]
    }
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
    We pass compact EVIDENCE and show the exact schema we want back.
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
You are a director-level video analyst for short-form content (TikTok/Reels/Shorts).
Given EVIDENCE from a reference video, return JSON ONLY matching the schema below. Do not include any extra keys or commentary.

EVIDENCE (JSON):
{json.dumps(evidence, ensure_ascii=False)}

INSTRUCTIONS:
- First, infer speech presence:
  - If transcript_text is empty or only a few words and OCR shows few/no captions, set "speech_presence" = "none" or "low".
- Choose exactly one "archetype" from:
  SHOWCASE, NARRATIVE, TUTORIAL, COMPARISON, TEST_DEMO, TESTIMONIAL_UGC, TIMELAPSE, ANNOUNCEMENT.
  *SHOWCASE* is common for ASMR/unbox/feature demos with little/no speech.
- Output only the phases that truly exist for the detected archetype:
  - SHOWCASE: use Unbox / Handle/Features / Demo / Outro if present. Do NOT invent Problem/Solution/CTA unless clear evidence exists.
  - NARRATIVE: Hook / Problem / Solution / Proof / CTA if supported by speech or visible captions.
  - TUTORIAL: Step 1..N with brief actions.
  - COMPARISON: Comparison Setup / Side-by-side Test / Result.
- Every phase MUST include an "evidence" array that references items from the EVIDENCE (e.g., "kf@9.2", "ocr@'Quick-fold latch'"). If unsure, leave "evidence" empty but prefer omission over hallucination.
- Keep times reasonable and non-overlapping; omit times if unknown.
- Only list "visible_product_features" that can plausibly be seen in keyframes/visuals.
- Include a single "influencer_DNA" block with persona/pacing/motifs/edit_habits inferred from visuals.

Return JSON ONLY with this top-level structure:
{json.dumps(ANALYZER_SCHEMA_EXAMPLE, ensure_ascii=False, indent=2)}
"""
    return prompt


# ===== Script schema example (unchanged structure, works for both modes) =====
SCRIPT_SCHEMA_EXAMPLE = {
    "product": {"brand": "Brand", "name": "Product"},
    "target_runtime_s": 20.0,
    "style_transfer": {
        "preserve": ["cadence", "visual motifs"],
        "adapt": ["map actions to this product"],
        "affordance_map": [{"from": "orig action", "to": "new product action"}],
    },
    "script": {
        # Keep opening optional for showcase; model may leave it short/empty
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
    """
    Build a script-generation prompt that performs style transfer using Analyzer output.
    Archetype-aware: If archetype=SHOWCASE or speech_presence is none/low,
    do NOT force Problem/Solution; keep VO optional and minimal.
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
"""
    return prompt
