# exemplars.py
# Lightweight few-shot anchors (you can add real ones later).
from typing import Dict, Any, List

# Each exemplar has: name, archetype, feature_vector, mini_analysis (phases abbreviated)
# feature_vector = [speech_ratio, cut_density, motion_mean, ocr_density]
EXEMPLARS: List[Dict[str, Any]] = [
    {
        "name": "ASMR Unbox – Gadget",
        "archetype": "SHOWCASE",
        "feature_vector": [0.02, 0.20, 0.18, 0.05],
        "mini_analysis": {
            "phases": [
                {"name":"Unbox","what_happens":"open box, remove unit"},
                {"name":"Handle/Features","what_happens":"macro CU features"},
                {"name":"Demo","what_happens":"simple usage"},
                {"name":"Outro","what_happens":"hold to camera"}
            ]
        }
    },
    {
        "name": "Hands-only Beauty ASMR",
        "archetype": "SHOWCASE",
        "feature_vector": [0.00, 0.32, 0.22, 0.10],
        "mini_analysis": {"phases":[
            {"name":"Handle/Features","what_happens":"texture, applicator"},
            {"name":"Demo","what_happens":"swatch/application"}
        ]}
    },
    {
        "name": "Talky Review – Tech",
        "archetype": "NARRATIVE",
        "feature_vector": [0.80, 0.28, 0.25, 0.15],
        "mini_analysis": {"phases":[
            {"name":"Hook","what_happens":"opinion claim"},
            {"name":"Solution","what_happens":"why it’s good"},
            {"name":"Proof","what_happens":"demo"},
            {"name":"CTA","what_happens":"short ask"}
        ]}
    },
    {
        "name": "Tutorial – 3 Steps",
        "archetype": "TUTORIAL",
        "feature_vector": [0.60, 0.35, 0.20, 0.25],
        "mini_analysis": {"phases":[
            {"name":"Step 1"},{"name":"Step 2"},{"name":"Step 3"},{"name":"Result"}
        ]}
    },
    {
        "name": "Comparison – Side by Side",
        "archetype": "COMPARISON",
        "feature_vector": [0.55, 0.30, 0.30, 0.12],
        "mini_analysis": {"phases":[
            {"name":"Comparison Setup"},{"name":"Side-by-side Test"},{"name":"Result"}
        ]}
    },
    {
        "name": "Field Test – Outdoor",
        "archetype": "TEST_DEMO",
        "feature_vector": [0.25, 0.22, 0.40, 0.05],
        "mini_analysis": {"phases":[
            {"name":"Setup"},{"name":"Test"},{"name":"Results"},{"name":"Takeaway"}
        ]}
    }
]

def cosine(a, b):
    import math
    da = math.sqrt(sum(x*x for x in a)) or 1e-9
    db = math.sqrt(sum(x*x for x in b)) or 1e-9
    return sum(x*y for x,y in zip(a,b)) / (da*db)

def top_k_exemplars(vec, k=2):
    scored = [(cosine(vec, e["feature_vector"]), e) for e in EXEMPLARS]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:k]]
