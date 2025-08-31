# archetypes.py
from typing import Dict, List

ARCHETYPES: List[str] = [
    "SHOWCASE", "NARRATIVE", "TUTORIAL", "COMPARISON",
    "TEST_DEMO", "TESTIMONIAL_UGC", "TIMELAPSE", "ANNOUNCEMENT"
]

# Allowed/typical phases per archetype (not forced; just a grammar menu)
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
