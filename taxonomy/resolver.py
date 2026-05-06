"""Sulzer role taxonomy resolver.

Matches a job title/description to the closest Sulzer role family and returns
implied skills that the JD may not have explicitly mentioned.
Zero API calls — pure local lookup.
"""

import json
import re
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_TAXONOMY_PATH = Path(__file__).parent / "sulzer_taxonomy.json"


@lru_cache(maxsize=1)
def _load() -> dict:
    """Load taxonomy JSON once and cache it for the lifetime of the process."""
    return json.loads(_TAXONOMY_PATH.read_text(encoding="utf-8"))


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).strip()


def match_role(job_title: str, job_description: str = "") -> Optional[dict]:
    """Match a job title + description to the best Sulzer role family.

    Returns the matched role family dict (with required_skills, implied_skills,
    certifications, experience_bands) or None if no confident match found.
    """
    taxonomy = _load()
    needle = _normalise(f"{job_title} {job_description[:300]}")

    best_match = None
    best_score = 0

    for division in taxonomy["divisions"].values():
        for family_key, family in division["role_families"].items():
            score = 0
            for title in family["titles"]:
                title_norm = _normalise(title)
                # Exact title match scores highest
                if title_norm in needle:
                    score += 10
                else:
                    # Partial word overlap
                    words = set(title_norm.split())
                    overlap = sum(1 for w in words if w in needle and len(w) > 3)
                    score += overlap

            if score > best_score:
                best_score = score
                best_match = {
                    "division": division["label"],
                    "family": family_key,
                    "required_skills": family["required_skills"],
                    "implied_skills": family["implied_skills"],
                    "soft_skills": family["soft_skills"],
                    "certifications": family["certifications"],
                    "experience_bands": family["experience_bands"],
                }

    if best_score >= 3:
        logger.info(f"Taxonomy match: '{job_title}' → {best_match['division']} / {best_match['family']} (score={best_score})")
        return best_match

    logger.info(f"No confident taxonomy match for '{job_title}' (best score={best_score})")
    return None


def build_enriched_jd(job_title: str, job_description: str) -> str:
    """Return the job description enriched with implied skills from taxonomy.

    Appends a clearly labelled section so Gemini knows these are implied
    skills from Sulzer's role taxonomy — not invented requirements.
    If no match found, returns the original JD unchanged.
    """
    match = match_role(job_title, job_description)
    if not match:
        return job_description

    # Only add implied skills not already mentioned in the JD
    jd_lower = job_description.lower()
    missing_implied = [
        s for s in match["implied_skills"]
        if _normalise(s) not in jd_lower
    ]
    missing_certs = [
        c for c in match["certifications"]
        if _normalise(c) not in jd_lower
    ]

    if not missing_implied and not missing_certs:
        return job_description

    enrichment_lines = [
        "\n\n--- SULZER ROLE TAXONOMY ENRICHMENT ---",
        f"Division: {match['division']}",
        "The following skills are IMPLIED by this role at Sulzer even if not explicitly stated above.",
        "Evaluate candidates for these as well, but weight them lower than explicitly stated requirements.",
    ]
    if missing_implied:
        enrichment_lines.append(f"Implied technical skills: {', '.join(missing_implied)}")
    if missing_certs:
        enrichment_lines.append(f"Typical certifications for this role: {', '.join(missing_certs)}")

    exp_bands = match["experience_bands"]
    enrichment_lines.append(
        f"Sulzer experience benchmarks — Junior: {exp_bands['junior']}y, "
        f"Mid: {exp_bands['mid']}y, Senior: {exp_bands['senior']}y"
    )
    enrichment_lines.append("--- END TAXONOMY ENRICHMENT ---")

    return job_description + "\n".join(enrichment_lines)
