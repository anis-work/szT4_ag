"""Validator module for CV Ranking Agent.

Validates ranking results and ensures schema compliance.
"""

import logging
from typing import List
from models import RankedResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_results(results: List[RankedResult]) -> List[RankedResult]:
    """Validate, deduplicate, and sanitize ranking results."""
    # Step 1: basic validation
    valid = []
    for result in results:
        if not (0 <= result.score <= 100):
            logger.warning(f"Score out of range for {result.candidate_name}: {result.score}")
            continue
        if not result.reason or not result.reason.strip():
            logger.warning(f"Empty reason for {result.candidate_name}")
            continue
        valid.append(result)

    # Step 2: deduplicate — keep highest score per candidate
    # Primary key: cv_id (if present), fallback: normalised name
    seen: dict = {}
    for result in valid:
        key = result.cv_id if result.cv_id else result.candidate_name.lower().strip()
        if key not in seen or result.score > seen[key].score:
            seen[key] = result

    deduped = list(seen.values())

    # Step 3: re-rank sequentially by score
    deduped.sort(key=lambda r: r.score, reverse=True)
    final = [
        r.model_copy(update={"rank": i + 1})
        for i, r in enumerate(deduped)
    ]

    logger.info(f"Validated {len(results)} → {len(valid)} valid → {len(final)} after dedup")
    return final
