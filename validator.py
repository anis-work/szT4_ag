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
    """Validate and sanitize ranking results.
    
    Checks:
    - All ranks are unique and sequential
    - All scores are between 0 and 100
    - All reasons are non-empty
    
    Args:
        results: List of RankedResult objects to validate.
    
    Returns:
        List of valid RankedResult objects (invalid entries removed).
    """
    valid_results = []
    ranks_seen = set()
    
    for result in results:
        errors = []
        
        # Validate rank uniqueness
        if result.rank in ranks_seen:
            errors.append(f"Duplicate rank {result.rank}")
        ranks_seen.add(result.rank)
        
        # Validate score range
        if not (0 <= result.score <= 100):
            errors.append(f"Score {result.score} out of range [0-100]")
        
        # Validate reason is not empty
        if not result.reason or not result.reason.strip():
            errors.append("Reason is empty")
        
        if errors:
            logger.warning(
                f"Invalid result for {result.candidate_name}: {'; '.join(errors)}"
            )
        else:
            valid_results.append(result)
    
    # Log summary
    logger.info(
        f"Validated {len(results)} results: {len(valid_results)} valid, "
        f"{len(results) - len(valid_results)} invalid"
    )
    
    # Sort by rank
    valid_results.sort(key=lambda r: r.rank)
    
    return valid_results
