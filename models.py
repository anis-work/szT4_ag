"""Pydantic models for CV Ranking Agent.

Defines data structures for CVs, job descriptions, and ranking results.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class CV(BaseModel):
    """Model representing a candidate CV.
    
    Attributes:
        id: Unique identifier for the CV.
        candidate_name: Full name of the candidate.
        raw_text: Full text content of the CV.
        embedding: Vector embedding of the CV text (optional).
    """
    id: str = Field(..., description="Unique identifier for the CV")
    candidate_name: str = Field(..., description="Full name of the candidate")
    raw_text: str = Field(..., description="Full text content of the CV")
    embedding: Optional[List[float]] = Field(
        default=None, description="Vector embedding of the CV text"
    )


class JobDescription(BaseModel):
    """Model representing a job description.
    
    Attributes:
        role: Job title/role.
        requirements: Full requirements and description text.
        embedding: Vector embedding of the requirements (optional).
    """
    role: str = Field(..., description="Job title or role name")
    requirements: str = Field(..., description="Full job description and requirements")
    embedding: Optional[List[float]] = Field(
        default=None, description="Vector embedding of the job description"
    )


class RankedResult(BaseModel):
    """Model representing a ranked CV result.
    
    Attributes:
        rank: Ranking position (1-based).
        candidate_name: Name of the candidate.
        score: Ranking score (0-100).
        reason: Explanation for the ranking.
    """
    rank: int = Field(..., description="Ranking position (1-based)")
    candidate_name: str = Field(..., description="Name of the candidate")
    score: int = Field(..., ge=0, le=100, description="Ranking score 0-100")
    reason: str = Field(..., description="Explanation for the ranking")
