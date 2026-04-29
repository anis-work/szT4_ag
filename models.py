"""Pydantic models for CV Ranking Agent."""

from typing import Optional, List
from pydantic import BaseModel, Field


class CV(BaseModel):
    id: str = Field(..., description="Unique identifier for the CV")
    candidate_name: str = Field(..., description="Full name of the candidate")
    raw_text: str = Field(..., description="Full text content of the CV")
    embedding: Optional[List[float]] = Field(default=None)


class JobDescription(BaseModel):
    role: str = Field(..., description="Job title or role name")
    requirements: str = Field(..., description="Full job description and requirements")
    embedding: Optional[List[float]] = Field(default=None)


class RankedResult(BaseModel):
    rank: int = Field(..., description="Ranking position (1-based)")
    candidate_name: str = Field(..., description="Name of the candidate")
    score: int = Field(..., ge=0, le=100, description="Ranking score 0-100")
    reason: str = Field(..., description="Explanation for the ranking")

    # AI detection
    ai_verdict: str = Field(default="Unknown", description="Human / Likely AI / Uncertain")
    ai_confidence: int = Field(default=0, ge=0, le=100, description="AI detection confidence 0-100")
    ai_reason: str = Field(default="", description="Brief reason for AI verdict")

    # Red flags
    red_flags: List[str] = Field(default_factory=list, description="List of red flags detected")

    # Skills gap
    skills_match: List[str] = Field(default_factory=list, description="JD requirements the candidate meets")
    skills_missing: List[str] = Field(default_factory=list, description="JD requirements the candidate lacks")

    # Seniority
    seniority_fit: str = Field(default="", description="Underqualified / Good Fit / Overqualified + brief note")
