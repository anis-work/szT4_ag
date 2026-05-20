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
    experience_years: float = Field(default=0.0, description="Total years of experience extracted from CV")
    key_strengths: str = Field(default="", description="Brief summary of candidate's key strengths for comparison")
    skills_matched: int = Field(default=0, description="Number of required skills matched")
    skills_missing: str = Field(default="", description="Critical skills missing from CV")