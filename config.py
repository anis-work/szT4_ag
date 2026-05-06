"""Configuration module for CV Ranking Agent."""

import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not set in environment.")

GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")

TOP_K: int = int(os.getenv("TOP_K", "20"))
if TOP_K <= 0:
    raise ValueError("TOP_K must be a positive integer")
