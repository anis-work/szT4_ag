"""Configuration module for CV Ranking Agent.

Loads environment variables and provides model/API configuration.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Google AI API Configuration
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not set in environment. "
        "Please create a .env file with GOOGLE_API_KEY=your_key"
    )

# Model Configuration
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "models/text-embedding-001")

# RAG Configuration
TOP_K: int = int(os.getenv("TOP_K", "3"))

# Validate configuration
if not GEMINI_MODEL:
    raise ValueError("GEMINI_MODEL not set")
if not EMBEDDING_MODEL:
    raise ValueError("EMBEDDING_MODEL not set")
if TOP_K <= 0:
    raise ValueError("TOP_K must be a positive integer")
