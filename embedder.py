"""Embedder module using google.genai for text embeddings."""

import time
import logging
from typing import List
from google import genai
from google.genai import types
from config import GOOGLE_API_KEY, EMBEDDING_MODEL

_client = genai.Client(api_key=GOOGLE_API_KEY)
logger = logging.getLogger(__name__)


def _embed_with_retry(contents: str, task_type: str, retries: int = 5) -> List[float]:
    """Embed with exponential backoff retry on 429."""
    for attempt in range(1, retries + 1):
        try:
            result = _client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=contents,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            return result.embeddings[0].values
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = 10 * attempt
                logger.warning(f"Embedding rate limited, retrying in {wait}s (attempt {attempt}/{retries})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Embedding failed after max retries")


def embed_text(text: str) -> List[float]:
    return _embed_with_retry(text, "RETRIEVAL_DOCUMENT")


def embed_query(text: str) -> List[float]:
    return _embed_with_retry(text, "RETRIEVAL_QUERY")
