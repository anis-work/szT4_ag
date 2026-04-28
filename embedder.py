"""Embedder module using google.genai for text embeddings."""

from typing import List
from google import genai
from google.genai import types
from config import GOOGLE_API_KEY, EMBEDDING_MODEL

_client = genai.Client(api_key=GOOGLE_API_KEY)


def embed_text(text: str) -> List[float]:
    result = _client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    return result.embeddings[0].values


def embed_query(text: str) -> List[float]:
    result = _client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return result.embeddings[0].values
