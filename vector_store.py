"""Vector store module for CV Ranking Agent.

Provides in-memory vector similarity search using numpy and cosine similarity.
"""

from typing import List
import numpy as np
from models import CV


class VectorStore:
    """In-memory vector store using cosine similarity.
    
    Stores CV embeddings and performs similarity-based retrieval.
    Uses numpy for cosine similarity computation.
    """
    
    def __init__(self) -> None:
        """Initialize an empty vector store."""
        self._cvs: List[CV] = []
        self._embeddings: List[np.ndarray] = []
    
    def add(self, cv: CV) -> None:
        """Add a CV with embedding to the store.
        
        Args:
            cv: CV object with populated embedding.
        
        Raises:
            ValueError: If CV embedding is None.
        """
        if cv.embedding is None:
            raise ValueError(f"CV {cv.id} has no embedding")
        
        self._cvs.append(cv)
        self._embeddings.append(np.array(cv.embedding, dtype=np.float32))
    
    def search(self, query_embedding: List[float], top_k: int = 3) -> List[CV]:
        """Search for the most similar CVs to a query embedding.
        
        Uses cosine similarity to rank CVs.
        
        Args:
            query_embedding: Query vector as list of floats.
            top_k: Number of top results to return.
        
        Returns:
            List of most similar CVs, ordered by similarity (highest first).
        
        Raises:
            ValueError: If store is empty or top_k is invalid.
        """
        if not self._cvs:
            raise ValueError("Vector store is empty")
        
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        
        if top_k > len(self._cvs):
            top_k = len(self._cvs)
        
        # Convert query to numpy array
        query_vec = np.array(query_embedding, dtype=np.float32)
        
        # Compute cosine similarity for all CVs
        # cosine_similarity = (A · B) / (||A|| * ||B||)
        similarities = []
        for emb in self._embeddings:
            dot_product = np.dot(query_vec, emb)
            norm_query = np.linalg.norm(query_vec)
            norm_emb = np.linalg.norm(emb)
            
            if norm_query == 0 or norm_emb == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (norm_query * norm_emb)
            
            similarities.append(similarity)
        
        # Get indices of top-k highest similarities
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return corresponding CVs
        return [self._cvs[i] for i in top_indices]
    
    def __len__(self) -> int:
        """Return the number of CVs in the store."""
        return len(self._cvs)
