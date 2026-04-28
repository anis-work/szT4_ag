"""CV Retrieval Plugin for Semantic Kernel.

Provides semantic kernel plugin for embedding queries and retrieving similar CVs.
"""

from typing import Annotated
from semantic_kernel.functions import kernel_function
from embedder import embed_query
from vector_store import VectorStore


class CVRetrievalPlugin:
    """Plugin for retrieving relevant CVs from vector store.
    
    Embeds a query and searches the vector store for the most similar CVs.
    """
    
    def __init__(self, vector_store: VectorStore) -> None:
        """Initialize the retrieval plugin.
        
        Args:
            vector_store: VectorStore instance containing embedded CVs.
        """
        self._vector_store = vector_store
    
    @kernel_function(description="Retrieve relevant CVs for a job query")
    def retrieve(
        self,
        query: Annotated[str, "The job requirements or query to search for"],
        top_k: Annotated[int, "Number of top CVs to retrieve"] = 3,
    ) -> str:
        """Retrieve the most relevant CVs for a given query.
        
        Embeds the query using the embedding model and searches the vector store
        for the top-k most similar CVs.
        
        Args:
            query: Job description or search query string.
            top_k: Number of top results to return (default: 3).
        
        Returns:
            Formatted string containing the top-k CVs for injection into ranking prompt.
        
        Raises:
            RuntimeError: If embedding or search fails.
        """
        try:
            # Embed the query
            query_embedding = embed_query(query)
            
            # Search vector store
            similar_cvs = self._vector_store.search(query_embedding, top_k=top_k)
            
            # Format results as string
            result_text = "RETRIEVED CANDIDATES:\n\n"
            for i, cv in enumerate(similar_cvs, 1):
                result_text += f"Candidate {i}: {cv.candidate_name}\n"
                result_text += f"ID: {cv.id}\n"
                result_text += f"Profile:\n{cv.raw_text}\n"
                result_text += "-" * 80 + "\n\n"
            
            return result_text
        
        except Exception as e:
            raise RuntimeError(
                f"Failed to retrieve CVs: {str(e)}"
            ) from e
