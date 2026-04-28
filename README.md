# CV Ranking Agent - RAG System

A production-ready RAG-based CV ranking agent using **Semantic Kernel** orchestrator with **Google Gemini** backend.

## Features

- **Semantic Retrieval**: Embed CVs and job descriptions with Google's text-embedding model
- **Vector Search**: In-memory cosine similarity search using NumPy
- **Semantic Ranking**: Use Gemini to intelligently rank candidates against job requirements
- **Semantic Kernel Integration**: Full SK plugin system with `.prompty` template support
- **Type Safety**: Pydantic models for all data structures
- **Comprehensive Validation**: Result schema validation and sanity checks
- **Detailed Logging**: Structured logging throughout the pipeline

## Project Structure

```
cv_ranker/
├── main.py                        # Entry point — builds kernel, runs pipeline
├── config.py                      # Loads env vars, model constants
├── models.py                      # Pydantic models for CV, JD, RankedResult
├── embedder.py                    # Raw genai embedding calls (not through SK)
├── vector_store.py                # In-memory cosine similarity store (numpy)
├── validator.py                   # Post-output schema + sanity validation
├── sample_data.py                 # 5 sample CVs + 1 job description
├── plugins/
│   ├── __init__.py
│   ├── cv_retrieval_plugin.py     # SK plugin: embed query, search vector store
│   └── cv_ranking_plugin.py       # SK plugin: rank retrieved CVs via Gemini
├── prompts/
│   └── rank_candidates.prompty    # SK .prompty prompt template for ranking
├── .env.example                   # Environment variable template
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Setup

### 1. Clone/Install Dependencies

```bash
cd cv_ranker
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your Google API key:

```bash
cp .env.example .env
```

Edit `.env`:
```
GOOGLE_API_KEY=your_actual_api_key_here
GEMINI_MODEL=gemini-1.5-flash
EMBEDDING_MODEL=models/text-embedding-004
TOP_K=3
```

### 3. Run the Agent

```bash
python main.py
```

## Architecture

### Data Flow

1. **CV Loading & Embedding** (main.py)
   - Load 5 sample CVs from `sample_data.py`
   - Embed each CV using `embedder.embed_text()` (raw genai call)
   - Store embeddings in `VectorStore`

2. **Query Processing** (CVRetrievalPlugin)
   - Embed job description query using `embedder.embed_query()`
   - Search vector store using cosine similarity
   - Return top-K most relevant CVs as formatted text

3. **Ranking** (CVRankingPlugin)
   - Pass JD + retrieved CVs to `rank_candidates.prompty`
   - Gemini scores each candidate 0-100
   - Returns JSON array of RankedResult objects

4. **Validation** (validator.py)
   - Verify unique ranks, valid scores (0-100), non-empty reasons
   - Log and remove invalid results
   - Return validated, sorted results

5. **Output** (main.py)
   - Display ranked candidates with scores and reasoning

### Key Design Decisions

**Semantic Kernel for Orchestration**
- Plugins are registered with `@kernel_function` decorator
- Both plugins use SK's invocation system (`kernel.invoke()`)
- Prompty template uses SK's standardized `.prompty` format with YAML frontmatter

**Raw GenAI for Embeddings**
- Embeddings stay outside SK to use task-type optimization
- `embed_text()` uses `task_type="RETRIEVAL_DOCUMENT"`
- `embed_query()` uses `task_type="RETRIEVAL_QUERY"`
- This ensures correct embedding optimization for retrieval tasks

**In-Memory Vector Store**
- NumPy-based cosine similarity
- No external vector DB dependency
- Suitable for demo/testing; replace with Pinecone/Weaviate for production

**Pydantic Validation**
- Strong typing throughout
- RankedResult ensures 0-100 score bounds
- Post-processing validator catches any edge cases

## Module Reference

### config.py
Loads configuration from `.env`:
- `GOOGLE_API_KEY`: Your Google AI API key
- `GEMINI_MODEL`: Model ID (default: "gemini-1.5-flash")
- `EMBEDDING_MODEL`: Embedding model ID (default: "models/text-embedding-004")
- `TOP_K`: Number of candidates to retrieve (default: 3)

### models.py
**CV**: Candidate profile with id, name, raw text, optional embedding
**JobDescription**: Role, requirements, optional embedding
**RankedResult**: rank (1-based), candidate name, score (0-100), reason text

### embedder.py
Raw google-generativeai calls:
- `embed_text(text: str) -> list[float]`: Document embedding with RETRIEVAL_DOCUMENT task
- `embed_query(text: str) -> list[float]`: Query embedding with RETRIEVAL_QUERY task

### vector_store.py
In-memory vector store using NumPy:
- `add(cv: CV)`: Register CV with embedding
- `search(query_embedding, top_k)`: Cosine similarity search
- Returns top-K CVs ordered by similarity

### plugins/cv_retrieval_plugin.py
SK plugin for semantic retrieval:
- `retrieve(query: str, top_k: int) -> str`: Embed query, search store, return formatted CVs
- Uses `@kernel_function` decorator for SK registration

### plugins/cv_ranking_plugin.py
SK plugin for Gemini-based ranking:
- `rank(job_description: str, retrieved_cvs: str) -> str`: Invoke prompty, return JSON
- Uses `kernel.invoke()` to execute the ranking prompt
- Validates JSON output

### prompts/rank_candidates.prompty
SK `.prompty` template:
- YAML frontmatter with model config (gemini-1.5-flash, temperature 0.7)
- Input variables: job_description, retrieved_cvs
- Instructs Gemini to score 0-100 and return ONLY JSON (no markdown)
- SK VS Code extension can preview/test this directly in the editor

### validator.py
Post-processing validation:
- `validate_results(results)`: Check unique ranks, score bounds, non-empty reasons
- Logs warnings for failures
- Returns only valid results, sorted by rank

### sample_data.py
Realistic test data:
- 5 diverse software engineer CVs (senior match, mid-level, full-stack, overqualified, frontend)
- 1 Senior Python Backend Engineer JD with FastAPI/PostgreSQL/AWS requirements

## Extending the Agent

### Add Custom CVs
Edit `sample_data.py`:
```python
def get_sample_cvs() -> List[CV]:
    cvs = [
        CV(
            id="cv_006",
            candidate_name="Your Name",
            raw_text="Your CV text..."
        ),
        # ...
    ]
    return cvs
```

### Modify Ranking Prompt
Edit `prompts/rank_candidates.prompty`:
- Adjust YAML model config (temperature, max_tokens, etc.)
- Rewrite the prompt instructions for different evaluation criteria
- SK VS Code extension will preview changes in real-time

### Replace Vector Store
Substitute `vector_store.py` with Pinecone/Weaviate client:
```python
class VectorStore:
    def add(self, cv: CV) -> None:
        self.index.upsert([(cv.id, cv.embedding)])
    
    def search(self, query_embedding, top_k):
        results = self.index.query(query_embedding, top_k=top_k)
        return [self._cvs[id] for id, _ in results]
```

### Add More Plugins
Create new plugin files in `plugins/` and register in main.py:
```python
kernel.add_plugin(YourPlugin(...), plugin_name="your_plugin")
```

## Error Handling

All modules use try/except with contextual error messages:
- Embedding failures logged with input text preview
- Vector store operations validate non-empty store
- JSON parsing errors show the invalid response
- Validation failures logged but don't crash the pipeline

## Testing

Run with the included sample data:
```bash
python main.py
```

Expected output:
```
==================================================
CV RANKING RESULTS
==================================================

[Rank 1] Alice Johnson
Score: 95/100
Reason: Exact match with 5+ years FastAPI and PostgreSQL...

[Rank 2] Bob Smith
Score: 78/100
Reason: Strong Python skills but less cloud experience...
...
```

## License

MIT
