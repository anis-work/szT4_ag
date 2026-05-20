# CV Ranking Agent

An AI-powered candidate shortlisting tool built with **Streamlit**, **Semantic Kernel**, and **Google Gemini**. Upload resumes, paste a job description, and get a ranked, scored shortlist in under a minute.

> **📖 For an in-depth architectural explanation, see [COMPREHENSIVE_ARCHITECTURE.md](COMPREHENSIVE_ARCHITECTURE.md)** — covers Semantic Kernel orchestration, data flow, design decisions, and complete system integration.

---

## What it does

- Accepts PDF and DOCX resumes — any number of files
- Reads and understands the full text of each resume
- Compares every candidate against your job description using semantic AI
- Returns a ranked list with a score (0–100) and a written reason per candidate
- Lets you download the results as a CSV

---

## System Overview

```
┌──────────────┐         ┌──────────────┐
│  Upload CVs  │         │  Job Details │
└──────┬───────┘         └──────┬───────┘
       └──────────────┬─────────┘
                      ↓
        ┌─────────────────────────┐
        │  File Extraction Layer  │
        │  (text + chunking)      │
        └───────────┬─────────────┘
                    ↓
        ┌─────────────────────────┐
        │  Embedding Layer        │
        │  (Google Gemini API)    │
        └───────────┬─────────────┘
                    ↓
        ┌─────────────────────────┐
        │  Vector Search Layer    │
        │  (cosine similarity)    │
        └───────────┬─────────────┘
                    ↓
        ┌─────────────────────────┐
        │  Semantic Kernel        │
        │  (orchestration)        │
        └───────────┬─────────────┘
                    ↓
        ┌─────────────────────────┐
        │  AI Ranking Pipeline    │
        │  (LLM evaluation)       │
        └───────────┬─────────────┘
                    ↓
        ┌─────────────────────────┐
        │  Validation + Display   │
        │  (CSV export)           │
        └─────────────────────────┘
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Anis196/ag_sz__t4.git
cd ag_sz__t4
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up your API key

```bash
cp .env
```

Open `.env` and add your Google AI API key:

```
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_MODEL=gemini-2.5-flash
EMBEDDING_MODEL=models/gemini-embedding-001
TOP_K=3
```

Get a free API key at [aistudio.google.com](https://aistudio.google.com/app/apikey).

### 4. Launch the app

```bash
streamlit run app.py
```

The app opens automatically in your browser at `http://localhost:8501`.

---

## How to use it

### Step 1 — Upload resumes
Click the upload area on the left and select one or more PDF or DOCX files.

> **Tip:** Name your files `firstname_lastname.pdf` — the filename is used as the candidate name in results. For example, `john_doe.pdf` appears as **John Doe**.

### Step 2 — Enter the job details
On the right panel:
- Enter the **Job Title** (e.g. *Senior Data Engineer*)
- Paste the **full job description** — the more detail you include, the more accurate the ranking

### Step 3 — Run the analysis
Click **Run Analysis**. The agent will:
1. Extract text from each resume
2. Embed and index all candidates
3. Retrieve the most relevant profiles
4. Ask Gemini to score and rank each candidate against your JD

### Step 4 — Review and export
Results appear as ranked candidate cards showing:
- Rank position
- Score out of 100
- Written justification from the AI

Use the **Download as CSV** button to export results for sharing or record-keeping.

---

## File naming convention

| Filename | Displayed as |
|---|---|
| `john_doe.pdf` | John Doe |
| `priya_nair.docx` | Priya Nair |
| `alex-kumar.pdf` | Alex Kumar |

Underscores and hyphens are both converted to spaces automatically.

---

## Supported file formats

| Format | Notes |
|---|---|
| `.pdf` | Must be text-based (not scanned images) |
| `.docx` | Standard Word documents |
| `.doc` | Older Word format |

Scanned PDFs (image-only) will be skipped with a warning.

---

## Score guide

| Score | Meaning |
|---|---|
| 75 – 100 | Strong match — meets most or all requirements |
| 50 – 74 | Partial match — meets some requirements |
| 0 – 49 | Weak match — significant gaps against the JD |

---

## Project structure

```
cv_ranker/
├── app.py                          # Streamlit web application
├── main.py                         # CLI entry point (alternative to app.py)
├── config.py                       # Loads .env configuration
├── models.py                       # Data models (CV, JobDescription, RankedResult)
├── embedder.py                     # Gemini embedding calls
├── vector_store.py                 # In-memory cosine similarity search
├── pdf_loader.py                   # PDF and DOCX text extraction + chunking
├── validator.py                    # Result validation
├── sample_data.py                  # Built-in sample CVs for testing
├── plugins/
│   ├── cv_ingestion_plugin.py      # Semantic Kernel plugin: load files
│   ├── cv_retrieval_plugin.py      # Semantic Kernel plugin: semantic search
│   └── cv_ranking_plugin.py        # Semantic Kernel plugin: rank candidates
├── .env.example                    # Environment variable template
├── requirements.txt                # Python dependencies
└── GUIDE.md                        # Detailed usage guide
```

---

## CLI usage (optional)

If you prefer the command line over the web UI:

```bash
# With your own resumes and JD
python main.py --folder resumes/ --role "Senior Python Backend Engineer" --requirements resumes/job_requirements.txt

# With built-in sample data
python main.py
```

---

## Configuration

All settings are in `.env`:

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | — | Your Google AI API key (required) |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model used for ranking |
| `EMBEDDING_MODEL` | `models/gemini-embedding-001` | Model used for embeddings |
| `TOP_K` | `3` | Candidates retrieved before ranking (auto-scales in app mode) |

---

## Troubleshooting

**"No text could be extracted"**
The PDF is likely a scanned image. Convert it to a text-based PDF or use the DOCX version.

**429 / 503 errors**
The Google AI free tier has rate limits. The app retries automatically up to 3 times. If it persists, wait a minute and try again, or upgrade your API plan.

**Candidate name shows as filename**
This is expected — the agent uses the filename as the candidate name. Rename files to `firstname_lastname.pdf` before uploading.

---

## Architecture Quick Reference

### High-Level Component Overview

```
APPLICATION LAYER (Streamlit)
├── app.py                          # UI orchestration
├── utils/ui_components.py          # Reusable UI components
├── utils/results_handler.py        # Results display
└── utils/pipeline.py               # Semantic Kernel orchestration

DATA LAYER
├── models.py                       # Pydantic schemas
├── embedder.py                     # Google Gemini embeddings
├── vector_store.py                 # In-memory similarity search
├── pdf_loader.py                   # Document extraction
└── validator.py                    # Result validation

PLUGIN LAYER (Semantic Kernel)
└── plugins/cv_retrieval_plugin.py  # Vector search as kernel function

EXTERNAL SERVICES
└── Google Gemini API               # Embeddings + LLM ranking
```

### Key Technologies

| Component | Purpose | Why Chosen |
|-----------|---------|-----------|
| **Semantic Kernel** | Orchestration & function composition | Plugin system, prompt management, service abstraction |
| **Google Gemini** | Embeddings + LLM ranking | Unified provider, excellent reasoning, cost-effective |
| **Streamlit** | Web UI framework | Rapid development, built-in state management |
| **NumPy** | Vector operations | Fast cosine similarity, minimal dependencies |
| **Pydantic** | Data validation | Type safety, automatic JSON handling |

### Data Flow

```
User Inputs (CVs + Job Description)
    ↓
File Processing (text extraction + chunking)
    ↓
Embedding (Google Gemini → 768-dim vectors)
    ↓
Vector Store (in-memory cosine similarity search)
    ↓
Semantic Kernel Orchestration (multi-step workflow)
    ├─ Retrieve relevant CVs (plugin)
    └─ Rank candidates (LLM prompt)
    ↓
Validation (schema check, score range)
    ↓
Display & Export (Streamlit UI + CSV)
```

### Why Semantic Kernel?

Semantic Kernel provides elegant orchestration for multi-step AI workflows:

1. **Plugin Architecture** — Register domain-specific functions (e.g., `CVRetrievalPlugin`)
2. **Prompt Management** — Template variables ({{$job_description}}) with type safety
3. **Function Composition** — Chain plugins and prompts seamlessly
4. **Service Abstraction** — Switch between LLM providers (Google, OpenAI, Azure) without code changes
5. **Built-In Resilience** — Error handling and retry logic

**Without SK**: Manual API calls, string formatting, error handling → boilerplate & maintenance burden  
**With SK**: Declarative workflow, automatic context passing, cleaner code

---

## Deep Dive Documentation

### 📖 For Complete Architectural Details

**See [COMPREHENSIVE_ARCHITECTURE.md](COMPREHENSIVE_ARCHITECTURE.md)** for:

- **Semantic Kernel Explained** — Why SK is the orchestration foundation, how it connects all pieces
- **System Architecture** — Layered design (Presentation → Orchestration → Business Logic → Data → External Services)
- **Complete Data Flow** — End-to-end request walkthrough with step-by-step transformations
- **File-by-File Breakdown** — Purpose, responsibilities, and key code sections for every file
- **Vector Search Details** — Embedding space concept, Google Gemini embedding model, cosine similarity algorithm
- **AI Ranking Pipeline** — Prompt design principles, scoring rules, LLM configuration
- **Design Decisions** — Why each tool was chosen (SK vs. direct API, Gemini vs. OpenAI, in-memory vs. DB, etc.)
- **API Methods** — Google Gemini endpoints, request/response formats, rate limiting, error handling

---

## Component Reference

### Semantic Kernel Plugins

**CVRetrievalPlugin** (`plugins/cv_retrieval_plugin.py`)
- **Kernel Function**: `retrieve(query, top_k)`
- **Task**: Embed job description + search vector store for similar CVs
- **Returns**: Formatted string for prompt injection

**Usage in pipeline**:
```python
kernel.add_plugin(CVRetrievalPlugin(vector_store), "retrieval")
retrieve_fn = kernel.get_function("retrieval", "retrieve")
results = await kernel.invoke(retrieve_fn, query=jd.requirements, top_k=len(cvs))
```

### Ranking Prompt Function

**Kernel Function**: `rank_candidates`  
**Template**: `RANKING_PROMPT` (in `utils/pipeline.py`)  
**Variables**: `{{$job_description}}`, `{{$retrieved_cvs}}`  
**Model**: Gemini 2.5 Flash  
**Config**: Temperature=0.0 (deterministic), Seed=42 (reproducible)  
**Output**: JSON array of ranked candidates with scores

### Embedding Models

**Document Embedding**
- Model: `models/gemini-embedding-001`
- Task Type: `RETRIEVAL_DOCUMENT`
- Use Case: Embed CVs
- Output: 768-dimensional vector

**Query Embedding**
- Model: `models/gemini-embedding-001`
- Task Type: `RETRIEVAL_QUERY`
- Use Case: Embed job description
- Output: 768-dimensional vector (same space as documents)

---

## Configuration Reference

### Environment Variables

```bash
# Required
GOOGLE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx

# Optional (defaults provided)
GEMINI_MODEL=gemini-2.5-flash
EMBEDDING_MODEL=models/gemini-embedding-001
TOP_K=20
```

### Ranking Scoring Rules

| Score Range | Interpretation | Criteria |
|---|---|---|
| 90–100 | **Exceptional** | Exceeds all requirements, has bonus skills |
| 75–89 | **Strong** | Meets all required skills and experience |
| 60–74 | **Adequate** | Meets 80%+ required skills, close to experience |
| 40–59 | **Weak** | Meets 50–79% required skills, gaps in experience |
| 0–39 | **Poor** | Missing critical skills (<50%), insufficient experience |

**Penalties**:
- Each missing required skill: −10 to −15 points
- Each missing critical skill: −20 points
- Lack of required certifications: −10 points

---

## Performance Notes

### Benchmarks (on typical system)

- **CV Embedding**: ~100–200ms per CV (Google API latency)
- **Vector Search**: <1ms for 500 CVs (cosine similarity, NumPy)
- **Ranking Invocation**: 2–5 seconds (LLM inference)
- **Total for 10 CVs**: ~15–30 seconds
- **Total for 100 CVs**: ~2–3 minutes

### Scalability

- **Recommended CV count per evaluation**: 10–500
- **Max tested**: 1,000 CVs (single batch)
- **Beyond 1,000**: Consider external vector DB (Pinecone, Weaviate)

### Cost Estimate (Google AI API)

- **Embedding**: $0.025 per 1M input tokens (~$0.000001 per CV)
- **LLM Ranking**: ~0.5 tokens per word in prompt
- **Typical cost per evaluation**: $0.01–$0.05 (10–100 CVs)

---

## Extending the System

### Add a New Plugin

```python
# plugins/my_new_plugin.py
from semantic_kernel.functions import kernel_function
from typing import Annotated

class MyNewPlugin:
    @kernel_function(description="My function description")
    def my_function(
        self,
        param1: Annotated[str, "Parameter description"]
    ) -> str:
        return "result"

# In utils/pipeline.py:
kernel.add_plugin(MyNewPlugin(), "my_plugin")
my_fn = kernel.get_function("my_plugin", "my_function")
result = await kernel.invoke(my_fn, param1="value")
```

### Add a New Validation Rule

```python
# In validator.py, extend validate_results():
def validate_results(results):
    # ... existing validations ...
    
    for result in results:
        # Add new validation
        if result.score > 100:
            logger.warning(f"Score exceeds 100: {result.score}")
            # Handle invalid result
```

### Switch LLM Provider

```python
# In utils/pipeline.py, modify get_kernel():
# Instead of GoogleAIChatCompletion:
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

kernel.add_service(OpenAIChatCompletion(
    model_id="gpt-4",
    api_key=OPENAI_API_KEY
))
```

---

## Troubleshooting Guide

### API Errors

**Error: "GOOGLE_API_KEY not set in environment"**
- Solution: Ensure `.env` file exists and contains `GOOGLE_API_KEY`

**Error: 429 (Too Many Requests)**
- Solution: App retries automatically. If persistent, upgrade API plan or reduce batch size

**Error: 503 (Service Unavailable)**
- Solution: Google services temporarily down. App retries with exponential backoff

### File Processing Errors

**"No text could be extracted"**
- Likely cause: Scanned PDF (image-only)
- Solution: Use text-based PDF or convert to DOCX first

**"Path traversal attempt blocked"**
- Cause: Malicious filename (e.g., `../../../etc/passwd`)
- Solution: Rename file with safe characters

### LLM Output Errors

**"Failed to parse result item"**
- Cause: LLM returned invalid JSON
- Solution: Invalid results are logged and skipped. Check logs for details.

**Inconsistent scores across runs**
- Cause: Temperature not 0.0 or seed not set
- Solution: Check `GoogleAIPromptExecutionSettings` in `get_kernel()`

---

## License

MIT
