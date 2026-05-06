# CV Ranking Agent: Complete Architectural Deep-Dive

## Executive Summary

The CV Ranking Agent is an enterprise-grade AI-powered recruitment platform built on **Microsoft's Semantic Kernel** orchestration framework. It seamlessly integrates vector-based semantic search with multi-step AI reasoning pipelines to perform sophisticated candidate evaluation. The system leverages **Google Gemini** for both embeddings and LLM-based ranking, wrapped in **Semantic Kernel's plugin system** for modular, composable business logic.

---

## Table of Contents

1. [Semantic Kernel: The Orchestration Foundation](#semantic-kernel-the-orchestration-foundation)
2. [Technology Stack & Component Selection](#technology-stack--component-selection)
3. [System Architecture & Layers](#system-architecture--layers)
4. [Complete Data Flow](#complete-data-flow)
5. [File-by-File Architecture](#file-by-file-architecture)
6. [Plugin System & Semantic Kernel Integration](#plugin-system--semantic-kernel-integration)
7. [Vector Search & Semantic Matching](#vector-search--semantic-matching)
8. [AI Ranking Pipeline](#ai-ranking-pipeline)
9. [Design Decisions & Tool Selection Rationale](#design-decisions--tool-selection-rationale)
10. [API Methods & External Integrations](#api-methods--external-integrations)

---

## Semantic Kernel: The Orchestration Foundation

### What is Semantic Kernel?

**Semantic Kernel (SK)** is Microsoft's open-source orchestration framework that acts as a "middleware" between your application and Large Language Models (LLMs). It provides:

1. **Function Composition** — Combine AI functions, code functions, and plugins into workflows
2. **Prompt Management** — Store, version, and execute prompts with templating
3. **Plugin Architecture** — Modular, reusable components exposing AI capabilities
4. **Context Variables** — Pass data between functions elegantly
5. **Error Handling & Retry Logic** — Built-in resilience for API calls
6. **Service Abstraction** — Switch between different LLM providers seamlessly

### Why Semantic Kernel for This Project?

#### Problem Without SK:
```python
# Without SK, you'd manually orchestrate:
- Embedding queries and documents
- Calling the LLM with hand-crafted prompts
- Parsing and validating responses
- Error handling for each service
- Passing context between steps
```

#### Solution With SK:
Semantic Kernel elegantly handles this through:

**1. Function-as-a-Plugin Pattern**
```python
@kernel_function(description="Retrieve relevant CVs for a job query")
def retrieve(self, query: str, top_k: int = 3) -> str:
    # Pure business logic — SK handles invocation, retries, error handling
    pass
```

**2. Prompt-as-a-Function**
```python
# RANKING_PROMPT is a kernel function that:
# - Accepts context variables ({{$job_description}}, {{$retrieved_cvs}})
# - Executes with configured LLM settings (temperature, seed)
# - Returns structured JSON results
```

**3. Orchestration Pipeline**
```python
# Run multiple steps through kernel.invoke():
# Step 1: Embed and store CVs
# Step 2: Retrieve similar candidates (via plugin)
# Step 3: Rank candidates (via prompt function)
# SK automatically manages context passing between steps
```

### How SK Connects All Pieces

```
User Input (CVs + Job Description)
        ↓
[Kernel orchestrates multi-step workflow]
        ↓
Step 1: Embed CVs (via embedder module, called directly)
        ↓
Step 2: Store in Vector Store (in-memory)
        ↓
Step 3: Plugin Invoke (CVRetrievalPlugin.retrieve())
        │ └─→ SK calls kernel function with job_description as {{$query}}
        │ └─→ Returns top-k semantically similar CVs
        ↓
Step 4: Ranking Invoke (rank_candidates prompt function)
        │ └─→ SK injects {{$job_description}} and {{$retrieved_cvs}}
        │ └─→ Calls Gemini with temperature=0.0 (deterministic)
        │ └─→ Parses JSON response
        ↓
Step 5: Validation (deterministic scoring)
        ↓
Ranked Results with Scores & Explanations
```

---

## Technology Stack & Component Selection

### Core Framework Stack

| Component | Tool | Version | Why This Choice |
|-----------|------|---------|-----------------|
| **Orchestration** | Microsoft Semantic Kernel | ≥1.3.0 | Plugin system, prompt management, service abstraction, built-in retry logic |
| **LLM Service** | Google Gemini (via google-generativeai & google-genai) | ≥0.8.0 (genai), ≥1.0.0 (google-genai) | State-of-the-art reasoning, real-time API, cost-effective, excellent JSON output |
| **Embeddings** | Google Gemini Embeddings (models/gemini-embedding-001) | Latest | Unified provider (avoids multi-service complexity), optimized for semantic search, RETRIEVAL_DOCUMENT & RETRIEVAL_QUERY task types |
| **Web Framework** | Streamlit | ≥1.35.0 | Rapid prototyping, stateful sessions, built-in async, perfect for data apps, minimal boilerplate |
| **Data Validation** | Pydantic | ≥2.0.0 | Type safety, runtime validation, automatic JSON serialization/deserialization |
| **Vector Search** | NumPy | ≥1.26.0 | In-memory cosine similarity, fast dot-product computation, no external DB overhead |
| **Document Processing** | pypdf + python-docx | ≥4.0.0 (pypdf), ≥1.1.0 (docx) | Industry-standard PDF extraction, native DOCX support, handles chunking |
| **Environment Config** | python-dotenv | ≥1.0.0 | Secure API key management, environment-specific configs |
| **Data Manipulation** | Pandas | ≥2.0.0 | CSV export, data framing, Excel compatibility |

### Architectural Benefits of This Stack

**1. Unified Provider Model**
- Google Gemini for both embeddings AND LLM ranking
- No provider switching overhead
- Unified API key management
- Consistent latency and behavior

**2. Semantic Kernel as Glue**
- Abstracts LLM service calls
- Manages prompt templating
- Handles function composition
- Provides retry resilience

**3. In-Memory Vector Store**
- No database setup complexity
- Suitable for typical recruitment scenarios (10-500 CVs)
- Sub-millisecond similarity search
- Pure Python implementation

---

## System Architecture & Layers

### Layered Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│         PRESENTATION LAYER (Streamlit)                  │
│  ┌──────────────────────────────────────────────────┐   │
│  │  app.py → UI Components (ui_components.py)       │   │
│  │  • Session state management                      │   │
│  │  • User event handlers                           │   │
│  │  • Result display (results_handler.py)           │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│    ORCHESTRATION LAYER (Semantic Kernel)                │
│  ┌──────────────────────────────────────────────────┐   │
│  │  pipeline.py:run_pipeline()                      │   │
│  │  • Coordinates embedding, retrieval, ranking     │   │
│  │  • Manages kernel functions                      │   │
│  │  • Handles retry logic via _invoke_with_retry()  │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│    BUSINESS LOGIC LAYER (Plugins & Functions)           │
│  ┌──────────────────────────────────────────────────┐   │
│  │ CVRetrievalPlugin (via SK plugin architecture)   │   │
│  │  • @kernel_function decorator                    │   │
│  │  • retrieve(query, top_k) → string               │   │
│  │  • Bridges embedder + vector_store               │   │
│  │                                                  │   │
│  │ Ranking Prompt Function (KernelFunctionFromPrompt) │
│  │  • Template: RANKING_PROMPT                      │   │
│  │  • Injects: {{$job_description}}, {{$retrieved_cvs}} │
│  │  • Temperature: 0.0 (deterministic)              │   │
│  │  • Output: JSON array                            │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│    DATA LAYER (Core Utilities)                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  embedder.py                                      │   │
│  │  • embed_text() → List[float]                     │   │
│  │  • embed_query() → List[float]                    │   │
│  │                                                    │   │
│  │  vector_store.py                                  │   │
│  │  • VectorStore class (in-memory)                  │   │
│  │  • add(cv) → stores embedding                     │   │
│  │  • search(embedding, top_k) → [CV]                │   │
│  │                                                    │   │
│  │  pdf_loader.py                                    │   │
│  │  • _extract_text() → str                          │   │
│  │  • _chunk_text() → [str]                          │   │
│  │                                                    │   │
│  │  validator.py                                     │   │
│  │  • validate_results() → [RankedResult]            │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│    EXTERNAL APIs & SERVICES                             │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Google Gemini (via google-genai)                 │   │
│  │  • Embeddings: models/gemini-embedding-001        │   │
│  │  • LLM Ranking: gemini-2.5-flash                  │   │
│  │  • Single API key, unified authentication          │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Layer Responsibilities

#### Presentation Layer (Streamlit)
- **File**: `app.py`
- **Purpose**: User interface and interaction
- **Responsibilities**:
  - Render UI components (file upload, job details, results)
  - Manage Streamlit session state
  - Trigger pipeline execution
  - Display results with styling
  - Handle user events (button clicks, file selections)

#### Orchestration Layer (Semantic Kernel)
- **File**: `utils/pipeline.py` (contains kernel initialization and orchestration)
- **Purpose**: Coordinate multi-step AI workflow
- **Responsibilities**:
  - Initialize Semantic Kernel with Google Gemini service
  - Register plugins (CVRetrievalPlugin)
  - Register prompt functions (rank_candidates)
  - Execute functions via `kernel.invoke()`
  - Implement retry logic for transient failures
  - Pass context between workflow steps

#### Business Logic Layer (Plugins)
- **File**: `plugins/cv_retrieval_plugin.py`
- **Purpose**: Domain-specific functionality exposed as kernel functions
- **Responsibilities**:
  - Retrieve semantically similar CVs
  - Format results for downstream processing
  - Pure business logic (no UI, no direct API calls to Gemini — embedder module handles that)

#### Data Layer (Core Utilities)
- **Files**: `embedder.py`, `vector_store.py`, `pdf_loader.py`, `validator.py`, `models.py`
- **Purpose**: Foundation functions and data structures
- **Responsibilities**:
  - Text embedding via Google API
  - Vector similarity search
  - Document extraction and chunking
  - Result validation
  - Data model definitions (Pydantic)

#### External Services
- **Provider**: Google Gemini API
- **Endpoints**:
  - `/models/embed_content` — embeddings
  - LLM inference — Gemini model predictions

---

## Complete Data Flow

### End-to-End Request Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│ USER INTERACTION (Streamlit UI)                                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│ 1. User uploads PDF/DOCX files (CVs)                                     │
│ 2. User enters Job Title and Job Description                            │
│ 3. User clicks "Analyze & Rank Candidates"                              │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
                                 ↓
┌──────────────────────────────────────────────────────────────────────────┐
│ FILE HANDLING (app.py → pipeline.py)                                    │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│ save_uploads(uploaded_files)                                             │
│   ├─ Create temporary directory                                          │
│   ├─ Sanitize filenames (secure_filename)                               │
│   ├─ Write bytes to disk                                                 │
│   └─ Return (tmp_dir, [filenames])                                       │
│                                                                           │
│ build_cvs(tmp_dir, filenames)                                            │
│   ├─ For each file:                                                      │
│   │  ├─ _extract_text(path)                                             │
│   │  │  ├─ If .pdf: use PdfReader, extract from all pages               │
│   │  │  ├─ If .docx: use python-docx, extract paragraphs                │
│   │  │  └─ Return concatenated string                                    │
│   │  │                                                                    │
│   │  ├─ _chunk_text(text)                                               │
│   │  │  ├─ Split into 800-char chunks with 100-char overlap             │
│   │  │  └─ Return [chunk1, chunk2, ...]                                 │
│   │  │                                                                    │
│   │  ├─ Create CV object                                                │
│   │  │  ├─ id: UUID (for tracking)                                      │
│   │  │  ├─ candidate_name: derived from filename                        │
│   │  │  ├─ raw_text: full text (chunks rejoined)                        │
│   │  │  ├─ embedding: None (not yet embedded)                           │
│   │  │  └─ Append to cvs list                                            │
│   │  │                                                                    │
│   │  └─ On error: add (filename, error_reason) to skipped list           │
│   │                                                                        │
│   └─ Return (cvs, skipped)                                               │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
                                 ↓
┌──────────────────────────────────────────────────────────────────────────┐
│ PIPELINE EXECUTION (pipeline.py:run_pipeline)                           │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│ ┌─ STEP 1: EMBEDDING PHASE ─────────────────────────────────────────┐   │
│ │                                                                    │   │
│ │ For each CV in cvs:                                              │   │
│ │   ├─ embed_text(cv.raw_text)                                     │   │
│ │   │  ├─ Call Google Gemini API: /models/embed_content            │   │
│ │   │  ├─ Config: task_type="RETRIEVAL_DOCUMENT"                   │   │
│ │   │  │  └─ Tells Gemini this is a document (not a query)         │   │
│ │   │  │  └─ Optimizes embedding for document retrieval            │   │
│ │   │  ├─ Returns: EmbeddingResult with embeddings[0].values       │   │
│ │   │  └─ Return List[float] (768-dimensional vector)              │   │
│ │   │                                                               │   │
│ │   ├─ cv.embedding = embedding_vector                             │   │
│ │   ├─ Render progress bar (i+1 / len(cvs))                        │   │
│ │   └─ Continue to next CV                                          │   │
│ │                                                                    │   │
│ └────────────────────────────────────────────────────────────────┘   │
│ Status: "📄 Step 1/3: Extracting and embedding resumes..."            │
│                                                                        │
│ ┌─ STEP 2: VECTOR STORE & RETRIEVAL ────────────────────────────┐   │
│ │                                                                 │   │
│ │ vs = VectorStore()                                             │   │
│ │   ├─ Initialize with empty _cvs=[] and _embeddings=[]         │   │
│ │   └─ This is an in-memory object (not persisted)              │   │
│ │                                                                 │   │
│ │ For each cv in cvs:                                           │   │
│ │   ├─ vs.add(cv)                                               │   │
│ │   │  ├─ Validate cv.embedding is not None                     │   │
│ │   │  ├─ Append cv to _cvs list                                │   │
│ │   │  ├─ Convert embedding to np.ndarray(dtype=float32)         │   │
│ │   │  └─ Append np array to _embeddings list                    │   │
│ │   └─ Continue                                                   │   │
│ │                                                                 │   │
│ │ kernel.add_plugin(CVRetrievalPlugin(vs), plugin_name="retrieval") │
│ │   ├─ Initialize CVRetrievalPlugin with reference to vs        │   │
│ │   ├─ Register plugin with kernel                              │   │
│ │   └─ Plugin is now available as kernel function               │   │
│ │                                                                 │   │
│ │ retrieve_fn = kernel.get_function("retrieval", "retrieve")    │   │
│ │   ├─ Retrieve the @kernel_function handle                     │   │
│ │   └─ Use for invocation in next step                          │   │
│ │                                                                 │   │
│ │ retrieved = await _invoke_with_retry(                          │   │
│ │     kernel,                                                    │   │
│ │     retrieve_fn,                                              │   │
│ │     query=jd.requirements,     # ← context variable           │   │
│ │     top_k=len(cvs)             # ← all candidates (no limit)  │   │
│ │ )                                                              │   │
│ │   ├─ kernel.invoke(retrieve_fn, query=..., top_k=...)        │   │
│ │   │  ├─ SK routes to CVRetrievalPlugin.retrieve()            │   │
│ │   │  ├─ Function receives:                                    │   │
│ │   │  │  ├─ query = jd.requirements (full job description)    │   │
│ │   │  │  └─ top_k = len(cvs)                                  │   │
│ │   │  ├─ Inside retrieve():                                   │   │
│ │   │  │  ├─ query_embedding = embed_query(query)              │   │
│ │   │  │  │  ├─ Call Google Gemini API with same query string  │   │
│ │   │  │  │  ├─ Config: task_type="RETRIEVAL_QUERY"            │   │
│ │   │  │  │  │  └─ Gemini optimizes for query embedding        │   │
│ │   │  │  │  └─ Return List[float] (768-dim, same space as docs) │   │
│ │   │  │  │                                                    │   │
│ │   │  │  ├─ similar_cvs = vs.search(query_embedding, top_k)   │   │
│ │   │  │  │  ├─ For each CV embedding:                         │   │
│ │   │  │  │  │  ├─ dot_product = np.dot(query_vec, cv_emb)    │   │
│ │   │  │  │  │  ├─ norm_q = ||query_vec||_2                   │   │
│ │   │  │  │  │  ├─ norm_cv = ||cv_emb||_2                     │   │
│ │   │  │  │  │  ├─ cosine_sim = dot_product / (norm_q * norm_cv) │   │
│ │   │  │  │  │  └─ Append (sim_score, idx) to list            │   │
│ │   │  │  │  ├─ Sort by similarity (descending)               │   │
│ │   │  │  │  ├─ Return top_k CVs in order                     │   │
│ │   │  │  │  │  └─ Higher similarity = better match            │   │
│ │   │  │  │  └─ If top_k > len(cvs): return all CVs            │   │
│ │   │  │  │                                                    │   │
│ │   │  │  └─ result_text = format_cvs_as_string(similar_cvs)  │   │
│ │   │  │     └─ Create readable text: "Candidate 1: ...\nID:...\n..." │   │
│ │   │  │                                                        │   │
│ │   │  └─ Return result_text (string)                          │   │
│ │   │                                                           │   │
│ │   ├─ Retry logic (if service error):                         │   │
│ │   │  ├─ If error contains "503", "429", "UNAVAILABLE", etc.: │   │
│ │   │  ├─ Wait = 30s * attempt_number                          │   │
│ │   │  ├─ Show toast: "⏳ Service busy — retrying..."          │   │
│ │   │  ├─ asyncio.sleep(wait)                                  │   │
│ │   │  └─ Retry (up to 5 times)                                │   │
│ │   │                                                           │   │
│ │   └─ retrieved = str(KernelInvokeResult)                      │   │
│ │                                                                │   │
│ └────────────────────────────────────────────────────────────┘   │
│ Status: "🔍 Step 2/3: Retrieving relevant candidates..."            │
│                                                                    │
│ ┌─ STEP 3: AI RANKING VIA SEMANTIC KERNEL ──────────────────┐   │
│ │                                                            │   │
│ │ rank_fn = kernel.get_function("ranking", "rank_candidates") │   │
│ │   └─ This is KernelFunctionFromPrompt:                    │   │
│ │      ├─ Prompt: RANKING_PROMPT (template with {{$vars}}) │   │
│ │      ├─ Service: GoogleAIChatCompletion (Gemini)         │   │
│ │      └─ Settings: temperature=0.0, seed=42 (deterministic) │   │
│ │                                                            │   │
│ │ result = await _invoke_with_retry(                        │   │
│ │     kernel,                                               │   │
│ │     rank_fn,                                              │   │
│ │     job_description=jd.requirements,  # ← template var    │   │
│ │     retrieved_cvs=str(retrieved)       # ← template var    │   │
│ │ )                                                          │   │
│ │   ├─ kernel.invoke() calls GoogleAIChatCompletion        │   │
│ │   ├─ SK template substitution:                           │   │
│ │   │  ├─ {{$job_description}} = jd.requirements (full JD) │   │
│ │   │  └─ {{$retrieved_cvs}} = formatted CV strings        │   │
│ │   │                                                      │   │
│ │   ├─ Prompt Template (RANKING_PROMPT):                  │   │
│ │   │  ├─ "You are a STRICT technical recruiter..."      │   │
│ │   │  ├─ "JOB DESCRIPTION: {{$job_description}}"         │   │
│ │   │  ├─ "CANDIDATE PROFILES: {{$retrieved_cvs}}"        │   │
│ │   │  ├─ "EVALUATION CRITERIA (STRICT): ..."             │   │
│ │   │  │  ├─ 1. REQUIRED SKILLS MATCHING                  │   │
│ │   │  │  ├─ 2. EXPERIENCE REQUIREMENTS                   │   │
│ │   │  │  ├─ 3. SCORING RULES (0-100 scale)              │   │
│ │   │  │  └─ 4. MISSING SKILLS PENALTY                    │   │
│ │   │  └─ "Return ONLY valid JSON array: [...]"           │   │
│ │   │                                                      │   │
│ │   ├─ Google Gemini Call:                                │   │
│ │   │  ├─ Model: gemini-2.5-flash                         │   │
│ │   │  ├─ Temperature: 0.0 (deterministic output)         │   │
│ │   │  ├─ Seed: 42 (reproducible results)                │   │
│ │   │  ├─ Input: Full prompt with substituted variables  │   │
│ │   │  └─ Returns: JSON array string                      │   │
│ │   │                                                      │   │
│ │   ├─ Response Parsing:                                  │   │
│ │   │  ├─ ranking_json = str(result).strip()             │   │
│ │   │  ├─ If wrapped in ```json ```: extract JSON         │   │
│ │   │  ├─ json.loads(ranking_json) → list of dicts        │   │
│ │   │  └─ Each dict: {rank, candidate_name, score, ...}  │   │
│ │   │                                                      │   │
│ │   └─ Retry logic (same as retrieval)                    │   │
│ │                                                            │   │
│ └────────────────────────────────────────────────────────┘   │
│ Status: "🤖 Step 3/3: AI ranking candidates..."               │
│                                                               │
│ ┌─ RESULT VALIDATION ────────────────────────────────────┐   │
│ │                                                         │   │
│ │ results = [RankedResult(**item) for item in items]    │   │
│ │   ├─ For each dict in JSON response:                  │   │
│ │   │  ├─ Validate against Pydantic schema (RankedResult) │   │
│ │   │  ├─ Fields: rank, candidate_name, score, reason,  │   │
│ │   │  │           experience_years, key_strengths,     │   │
│ │   │  │           skills_matched, skills_missing       │   │
│ │   │  ├─ Type coercion (str→int for score, etc.)       │   │
│ │   │  └─ Create RankedResult object                     │   │
│ │   └─ On validation error: skip item, log warning       │   │
│ │                                                         │   │
│ │ validate_results(results)                              │   │
│ │   ├─ Check all ranks are unique                       │   │
│ │   ├─ Check all scores in [0-100]                      │   │
│ │   ├─ Check all reasons non-empty                      │   │
│ │   ├─ Log validation summary                           │   │
│ │   ├─ Sort by rank (ascending)                         │   │
│ │   └─ Return validated results                          │   │
│ │                                                         │   │
│ └────────────────────────────────────────────────────────┘   │
│                                                               │
│ status_placeholder.empty() ← Clear status message             │
│ Return results to app.py                                      │
│                                                               │
└──────────────────────────────────────────────────────────────────────────┘
                                 ↓
┌──────────────────────────────────────────────────────────────────────────┐
│ SESSION STATE & HISTORY (app.py)                                         │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│ st.session_state.history.append({                                        │
│     "timestamp": current_time_utc,                                        │
│     "role": job_role,                                                     │
│     "candidates": num_candidates_evaluated,                               │
│     "top_candidate": highest_ranked_name,                                 │
│     "top_score": highest_score,                                           │
│     "results": [RankedResult, ...],                                       │
│     "cvs": [CV, ...],                                                     │
│ })                                                                         │
│                                                                           │
│ st.session_state.current_results = {                                     │
│     "results": results,                                                   │
│     "role": job_role,                                                     │
│     "cvs": cvs,                                                           │
│ }                                                                          │
│                                                                           │
│ Store in Streamlit session (in-memory, lost on page reload)               │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
                                 ↓
┌──────────────────────────────────────────────────────────────────────────┐
│ RESULTS DISPLAY (results_handler.py → ui_components.py)                 │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│ show_results(results, role, cvs)                                         │
│   ├─ render_stats_bar(results)                                           │
│   │  ├─ Display: Candidates Evaluated, Top Score, Avg Score             │
│   │  └─ HTML: custom CSS styling (.stat-value, .stat-label)              │
│   │                                                                       │
│   ├─ render_results_table(results)                                       │
│   │  ├─ Create DataFrame from results                                    │
│   │  ├─ Format score with color-coded badge:                            │
│   │  │  ├─ 90-100: green badge (high match)                             │
│   │  │  ├─ 70-89: yellow badge (medium match)                           │
│   │  │  └─ <70: red badge (low match)                                   │
│   │  ├─ Render Streamlit table                                           │
│   │  └─ Columns: Rank, Name, Score, Experience, Strengths, Missing Skills │
│   │                                                                       │
│   └─ render_export_section(results, role)                                │
│      ├─ Generate CSV from results                                        │
│      ├─ Format: rank, candidate_name, score, reason, experience_years,  │
│      │           key_strengths, skills_matched, skills_missing           │
│      ├─ Create download button                                           │
│      └─ File: "ranking_{role}_{timestamp}.csv"                           │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
                                 ↓
                        USER SEES RESULTS
                     (ranked candidates with scores,
                  explanations, strengths, missing skills,
                          and export option)
```

### Key Data Transformations

```
Transformation 1: File → Text
  PDF/DOCX bytes → PyPDF/python-docx → Raw text string

Transformation 2: Text → Chunks
  Raw text → Split(800 chars, 100-char overlap) → [Chunks]
  Purpose: Prepare for embedding without exceeding API limits

Transformation 3: Text → Embedding
  Text string → Google Gemini API → List[float] (768 dims)
  Task-type optimized: RETRIEVAL_DOCUMENT for CVs, RETRIEVAL_QUERY for JD

Transformation 4: Query Embedding + CV Embeddings → Ranked CVs
  Query vector vs all CV vectors → Cosine similarity → Top-k → String format

Transformation 5: Formatted CVs + Job Description → Ranking JSON
  Prompt template → Google Gemini (reasoning) → JSON response
  LLM evaluates each candidate against job description
  Returns: rank, score (0-100), reason, skills matched/missing

Transformation 6: JSON → Validated Objects
  JSON dict → Pydantic validation → RankedResult objects
  Enforces schema, ensures type safety

Transformation 7: RankedResult → Display/Export
  Objects → Streamlit table → HTML display
  Objects → Pandas DataFrame → CSV export
```

---

## File-by-File Architecture

### 1. **app.py** — Entry Point & Session Orchestrator

**Location**: `/app.py`  
**Role**: Presentation Layer + Session Management  
**Lines**: ~100  

**Purpose**:
Main Streamlit application entry point. Orchestrates the user interface and coordinates pipeline execution.

**Key Responsibilities**:
- Initialize Streamlit page config (title, icon, layout)
- Initialize session state (history, results, uploaded files)
- Render UI components (header, file upload, job details)
- Handle user clicks on "Analyze" button
- Call `pipeline.run_pipeline()` with async orchestration
- Store results in session state for persistence during session
- Display success/error messages

**Key Code Sections**:

```python
# Page configuration
st.set_page_config(
    page_title="CV Ranking Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Session initialization
if "history" not in st.session_state:
    st.session_state.history = []  # Retain results from previous runs
if "current_results" not in st.session_state:
    st.session_state.current_results = None  # Display current results

# Event handling
if run:  # User clicked "Analyze"
    kernel = get_kernel()  # Initialize Semantic Kernel
    tmp_dir, filenames = save_uploads(uploaded)  # Save to temp storage
    cvs, skipped = build_cvs(tmp_dir, filenames)  # Extract text from files
    results = asyncio.run(run_pipeline(kernel, cvs, jd, status_placeholder))
    # Store in session
    st.session_state.current_results = {...}
```

**Why This Design**:
- Streamlit's session state naturally handles user data persistence during a session
- Minimal business logic in the entry point (follows separation of concerns)
- Async support allows long-running pipelines without blocking UI

---

### 2. **config.py** — Environment & Configuration

**Location**: `/config.py`  
**Role**: Configuration Management  
**Lines**: ~15  

**Purpose**:
Centralized configuration loading from environment variables.

**Key Variables**:
- `GOOGLE_API_KEY` — API key for Google Gemini (required)
- `GEMINI_MODEL` — LLM model ID (default: "gemini-2.5-flash")
- `EMBEDDING_MODEL` — Embedding model ID (default: "models/gemini-embedding-001")
- `TOP_K` — Number of top candidates to retrieve (default: 20, but pipeline uses all CVs)

**Key Code**:

```python
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not set in environment.")

GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
```

**Why This Design**:
- Separates configuration from code (12-factor app principle)
- Allows environment-specific settings without code changes
- Early validation (raises error if API key is missing)
- Centralized source of truth for all settings

---

### 3. **models.py** — Data Schema & Validation

**Location**: `/models.py`  
**Role**: Data Layer - Schema Definition  
**Lines**: ~40  

**Purpose**:
Define Pydantic models for type-safe data handling and automatic validation.

**Models**:

#### **CV** — Resume Object
```python
class CV(BaseModel):
    id: str                              # Unique identifier (UUID)
    candidate_name: str                  # Name inferred from filename
    raw_text: str                        # Full extracted text (with chunks)
    embedding: Optional[List[float]] = None  # 768-dim embedding vector
```

#### **JobDescription** — Job Details
```python
class JobDescription(BaseModel):
    role: str                            # Job title
    requirements: str                    # Full job description
    embedding: Optional[List[float]] = None
```

#### **RankedResult** — Ranking Output
```python
class RankedResult(BaseModel):
    rank: int                            # Position 1-based
    candidate_name: str                  # Name of candidate
    score: int                           # Score 0-100
    reason: str                          # Explanation from AI
    experience_years: float = 0.0        # Years extracted
    key_strengths: str = ""              # Matching strengths
    skills_matched: int = 0              # Count of required skills
    skills_missing: str = ""             # List of missing skills
```

**Why Pydantic**:
- **Runtime Validation** — Automatic type checking and coercion
- **JSON Serialization** — Automatic conversion to/from JSON
- **Field Documentation** — `description` parameter documents schema
- **Default Values** — Optional fields with sensible defaults
- **Error Messages** — Clear validation error messages

---

### 4. **embedder.py** — Google Gemini Embedding Service

**Location**: `/embedder.py`  
**Role**: Data Layer - External API Integration  
**Lines**: ~25  

**Purpose**:
Wrapper around Google Gemini's embedding API. Converts text to 768-dimensional vectors.

**Key Functions**:

#### **embed_text(text: str) → List[float]**
- **Purpose**: Embed a document (CV)
- **Config**: `task_type="RETRIEVAL_DOCUMENT"`
- **Why "RETRIEVAL_DOCUMENT"**: Tells Gemini this is a document to be retrieved (not a query)
- **Returns**: 768-dimensional float list

#### **embed_query(text: str) → List[float]**
- **Purpose**: Embed a query (job description)
- **Config**: `task_type="RETRIEVAL_QUERY"`
- **Why "RETRIEVAL_QUERY"**: Tells Gemini this is a query to search for documents
- **Returns**: 768-dimensional float list (same embedding space as documents)

**Key Code**:

```python
_client = genai.Client(api_key=GOOGLE_API_KEY)

def embed_text(text: str) -> List[float]:
    result = _client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    return result.embeddings[0].values
```

**Why This Design**:
- **Separation**: Embedding logic isolated in one place
- **Reusability**: Used by both embedder and pipeline
- **Google Gemini Unified Provider**: One API key, one service
- **Task-Type Differentiation**: Optimizes embeddings for retrieval task (better similarity search)

---

### 5. **vector_store.py** — In-Memory Vector Database

**Location**: `/vector_store.py`  
**Role**: Data Layer - Vector Search Engine  
**Lines**: ~80  

**Purpose**:
In-memory vector similarity search using cosine similarity and NumPy.

**Key Class: VectorStore**

#### **__init__()**
- Initializes empty `_cvs` (CV objects) and `_embeddings` (NumPy arrays)
- In-memory only (no persistence)

#### **add(cv: CV)**
- Validates CV has embedding
- Appends CV to `_cvs` list
- Converts embedding to `np.ndarray` and appends to `_embeddings` list
- **Why NumPy**: Fast vectorized dot-product and norm computations

#### **search(query_embedding: List[float], top_k: int) → List[CV]**
- Computes cosine similarity between query and all CVs:

$$\text{cosine\_similarity}(A, B) = \frac{A \cdot B}{||A|| \times ||B||}$$

- **Numerator**: `np.dot(query_vec, emb)` — dot product
- **Denominator**: `||query||` × `||emb||` — L2 norms
- Sorts by similarity (highest first)
- Returns top-k most similar CVs

**Why Cosine Similarity**:
- **Normalized**: Magnitude-invariant (penalizes neither long nor short documents)
- **Interpretable**: Range [−1, 1], 1 = identical direction
- **Fast**: O(n) dot-product for n vectors
- **Meaningful**: In embedding space, measures semantic similarity

**Why In-Memory**:
- **Simplicity**: No database setup required
- **Speed**: Sub-millisecond search for <10,000 vectors
- **Adequate for typical recruitment**: 10-500 CVs per evaluation
- **Cost**: Zero infrastructure cost

---

### 6. **pdf_loader.py** — Document Processing

**Location**: `/pdf_loader.py`  
**Role**: Data Layer - Document Extraction & Preprocessing  
**Lines**: ~80  

**Purpose**:
Extract text from PDF and DOCX files, with intelligent chunking.

**Key Functions**:

#### **_extract_text_pdf(path: str) → str**
- Opens PDF using `PdfReader` (pypdf library)
- Iterates through all pages
- Calls `page.extract_text()` on each page
- Joins all pages with `\n`
- Handles OCR-scanned PDFs gracefully (returns empty string)

#### **_extract_text_docx(path: str) → str**
- Opens DOCX using `Document` (python-docx library)
- Extracts text from all paragraphs
- Joins paragraphs with `\n`

#### **_extract_text(path: str) → str**
- Dispatcher function
- Determines file type by extension
- Calls appropriate extractor
- Raises error if unsupported type

#### **_chunk_text(text: str) → List[str]**
- Splits text into overlapping chunks:
  - **Chunk size**: 800 characters
  - **Overlap**: 100 characters
- **Why overlapping**: Preserves context at chunk boundaries (important for sentence fragments)
- **Why 800 chars**: Typical embedding API limit is ~2000 tokens, 800 chars ≈ 200 tokens (safe margin)

**Chunking Algorithm**:

```python
chunks, start = [], 0
while start < len(text):
    end = start + CHUNK_SIZE  # 800
    chunks.append(text[start:end])
    start += CHUNK_SIZE - CHUNK_OVERLAP  # step = 700
# Result: overlapping windows covering full text
```

**Why Chunking**:
- **Embedding Limits**: APIs have token limits; chunking ensures compliance
- **Information Preservation**: Overlap maintains context across chunks
- **Later Rejoin**: pipeline.py joins chunks back together for ranking prompt

---

### 7. **validator.py** — Result Validation

**Location**: `/validator.py`  
**Role**: Data Layer - Quality Assurance  
**Lines**: ~50  

**Purpose**:
Validate ranking results for schema compliance and correctness.

**Key Function: validate_results(results: List[RankedResult]) → List[RankedResult]**

**Validations**:
1. **Rank Uniqueness**: No duplicate ranks
2. **Score Range**: 0 ≤ score ≤ 100
3. **Non-Empty Reason**: Every result has explanation text
4. **Logging**: Summarizes validation (valid/invalid counts)
5. **Sorting**: Sort results by rank (ascending)

**Error Handling**:
- Invalid entries are logged but not raised (graceful degradation)
- Removes bad entries from result set
- Preserves valid entries

**Why This Design**:
- **Safety**: Catches AI hallucinations (e.g., score=150, duplicate ranks)
- **Observability**: Logging shows data quality issues
- **Robustness**: Application continues even if some results malformed

---

### 8. **utils/pipeline.py** — Semantic Kernel Orchestration

**Location**: `/utils/pipeline.py`  
**Role**: Orchestration Layer - Multi-Step Workflow  
**Lines**: ~280  

**Purpose**:
Orchestrate the complete CV ranking pipeline using Semantic Kernel. This is the heart of the system.

**Key Components**:

#### **RANKING_PROMPT** — Template-Based AI Function

```python
RANKING_PROMPT = """You are a STRICT technical recruiter...
JOB DESCRIPTION:
{{$job_description}}

CANDIDATE PROFILES:
{{$retrieved_cvs}}

EVALUATION CRITERIA (STRICT):
1. REQUIRED SKILLS MATCHING:
   - Extract ALL required skills from job description
   - Check if candidate has EXPLICIT evidence of each skill
   - Count only skills with clear proof
   - Missing a required skill = significant score penalty

2. EXPERIENCE REQUIREMENTS:
   - If JD requires X years, candidate must have X+ years
   - Less experience = major score deduction

3. SCORING RULES (BE HARSH):
   - 90-100: Exceeds ALL requirements, has bonus skills
   - 75-89: Meets ALL required skills, meets experience requirement
   - 60-74: Meets MOST required skills (80%+)
   - 40-59: Meets SOME required skills (50-79%)
   - 0-39: Missing critical skills (<50%)

Return JSON array: [...]
"""
```

**Why This Prompt**:
- **System Instructions**: Frames AI as "STRICT recruiter" (sets tone)
- **Template Variables**: {{$job_description}}, {{$retrieved_cvs}} injected by SK
- **Explicit Criteria**: Clear rubric for scoring (reduces hallucination)
- **Output Format**: Specifies JSON array structure
- **Harsh Grading**: Encourages lower scores (more realistic/defensible)

#### **get_kernel() → Kernel**

```python
@st.cache_resource
def get_kernel() -> Kernel:
    kernel = Kernel()
    
    # Add Google Gemini service
    kernel.add_service(GoogleAIChatCompletion(
        gemini_model_id=GEMINI_MODEL,
        api_key=GOOGLE_API_KEY
    ))
    
    # Register ranking prompt as kernel function
    kernel.add_function(
        plugin_name="ranking",
        function=KernelFunctionFromPrompt(
            function_name="rank_candidates",
            plugin_name="ranking",
            prompt=RANKING_PROMPT,
            prompt_execution_settings=GoogleAIPromptExecutionSettings(
                temperature=0.0,  # Deterministic
                seed=42,          # Reproducible
            ),
        )
    )
    return kernel
```

**Why This Design**:
- **@st.cache_resource**: Initialize kernel once, reuse across sessions (cost savings)
- **GoogleAIChatCompletion**: SK's adapter for Google Gemini
- **KernelFunctionFromPrompt**: Turn prompt template into kernel function
- **Temperature=0.0**: Deterministic output (consistent rankings for same input)
- **Seed=42**: Reproducible results

#### **save_uploads(uploaded_files) → Tuple[str, List[str]]**

```python
def save_uploads(uploaded_files) -> tuple:
    tmp_dir = tempfile.mkdtemp()
    names = []
    for f in uploaded_files:
        safe_name = secure_filename(f.name)
        if not safe_name:
            continue
        dest = os.path.join(tmp_dir, safe_name)
        if not dest.startswith(tmp_dir):  # Path traversal protection
            logger.warning(f"Path traversal attempt blocked: {f.name}")
            continue
        with open(dest, "wb") as out:
            out.write(f.read())
        names.append(safe_name)
    return tmp_dir, names
```

**Security**:
- `secure_filename()`: Removes unsafe characters from filenames
- Path traversal check: `dest.startswith(tmp_dir)` prevents `../../../etc/passwd` attacks

#### **build_cvs(tmp_dir: str, filenames: list) → Tuple[List[CV], List[Tuple[str, str]]]**

```python
def build_cvs(tmp_dir: str, filenames: list) -> tuple:
    cvs, skipped = [], []
    for fname in sorted(filenames):
        path = os.path.join(tmp_dir, secure_filename(fname))
        try:
            text = _extract_text(path).strip()
        except Exception as e:
            skipped.append((fname, str(e)))
            continue
        if not text:
            skipped.append((fname, "No text extracted — scanned image?"))
            continue
        
        full_text = "\n\n---\n\n".join(_chunk_text(text))
        name = Path(fname).stem.replace("_", " ").replace("-", " ").title()
        cvs.append(CV(
            id=str(uuid.uuid4()),
            candidate_name=name,
            raw_text=full_text
        ))
    return cvs, skipped
```

**Process**:
1. Extract text from each file
2. Chunk text (and rejoin to preserve full context)
3. Create CV object with UUID
4. Infer name from filename
5. Return CVs and list of skipped files (with reasons)

#### **_invoke_with_retry(kernel, fn, retries=5, delay=30, **kwargs)**

```python
async def _invoke_with_retry(kernel, fn, retries=5, delay=30, **kwargs):
    for attempt in range(1, retries + 1):
        try:
            return await kernel.invoke(fn, **kwargs)
        except Exception as e:
            msg = str(e) + str(getattr(e, '__cause__', ''))
            if attempt < retries and any(x in msg for x in 
                ("503", "429", "UNAVAILABLE", "EXHAUSTED", "ServerError")):
                wait = delay * attempt
                st.toast(f"⏳ Service busy — retrying ({attempt}/{retries}) in {wait}s...")
                await asyncio.sleep(wait)
            else:
                raise
```

**Retry Strategy**:
- **Transient Errors**: 503 (Service Unavailable), 429 (Rate Limited), etc.
- **Backoff**: wait = 30s × attempt (exponential backoff)
- **Max Retries**: 5 attempts
- **Feedback**: Toast notification to user

**Why Retry**:
- Google API occasionally throttles or experiences temporary outages
- Exponential backoff avoids overwhelming the service
- Transparent to user (shows status)

#### **run_pipeline(kernel, cvs, jd, status_placeholder) → List[RankedResult]**

**Orchestration Steps**:

**Step 1: Embedding**
```python
status_placeholder.info("📄 Step 1/3: Extracting and embedding resumes...")
for i, cv in enumerate(cvs):
    cv.embedding = embed_text(cv.raw_text)  # Google Gemini API
    bar.progress((i + 1) / len(cvs))
```

**Step 2: Vector Store + Retrieval**
```python
status_placeholder.info("🔍 Step 2/3: Retrieving relevant candidates...")
vs = VectorStore()
for cv in cvs:
    vs.add(cv)  # Store embeddings

kernel.add_plugin(CVRetrievalPlugin(vs), plugin_name="retrieval")
retrieve_fn = kernel.get_function("retrieval", "retrieve")
retrieved = await _invoke_with_retry(
    kernel,
    retrieve_fn,
    query=jd.requirements,
    top_k=len(cvs)  # No filtering; all candidates
)
```

**Step 3: AI Ranking**
```python
status_placeholder.info("🤖 Step 3/3: AI ranking candidates...")
rank_fn = kernel.get_function("ranking", "rank_candidates")
result = await _invoke_with_retry(
    kernel,
    rank_fn,
    job_description=jd.requirements,
    retrieved_cvs=str(retrieved).strip()
)

# Parse JSON response
ranking_json = str(result).strip()
if ranking_json.startswith("```"):  # Handle markdown-wrapped JSON
    ranking_json = ranking_json.split("```")[1]
    if ranking_json.startswith("json"):
        ranking_json = ranking_json[4:]
items = json.loads(ranking_json)
```

**Step 4: Validation**
```python
results = []
for item in items:
    try:
        results.append(RankedResult(**item))
    except Exception as e:
        logger.warning(f"Failed to parse result: {e}")

status_placeholder.empty()
return validate_results(results)
```

**Why Semantic Kernel Here**:
- **Composition**: `kernel.invoke()` handles function lookup + invocation
- **Context Passing**: Variables flow between steps (jd.requirements → retrieval → ranking)
- **Service Management**: SK abstracts Google API details
- **Retry Logic**: Built into `_invoke_with_retry()` wrapper
- **Testability**: Kernel and plugins are mockable for unit tests

---

### 9. **plugins/cv_retrieval_plugin.py** — Kernel Plugin

**Location**: `/plugins/cv_retrieval_plugin.py`  
**Role**: Orchestration Layer - Plugin for Vector Search  
**Lines**: ~60  

**Purpose**:
Semantic Kernel plugin that retrieves semantically similar CVs. Bridges embeddings and vector search.

**Key Class: CVRetrievalPlugin**

```python
class CVRetrievalPlugin:
    def __init__(self, vector_store: VectorStore):
        self._vector_store = vector_store
    
    @kernel_function(description="Retrieve relevant CVs for a job query")
    def retrieve(
        self,
        query: Annotated[str, "The job requirements or query to search for"],
        top_k: Annotated[int, "Number of top CVs to retrieve"] = 3,
    ) -> str:
        # Embed query
        query_embedding = embed_query(query)
        
        # Search vector store
        similar_cvs = self._vector_store.search(query_embedding, top_k=top_k)
        
        # Format as string for prompt injection
        result_text = "RETRIEVED CANDIDATES:\n\n"
        for i, cv in enumerate(similar_cvs, 1):
            result_text += f"Candidate {i}: {cv.candidate_name}\n"
            result_text += f"ID: {cv.id}\n"
            result_text += f"Profile:\n{cv.raw_text}\n"
            result_text += "-" * 80 + "\n\n"
        
        return result_text
```

**Why @kernel_function Decorator**:
- Makes method discoverable by Semantic Kernel
- Automatic function registration
- `description` parameter appears in function catalog
- `Annotated` parameters create function schema

**Data Flow**:
1. SK calls `retrieve()` with `query=jd.requirements`, `top_k=len(cvs)`
2. Function embeds query (Google Gemini API)
3. Searches vector store (cosine similarity, NumPy)
4. Formats top-k CVs as readable string
5. SK injects returned string as {{$retrieved_cvs}} in ranking prompt

**Why This Pattern**:
- **Composability**: Easily add more plugins (e.g., skill extraction, background verification)
- **Testability**: Plugin is a pure class (easy to mock)
- **Reusability**: Could use different retrieval strategies

---

### 10. **utils/ui_components.py** — UI Rendering

**Location**: `/utils/ui_components.py`  
**Role**: Presentation Layer - Component Rendering  
**Lines**: ~200  

**Purpose**:
Encapsulates all Streamlit UI rendering logic and custom CSS styling.

**Key Functions**:

#### **apply_custom_styles()**
- Injects custom CSS for professional appearance
- Styles headers, input sections, result tables, badges
- Hides Streamlit footer and menu

#### **render_header()**
- Displays "📊 CV Ranking Agent" title
- Shows tagline: "AI-powered candidate evaluation and ranking system"

#### **render_file_upload_section(uploaded_files)**
- File uploader widget (PDF/DOCX)
- Shows count and file sizes for uploaded files
- Returns list of uploaded files

#### **render_job_details_section()**
- Text input for job title
- Large text area for job description
- Returns (role, jd_text)

#### **render_action_button(uploaded, role, jd_text)**
- "Analyze & Rank Candidates" button
- Only enabled if all required fields filled
- Shows status message when clicked

#### **render_stats_bar(results)**
- Displays summary stats: total candidates, top score, average score
- HTML badges with color coding

#### **render_results_table(results)**
- Streamlit DataFrame showing ranked candidates
- Columns: Rank, Name, Score, Experience, Strengths, Missing Skills
- Color-coded score badges (green/yellow/red)

#### **render_export_section(results, role)**
- CSV export button
- File format: `ranking_{role}_{timestamp}.csv`
- Columns: rank, candidate_name, score, reason, experience_years, key_strengths, skills_matched, skills_missing

**Why Separation from Business Logic**:
- UI changes don't affect pipeline logic
- Easy to swap UI frameworks (e.g., replace Streamlit with FastAPI + React)
- Testable without Streamlit context
- Reusable components

---

### 11. **utils/results_handler.py** — Results Display Orchestration

**Location**: `/utils/results_handler.py`  
**Role**: Presentation Layer - Display Logic  
**Lines**: ~20  

**Purpose**:
Coordinate display of ranking results via UI components.

**Key Function: show_results(results, role, cvs)**

```python
def show_results(results, role, cvs):
    render_stats_bar(results)
    render_results_table(results)
    render_export_section(results, role)
```

**Why This Abstraction**:
- Separates display orchestration from UI components
- Easy to add post-processing (e.g., filtering, sorting)
- Could extend with additional analytics

---

### 12. **requirements.txt** — Dependency Manifest

**Location**: `/requirements.txt`  
**Role**: Dependency Management  

```
semantic-kernel>=1.3.0          # Orchestration framework
google-generativeai>=0.8.0      # Google Gemini API (embeddings)
google-genai>=1.0.0             # Google Gemini API (LLM)
python-dotenv>=1.0.0            # Environment variable loading
pydantic>=2.0.0                 # Data validation
numpy>=1.26.0                   # Vector operations
pypdf>=4.0.0                    # PDF text extraction
python-docx>=1.1.0              # DOCX text extraction
streamlit>=1.35.0               # Web UI framework
pandas>=2.0.0                   # CSV export, data manipulation
```

---

## Plugin System & Semantic Kernel Integration

### Plugin Architecture

**Definition**: A Semantic Kernel plugin is a collection of functions exposed to the kernel for invocation.

**Two Types of Functions in SK**:

1. **Prompt Functions** (Templates)
   ```python
   KernelFunctionFromPrompt(
       prompt="You are {{$role}}. {{$task}}",
       prompt_execution_settings=GoogleAIPromptExecutionSettings(...)
   )
   ```
   - Variables: {{$role}}, {{$task}}
   - Executed by LLM
   - Deterministic with temperature=0.0

2. **Kernel Functions** (Python Methods)
   ```python
   @kernel_function(description="...")
   def my_function(self, param: str) -> str:
       return "result"
   ```
   - Pure Python logic
   - Called by kernel
   - Deterministic by nature

**How Plugins Register**:

```python
kernel.add_plugin(CVRetrievalPlugin(vs), plugin_name="retrieval")
# Now can call:
retrieve_fn = kernel.get_function("retrieval", "retrieve")
result = await kernel.invoke(retrieve_fn, query="...", top_k=10)
```

### Semantic Kernel Orchestration Flow in This Project

```
┌─────────────────────────────────────────────────────────┐
│         User Interaction (Streamlit)                    │
│  app.py → render UI → collect inputs → button click     │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│         Pipeline Initialization (pipeline.py)           │
│  kernel = get_kernel()                                  │
│  ├─ new Kernel()                                        │
│ ├─ add_service(GoogleAIChatCompletion)                  │
│ └─ add_function(KernelFunctionFromPrompt for ranking)  │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│         Step 1: Embedding (Direct Call)                │
│  for cv in cvs:                                         │
│      cv.embedding = embed_text(cv.raw_text)            │
│      └─ Direct call to embedder.py (not SK function)   │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│         Step 2: Vector Store Setup (Direct)            │
│  vs = VectorStore()                                     │
│  for cv in cvs:                                         │
│      vs.add(cv)  # Store embeddings                     │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│         Step 3a: Plugin Registration (SK)              │
│  kernel.add_plugin(CVRetrievalPlugin(vs), "retrieval") │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│         Step 3b: Plugin Invocation (SK)                │
│  retrieve_fn = kernel.get_function("retrieval", "retrieve") │
│  retrieved = await kernel.invoke(                       │
│      retrieve_fn,                                       │
│      query=jd.requirements,                             │
│      top_k=len(cvs)                                     │
│  )                                                      │
│  ├─ SK calls CVRetrievalPlugin.retrieve()               │
│  ├─ Function embeds query (direct call to embedder)    │
│  ├─ Function searches vector store                      │
│  └─ Returns formatted string                            │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│         Step 4: Ranking Prompt Invocation (SK)         │
│  rank_fn = kernel.get_function("ranking", "rank_candidates") │
│  result = await kernel.invoke(                          │
│      rank_fn,                                           │
│      job_description=jd.requirements,                   │
│      retrieved_cvs=str(retrieved)                       │
│  )                                                      │
│  ├─ SK performs template substitution:                 │
│  │  ├─ {{$job_description}} = jd.requirements          │
│  │  └─ {{$retrieved_cvs}} = retrieved string           │
│  ├─ SK calls GoogleAIChatCompletion                    │
│  ├─ Gemini receives full prompt (with substitutions)   │
│  ├─ Gemini returns JSON array                          │
│  └─ SK passes back to application                      │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│         Post-Processing (No SK Involvement)            │
│  Parse JSON → Validate → Return RankedResult[]         │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│         Display Results (Streamlit)                     │
│  show_results(results, role, cvs)                       │
│  ├─ render_stats_bar()                                  │
│  ├─ render_results_table()                              │
│  └─ render_export_section()                             │
└─────────────────────────────────────────────────────────┘
```

### Why Semantic Kernel Excels Here

| Aspect | Without SK | With SK |
|--------|-----------|---------|
| **Function Registration** | Manual function discovery | @kernel_function decorator + automatic registration |
| **Template Management** | String formatting + variables | {{$var}} substitution + validation |
| **Prompt Execution** | Manual API calls + error handling | `kernel.invoke()` + built-in retry |
| **Service Switching** | Rewrite all API calls | Change service configuration |
| **Composition** | Chain functions manually | SK handles context passing |
| **Type Safety** | Loose typing | Annotated function parameters |

---

## Vector Search & Semantic Matching

### Embedding Space Concept

**What is an Embedding?**
- A fixed-size numerical representation of text
- Captures semantic meaning in continuous space
- Produced by neural networks (in this case, Google Gemini)
- **Dimensionality**: 768 dimensions

**Visualization (Conceptual)**:
```
Text: "Python developer with AWS experience"
         ↓
    [0.142, -0.531, 0.893, 0.002, ..., 0.667]  ← 768 floats
         ↓
    Points in 768-dimensional space
    (can't visualize, but imagine 3D)
```

**Key Property: Semantic Similarity**:
- Texts with similar meaning have embeddings close in space
- Measured by cosine similarity: values closer → higher match

### Google Gemini Embedding Model

**Model**: `models/gemini-embedding-001`

**Task-Type Optimization**:

| Task Type | Purpose | Configuration |
|-----------|---------|---|
| `RETRIEVAL_DOCUMENT` | Embedding a document to be retrieved | Optimizes for document representation |
| `RETRIEVAL_QUERY` | Embedding a query to search | Optimizes query for matching documents |

**Why Separate**:
- Documents and queries are different distributions
- Optimizing separately improves retrieval accuracy
- Example: Query "Python" embeds differently from document containing "Python" as context

**API Call Example**:
```python
# Document embedding
result = client.models.embed_content(
    model="models/gemini-embedding-001",
    contents="I have 5 years of Python experience using Django and FastAPI",
    config=EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
)
# Returns embedding optimized for document search

# Query embedding
result = client.models.embed_content(
    model="models/gemini-embedding-001",
    contents="Python backend developer with web framework experience",
    config=EmbedContentConfig(task_type="RETRIEVAL_QUERY")
)
# Returns embedding optimized for query matching
```

### Vector Search Algorithm

**Cosine Similarity in vector_store.py**:

```python
def search(self, query_embedding, top_k):
    query_vec = np.array(query_embedding, dtype=np.float32)
    
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
    
    # Get top-k indices
    indices = np.argsort(similarities)[::-1][:top_k]
    return [self._cvs[i] for i in indices]
```

**Time Complexity**: O(n × 768) where n = number of CVs
- Each dot-product: 768 multiplications
- Total for n vectors: n × 768 operations
- Acceptable for n < 10,000

**Why Not Use Database Index**:
- For small n (recruitment: 10-500 CVs), brute-force is sufficient
- No setup overhead (no DB required)
- Fully in-memory (sub-millisecond latency)

---

## AI Ranking Pipeline

### Ranking Prompt Design

**Template Structure**:

```
You are a STRICT technical recruiter...

JOB DESCRIPTION:
{{$job_description}}

CANDIDATE PROFILES:
{{$retrieved_cvs}}

EVALUATION CRITERIA:
1. REQUIRED SKILLS MATCHING
2. EXPERIENCE REQUIREMENTS
3. SCORING RULES (0-100 scale)
4. MISSING SKILLS PENALTY

Return ONLY JSON array: [...]
```

**Design Principles**:

1. **Role Framing** ("STRICT technical recruiter")
   - Establishes context and tone
   - Encourages thorough evaluation
   - Justifies harsh grading

2. **Explicit Criteria**
   - Clear rubric (reduces hallucination)
   - Specific score ranges with justification
   - Penalizes missing skills

3. **Structured Output**
   - Specifies JSON format
   - Lists all required fields
   - Example provided

4. **Emphasis on Hardness**
   - "Only truly exceptional candidates should score above 85"
   - "BE HARSH"
   - Prevents score inflation

### Scoring Rules

| Score | Interpretation | Criteria |
|-------|----------------|----------|
| 90–100 | **Exceptional** | Exceeds ALL requirements, has bonus skills, perfect fit |
| 75–89 | **Strong** | Meets ALL required skills, meets experience requirement |
| 60–74 | **Adequate** | Meets MOST required skills (80%+), close to experience |
| 40–59 | **Weak** | Meets SOME required skills (50–79%), gaps in experience |
| 0–39 | **Poor** | Missing critical skills (<50%), insufficient experience |

**Penalty System**:
- Each missing required skill: −10 to −15 points
- Each missing critical skill (mentioned multiple times): −20 points
- Lack of required certifications: −10 points

### LLM Configuration

**Model**: `gemini-2.5-flash`
- Fast and cost-effective
- Excellent at JSON generation
- Good reasoning for recruitment scenarios

**Settings**:
```python
GoogleAIPromptExecutionSettings(
    temperature=0.0,  # Deterministic (no randomness)
    seed=42,          # Reproducible across runs
)
```

**Why Temperature=0.0**:
- Recruitment decisions must be reproducible
- Same CV + Job Description should yield same score
- No creativity/variance needed (deterministic reasoning)

**Why Seed=42**:
- Ensures reproducibility even if model implementation changes
- Allows auditing/verification
- Good for compliance (HR decisions need justification)

---

## Design Decisions & Tool Selection Rationale

### 1. Semantic Kernel over Direct LLM API Calls

**Alternative**: Directly call `client.models.generate_content()`

**Why SK is Superior**:

| Aspect | Direct API | Semantic Kernel |
|--------|-----------|-----------------|
| **Prompt Management** | Manual string formatting | Template variables {{$var}} |
| **Plugin System** | No composition pattern | Plugins as first-class functions |
| **Retries** | Manual try-catch | Built-in retry with backoff |
| **Service Switching** | Rewrite all calls | Change configuration |
| **Testing** | Requires mocking API | Mock kernel functions |
| **Scalability** | Repeats error handling in each call | Centralized error management |

**SK provides structure and scalability** for complex workflows.

### 2. Google Gemini for Both Embeddings & LLM

**Alternative**: Use OpenAI for embeddings + Gemini for LLM

**Why Single Provider**:

| Aspect | Single Provider | Multiple Providers |
|--------|---|---|
| **Setup** | 1 API key | 2 API keys + configs |
| **Latency** | Consistent | Varies by provider |
| **Cost** | Easier to track | Complex billing |
| **Error Handling** | Unified | Provider-specific logic |
| **Debugging** | Simpler request tracing | Harder to trace across services |

**Unified provider simplifies operations and debugging.**

### 3. In-Memory Vector Store vs. External DB

**Alternative**: PostgreSQL + pgvector, Pinecone, Weaviate

**Why In-Memory**:

| Aspect | In-Memory | External DB |
|--------|---|---|
| **Setup** | None | Infrastructure required |
| **Latency** | <1ms | 10–50ms |
| **Cost** | Free | Monthly subscription |
| **Use Case** | <1000 vectors | 1M+ vectors |
| **Persistence** | Session-only (ephemeral) | Permanent |

**For recruitment (10–500 CVs per evaluation), in-memory is optimal.**

### 4. Streamlit for Web Framework

**Alternative**: FastAPI + React, Django, Flask

**Why Streamlit**:

| Aspect | Streamlit | Alternatives |
|--------|---|---|
| **Dev Speed** | Rapid prototyping (no UI boilerplate) | Traditional frameworks slower |
| **State Management** | Built-in (session_state) | Manual Redux/Context |
| **Async Support** | Native asyncio integration | Requires async frameworks |
| **Deployment** | Streamlit Cloud (1-click) | Heroku, AWS, GCP |
| **Data Apps** | Optimized for | Generic frameworks |

**Streamlit excels at rapid data app development.**

### 5. Pydantic for Data Validation

**Alternative**: TypedDict, dataclasses

**Why Pydantic**:

| Aspect | Pydantic | Alternatives |
|--------|---|---|
| **Runtime Validation** | Automatic type checking | Manual validation |
| **JSON Serialization** | Built-in .model_dump_json() | Manual serialization |
| **Error Messages** | Detailed, field-specific | Generic |
| **Ecosystem** | FastAPI integration | Not as widely adopted |

**Pydantic enables robust data handling with minimal code.**

### 6. NumPy for Vector Operations

**Alternative**: SciPy, cupy, TensorFlow

**Why NumPy**:

| Aspect | NumPy | Alternatives |
|--------|---|---|
| **Simplicity** | Simple API for dot products | Overkill (cupy) or heavy (TF) |
| **Speed** | C-optimized operations | Slower (pure Python) |
| **Dependencies** | Minimal | Heavy (CUDA for GPU) |
| **Use Case** | <10K vectors | Large-scale ML |

**NumPy is perfect for this scale and use case.**

---

## API Methods & External Integrations

### Google Gemini API

**Endpoint 1: Embeddings**

```
POST https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent

Request:
{
  "model": "models/gemini-embedding-001",
  "contents": [{"parts": [{"text": "Your text here"}]}],
  "embed_content_config": {
    "task_type": "RETRIEVAL_DOCUMENT"  // or RETRIEVAL_QUERY
  }
}

Response:
{
  "embeddings": [
    {
      "values": [0.142, -0.531, 0.893, ..., 0.667]  // 768 floats
    }
  ]
}
```

**Client Library**: `google-genai`
```python
from google import genai
client = genai.Client(api_key=GOOGLE_API_KEY)
result = client.models.embed_content(
    model="models/gemini-embedding-001",
    contents=text,
    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
)
embeddings = result.embeddings[0].values  # List[float]
```

**Endpoint 2: LLM Inference (via Semantic Kernel)**

```
POST https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent

Request:
{
  "model": "gemini-2.5-flash",
  "contents": [{"parts": [{"text": "Your prompt here"}]}],
  "generation_config": {
    "temperature": 0.0,
    "seed": 42
  }
}

Response:
{
  "candidates": [
    {
      "content": {
        "parts": [{"text": "[{\"rank\": 1, ...}]"}]
      }
    }
  ]
}
```

**Client Library**: `google-generativeai`
```python
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.google.google_ai import GoogleAIChatCompletion

kernel = Kernel()
kernel.add_service(GoogleAIChatCompletion(
    gemini_model_id="gemini-2.5-flash",
    api_key=GOOGLE_API_KEY
))
```

### Rate Limiting & Error Handling

**Common Status Codes**:
- **429**: Rate Limited (too many requests)
- **503**: Service Unavailable
- **500**: Internal Server Error

**Retry Strategy** (in `_invoke_with_retry`):
```python
if "503" in error or "429" in error:
    wait = 30 * attempt
    await asyncio.sleep(wait)  # Exponential backoff
    retry()
else:
    raise
```

---

## Complete System Integration Summary

### Data Flow Summary

```
User Inputs (CVs + Job Description)
        ↓
┌─────────────────────────────────────────┐
│ FILE PROCESSING LAYER                   │
│ • Save uploads to temp directory        │
│ • Extract text (PDF/DOCX)               │
│ • Chunk text (800 chars, 100 overlap)   │
│ • Create CV objects                     │
│ Output: [CV objects]                    │
└─────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────┐
│ EMBEDDING LAYER                         │
│ • Embed each CV (Google Gemini)         │
│ • Task type: RETRIEVAL_DOCUMENT         │
│ • Output: CV objects + embeddings       │
└─────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────┐
│ VECTOR STORE LAYER                      │
│ • Store CV embeddings in-memory         │
│ • Embed job description (RETRIEVAL_QUERY) │
│ • Retrieve top-k similar CVs            │
│ • Output: Formatted candidate profiles  │
└─────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────┐
│ RANKING LAYER                           │
│ • Invoke ranking prompt via SK          │
│ • Inject job description + CVs          │
│ • Google Gemini evaluates candidates    │
│ • Returns JSON rankings                 │
└─────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────┐
│ VALIDATION LAYER                        │
│ • Validate JSON schema                  │
│ • Check score ranges                    │
│ • Remove invalid entries                │
│ • Sort by rank                          │
│ Output: [RankedResult]                  │
└─────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────┐
│ DISPLAY LAYER                           │
│ • Show stats bar                        │
│ • Render results table                  │
│ • Provide CSV export                    │
│ • Store in session history              │
│ Output: Web UI + CSV download           │
└─────────────────────────────────────────┘
```

### Why This Architecture is Optimal

1. **Separation of Concerns**: Each layer has a single responsibility
2. **Modularity**: Easy to replace components (e.g., swap in PostgreSQL)
3. **Testability**: Each layer can be unit-tested independently
4. **Scalability**: Add features without refactoring existing code
5. **Maintainability**: Clear data flow, minimal coupling
6. **Cost-Effective**: Minimal infrastructure (Google API only)
7. **Fast**: Optimized vector search + deterministic LLM

---

## Conclusion

The **CV Ranking Agent** demonstrates a modern AI application architecture:

- **Semantic Kernel** orchestrates multi-step workflows elegantly
- **Google Gemini** provides unified embeddings + LLM reasoning
- **Vector search** enables semantic matching at scale
- **Streamlit** enables rapid, iterative development
- **Pydantic** ensures data quality and type safety
- **NumPy** optimizes vector operations for speed

This design allows recruiters to evaluate hundreds of candidates in minutes with consistent, defensible scoring—making hiring faster, fairer, and more data-driven.

---

**Generated**: May 5, 2026  
**Project**: CV Ranking Agent  
**Framework**: Microsoft Semantic Kernel  
**LLM Provider**: Google Gemini  
**Language**: Python 3.10+
