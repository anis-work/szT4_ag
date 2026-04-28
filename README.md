# CV Ranking Agent

An AI-powered candidate shortlisting tool built with **Streamlit**, **Semantic Kernel**, and **Google Gemini**. Upload resumes, paste a job description, and get a ranked, scored shortlist in under a minute.

---

## What it does

- Accepts PDF and DOCX resumes — any number of files
- Reads and understands the full text of each resume
- Compares every candidate against your job description using semantic AI
- Returns a ranked list with a score (0–100) and a written reason per candidate
- Lets you download the results as a CSV

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
cp .env.example .env
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

## License

MIT
