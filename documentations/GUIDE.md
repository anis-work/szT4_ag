# CV Ranking Agent — User Guide

An AI-powered agent that reads resumes (PDF or DOCX), ranks them against a job description, and outputs a scored shortlist. Built with Semantic Kernel + Google Gemini.

---

## Prerequisites

- Python 3.10+
- A Google AI API key → get one free at https://aistudio.google.com/app/apikey

---

## 1. Setup (one-time)

### Install dependencies

```bash
cd cv_ranker
pip install -r requirements.txt
```

### Configure your API key

Open `.env` and set your key:

```
GOOGLE_API_KEY=your_actual_key_here
GEMINI_MODEL=gemini-2.5-flash
EMBEDDING_MODEL=models/gemini-embedding-001
TOP_K=3
```

> `TOP_K` controls how many CVs are retrieved from the vector store before ranking. When you use `--folder` mode it auto-scales to the total number of CVs you upload, so every resume gets ranked.


---
## 2. Prepare your files

### Resumes

- Supported formats: **PDF**, **DOCX**, **DOC**
- Put all resume files in a single folder, e.g. `resumes/`
- **Name each file after the candidate** — the agent uses the filename as the candidate name in results

```
resumes/
  john_doe.pdf
  jane_smith.docx
  alex_kumar.pdf
  priya_nair.docx
```

> `john_doe.pdf` → displayed as **John Doe**
> `jane_smith.docx` → displayed as **Jane Smith**

### Job Description

Create a plain `.txt` file with the full job description. Be as detailed as possible — the more context, the better the ranking.

```
resumes/job_requirements.txt
```

Example contents:

```
POSITION: Senior Python Backend Engineer

REQUIREMENTS:
- 4+ years of Python backend development
- Strong FastAPI or Django experience
- PostgreSQL database design and optimization
- AWS cloud experience (EC2, RDS, Lambda)
- Docker and Kubernetes
- Microservices architecture
- Ability to mentor junior engineers

PREFERRED:
- Kubernetes experience
- CI/CD pipeline knowledge
- Distributed systems background
```

---

## 3. Run the agent

### With your own resumes and JD (recommended)

```bash
python main.py --folder resumes --role "Senior Python Backend Engineer" --requirements resumes/job_requirements.txt
```

| Argument | Description |
|---|---|
| `--folder` | Path to folder containing your PDF/DOCX resumes |
| `--role` | Job title (shown in logs and output) |
| `--requirements` | Path to your `.txt` job description file |

### With built-in sample data (quick test)

```bash
python main.py
```

Runs with 5 pre-loaded fictional candidates and a sample Senior Backend Engineer JD — useful to verify everything is working before using real data.

---

## 4. Reading the output

```
==========================================================================================
                                    CV RANKING RESULTS
==========================================================================================

[Rank 1] David Brown
Score: 97/100
Reason: Exceptional system design and distributed systems expertise at principal level.
        9+ years experience, multi-region AWS, strong Python background.
------------------------------------------------------------------------------------------

[Rank 2] Alice Johnson
Score: 92/100
Reason: Direct match — 5 years FastAPI and PostgreSQL, AWS, Docker, Kubernetes,
        and mentoring experience. Ready to contribute immediately.
------------------------------------------------------------------------------------------

[Rank 3] Bob Smith
Score: 35/100
Reason: Falls short on experience (3.5 vs 4+ years required). PostgreSQL and AWS
        described as basic. No system design or mentoring evidence.
------------------------------------------------------------------------------------------
```

- **Rank** — ordered from best fit to worst
- **Score** — 0–100, how well the candidate matches the JD
- **Reason** — Gemini's explanation based purely on your JD

---

## 5. Folder structure reference

```
cv_ranker/
├── resumes/                  ← put your PDFs/DOCX here
│   ├── john_doe.pdf
│   ├── jane_smith.docx
│   └── job_requirements.txt  ← your JD text file
├── plugins/
│   ├── cv_ingestion_plugin.py   (loads files via SK)
│   ├── cv_retrieval_plugin.py   (semantic search via SK)
│   └── cv_ranking_plugin.py     (ranks via SK)
├── main.py                   ← entry point
├── pdf_loader.py             ← PDF/DOCX extraction + chunking
├── embedder.py               ← Gemini embeddings
├── vector_store.py           ← in-memory cosine similarity search
├── models.py                 ← Pydantic data models
├── validator.py              ← result validation
├── config.py                 ← loads .env
├── .env                      ← your API key and model config
└── requirements.txt
```

---

## 6. Tips

**More candidates = better ranking**
The agent ranks all CVs you provide. There is no upper limit — drop 20 resumes in the folder and all 20 get ranked.

**JD quality matters**
A detailed JD with specific skills, years of experience, and responsibilities gives Gemini more signal to rank accurately. A one-liner JD will produce generic results.

**File naming**
Use `firstname_lastname.pdf` format. Underscores and hyphens are converted to spaces and title-cased automatically.

**Scanned PDFs won't work**
The agent extracts text from PDFs. If a resume is a scanned image (no selectable text), it will be skipped. Use text-based PDFs or DOCX files.

**Rate limits**
The free tier of Google AI has per-minute and per-day limits. If you hit a `429` error, wait 30–60 seconds and retry. For large batches, consider upgrading to a paid API tier.

---

## 7. Changing the model

Edit `.env`:

```
GEMINI_MODEL=gemini-2.5-flash     # faster, cheaper
GEMINI_MODEL=gemini-2.5-pro       # more accurate, slower
```

---

## 8. Example end-to-end workflow

```bash
# 1. Put resumes in the folder
cp ~/Downloads/*.pdf resumes/

# 2. Write your JD
notepad resumes/job_requirements.txt

# 3. Run
python main.py --folder resumes --role "Data Engineer" --requirements resumes/job_requirements.txt

# 4. Review ranked output in terminal
```

That's it.
