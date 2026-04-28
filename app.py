"""Streamlit web UI for CV Ranking Agent."""

import asyncio
import json
import tempfile
import os
import uuid
from pathlib import Path

import streamlit as st
import pandas as pd

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.google.google_ai import GoogleAIChatCompletion
from semantic_kernel.connectors.ai.google.google_ai.google_ai_prompt_execution_settings import GoogleAIPromptExecutionSettings
from semantic_kernel.functions import KernelFunctionFromPrompt

from config import GOOGLE_API_KEY, GEMINI_MODEL
from models import CV, JobDescription, RankedResult
from embedder import embed_text
from vector_store import VectorStore
from plugins.cv_retrieval_plugin import CVRetrievalPlugin
from pdf_loader import _extract_text, _chunk_text
from validator import validate_results

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CV Ranking Agent",
    page_icon="assets/icon.png" if os.path.exists("assets/icon.png") else None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Header bar */
    .app-header {
        background: #1a2332;
        color: #ffffff;
        padding: 1.2rem 2rem;
        border-radius: 6px;
        margin-bottom: 1.5rem;
    }
    .app-header h1 {
        margin: 0;
        font-size: 1.5rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .app-header p {
        margin: 0.2rem 0 0 0;
        font-size: 0.85rem;
        color: #a0aec0;
    }

    /* Section labels */
    .section-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        margin-bottom: 0.4rem;
    }

    /* Candidate card */
    .candidate-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #1a2332;
        border-radius: 6px;
        padding: 1.1rem 1.4rem;
        margin-bottom: 0.75rem;
    }
    .candidate-card.top { border-left-color: #0f766e; }
    .candidate-card.mid { border-left-color: #b45309; }
    .candidate-card.low { border-left-color: #b91c1c; }

    .candidate-rank {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #64748b;
        margin-bottom: 0.2rem;
    }
    .candidate-name {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a2332;
        margin-bottom: 0.5rem;
    }
    .candidate-reason {
        font-size: 0.88rem;
        color: #475569;
        line-height: 1.55;
    }
    .score-badge {
        font-size: 1.8rem;
        font-weight: 700;
        line-height: 1;
    }
    .score-label {
        font-size: 0.75rem;
        color: #94a3b8;
        margin-top: 0.1rem;
    }

    /* Summary bar */
    .summary-bar {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 0.9rem 1.2rem;
        margin-bottom: 1.2rem;
        display: flex;
        gap: 2rem;
    }
    .summary-stat { text-align: center; }
    .summary-stat-value { font-size: 1.4rem; font-weight: 700; color: #1a2332; }
    .summary-stat-label { font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; }

    /* File list */
    .file-item {
        font-size: 0.85rem;
        color: #334155;
        padding: 0.3rem 0;
        border-bottom: 1px solid #f1f5f9;
    }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

RANKING_PROMPT = """You are an expert recruiter evaluating candidates for an open role.

JOB DESCRIPTION:
{{$job_description}}

CANDIDATE PROFILES:
{{$retrieved_cvs}}

INSTRUCTIONS:
- Evaluate each candidate strictly against the job description above
- Score 0-100 based on skills match, experience level, and role alignment
- Provide a concise reason (2-3 sentences) per candidate
- Rank from best fit (1) to worst fit

Return ONLY a valid JSON array, no markdown, no preamble:
[
  {
    "rank": 1,
    "candidate_name": "Name",
    "score": 95,
    "reason": "Brief reason."
  }
]"""


# ── Core helpers ──────────────────────────────────────────────────────────────
@st.cache_resource
def get_kernel() -> Kernel:
    kernel = Kernel()
    kernel.add_service(GoogleAIChatCompletion(gemini_model_id=GEMINI_MODEL, api_key=GOOGLE_API_KEY))
    kernel.add_function(plugin_name="ranking", function=KernelFunctionFromPrompt(
        function_name="rank_candidates",
        plugin_name="ranking",
        prompt=RANKING_PROMPT,
        prompt_execution_settings=GoogleAIPromptExecutionSettings(temperature=0.0),
    ))
    return kernel


def save_uploads(uploaded_files) -> tuple:
    tmp_dir = tempfile.mkdtemp()
    names = []
    for f in uploaded_files:
        dest = os.path.join(tmp_dir, f.name)
        with open(dest, "wb") as out:
            out.write(f.read())
        names.append(f.name)
    return tmp_dir, names


def build_cvs(tmp_dir: str, filenames: list) -> tuple:
    cvs, skipped = [], []
    for fname in sorted(filenames):
        path = os.path.join(tmp_dir, fname)
        try:
            text = _extract_text(path).strip()
        except Exception as e:
            skipped.append((fname, str(e)))
            continue
        if not text:
            skipped.append((fname, "No text could be extracted — file may be a scanned image."))
            continue
        full_text = "\n\n---\n\n".join(_chunk_text(text))
        name = Path(fname).stem.replace("_", " ").replace("-", " ").title()
        cvs.append(CV(id=str(uuid.uuid4()), candidate_name=name, raw_text=full_text))
    return cvs, skipped


async def _invoke_with_retry(kernel, fn, retries=3, delay=20, **kwargs):
    for attempt in range(1, retries + 1):
        try:
            return await kernel.invoke(fn, **kwargs)
        except Exception as e:
            msg = str(e)
            if attempt < retries and any(x in msg for x in ("503", "429", "UNAVAILABLE", "EXHAUSTED")):
                st.toast(f"Service busy — retrying ({attempt}/{retries})...")
                await asyncio.sleep(delay)
            else:
                raise


async def run_pipeline(kernel: Kernel, cvs: list, jd: JobDescription) -> list:
    bar = st.progress(0, text="Analysing resumes...")
    for i, cv in enumerate(cvs):
        cv.embedding = embed_text(cv.raw_text)
        bar.progress((i + 1) / len(cvs), text=f"Processing: {cv.candidate_name}")
    bar.empty()

    vs = VectorStore()
    for cv in cvs:
        vs.add(cv)

    kernel.add_plugin(CVRetrievalPlugin(vs), plugin_name="retrieval")

    retrieve_fn = kernel.get_function(plugin_name="retrieval", function_name="retrieve")
    retrieved = await _invoke_with_retry(kernel, retrieve_fn, query=jd.requirements, top_k=len(cvs))

    rank_fn = kernel.get_function(plugin_name="ranking", function_name="rank_candidates")
    result = await _invoke_with_retry(kernel, rank_fn,
                                      job_description=jd.requirements,
                                      retrieved_cvs=str(retrieved).strip())
    ranking_json = str(result).strip()
    if ranking_json.startswith("```"):
        ranking_json = ranking_json.split("```")[1]
        if ranking_json.startswith("json"):
            ranking_json = ranking_json[4:]
        ranking_json = ranking_json.strip()

    items = json.loads(ranking_json)
    results = []
    for item in items:
        try:
            results.append(RankedResult(**item))
        except Exception:
            pass
    return validate_results(results)


# ── Results display ───────────────────────────────────────────────────────────
def show_results(results: list, role: str):
    avg_score = round(sum(r.score for r in results) / len(results))
    top = results[0]

    # Summary bar
    st.markdown(f"""
    <div class="summary-bar">
        <div class="summary-stat">
            <div class="summary-stat-value">{len(results)}</div>
            <div class="summary-stat-label">Candidates Ranked</div>
        </div>
        <div class="summary-stat">
            <div class="summary-stat-value">{top.score}</div>
            <div class="summary-stat-label">Top Score</div>
        </div>
        <div class="summary-stat">
            <div class="summary-stat-value">{avg_score}</div>
            <div class="summary-stat-label">Average Score</div>
        </div>
        <div class="summary-stat">
            <div class="summary-stat-value">{sum(1 for r in results if r.score >= 75)}</div>
            <div class="summary-stat-label">Strong Matches</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Candidate cards
    for r in results:
        tier = "top" if r.score >= 75 else "mid" if r.score >= 50 else "low"
        score_color = "#0f766e" if r.score >= 75 else "#b45309" if r.score >= 50 else "#b91c1c"
        col1, col2 = st.columns([5, 1])
        with col1:
            st.markdown(f"""
            <div class="candidate-card {tier}">
                <div class="candidate-rank">Rank {r.rank}</div>
                <div class="candidate-name">{r.candidate_name}</div>
                <div class="candidate-reason">{r.reason}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style="text-align:center; padding-top:1.2rem;">
                <div class="score-badge" style="color:{score_color}">{r.score}</div>
                <div class="score-label">out of 100</div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(r.score / 100)

    # Download
    st.divider()
    df = pd.DataFrame([
        {"Rank": r.rank, "Candidate": r.candidate_name, "Score": r.score, "Reason": r.reason}
        for r in results
    ])
    col_a, col_b = st.columns([3, 1])
    with col_b:
        st.download_button(
            label="Download as CSV",
            data=df.to_csv(index=False),
            file_name=f"rankings_{role.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_a:
        st.caption(f"Results for: {role}  |  {len(results)} candidates evaluated")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>CV Ranking Agent</h1>
    <p>AI-powered candidate shortlisting — upload resumes, define the role, get ranked results.</p>
</div>
""", unsafe_allow_html=True)

# ── Input layout ──────────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="section-label">Candidate Resumes</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload PDF or DOCX files",
        type=["pdf", "docx", "doc"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Accepted formats: PDF, DOCX. Name files as firstname_lastname for best results.",
    )
    if uploaded:
        st.markdown(f'<div class="section-label" style="margin-top:0.8rem">{len(uploaded)} file(s) uploaded</div>', unsafe_allow_html=True)
        for f in uploaded:
            size_kb = round(f.size / 1024, 1)
            st.markdown(f'<div class="file-item">{f.name} &nbsp;<span style="color:#94a3b8">({size_kb} KB)</span></div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="border:1px dashed #cbd5e1;border-radius:6px;padding:2rem;text-align:center;color:#94a3b8;font-size:0.85rem">'
            'Drag and drop resume files here, or click Browse above.'
            '</div>',
            unsafe_allow_html=True,
        )

with right:
    st.markdown('<div class="section-label">Position Details</div>', unsafe_allow_html=True)
    role = st.text_input(
        "Job Title",
        placeholder="e.g. Senior Python Backend Engineer",
        label_visibility="collapsed",
    )
    st.markdown('<div class="section-label" style="margin-top:0.6rem">Job Description</div>', unsafe_allow_html=True)
    jd_text = st.text_area(
        "Job Description",
        height=240,
        placeholder="Paste the full job description here — required skills, years of experience, responsibilities, and any preferred qualifications.",
        label_visibility="collapsed",
    )

st.divider()

# ── Action ────────────────────────────────────────────────────────────────────
ready = bool(uploaded and role.strip() and jd_text.strip())

col_btn, col_hint = st.columns([1, 3])
with col_btn:
    run = st.button("Run Analysis", type="primary", use_container_width=True, disabled=not ready)
with col_hint:
    if not uploaded:
        st.caption("Upload at least one resume to continue.")
    elif not role.strip() or not jd_text.strip():
        st.caption("Enter the job title and description to continue.")
    else:
        st.caption(f"{len(uploaded)} resume(s) ready — click Run Analysis to begin.")

# ── Pipeline ──────────────────────────────────────────────────────────────────
if run:
    with st.spinner("Running analysis..."):
        try:
            kernel = get_kernel()
            tmp_dir, filenames = save_uploads(uploaded)
            cvs, skipped = build_cvs(tmp_dir, filenames)

            if skipped:
                for fname, reason in skipped:
                    st.warning(f"Skipped **{fname}**: {reason}")

            if not cvs:
                st.error("No readable resumes found. Ensure files are text-based PDFs or DOCX.")
                st.stop()

            jd = JobDescription(role=role.strip(), requirements=jd_text.strip())
            results = asyncio.run(run_pipeline(kernel, cvs, jd))

            st.divider()
            st.markdown('<div class="section-label">Ranking Results</div>', unsafe_allow_html=True)
            show_results(results, role.strip())

        except Exception as e:
            st.error(f"An error occurred: {e}")
