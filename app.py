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
    .app-header {
        background: #1a2332;
        color: #ffffff;
        padding: 1.2rem 2rem;
        border-radius: 6px;
        margin-bottom: 1.5rem;
    }
    .app-header h1 { margin: 0; font-size: 1.5rem; font-weight: 600; letter-spacing: 0.02em; }
    .app-header p { margin: 0.2rem 0 0 0; font-size: 0.85rem; color: #a0aec0; }

    .section-label {
        font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.08em; color: #64748b; margin-bottom: 0.4rem;
    }

    .candidate-card {
        background: #ffffff; border: 1px solid #e2e8f0;
        border-left: 4px solid #1a2332; border-radius: 6px;
        padding: 1.1rem 1.4rem; margin-bottom: 0.75rem;
    }
    .candidate-card.top { border-left-color: #0f766e; }
    .candidate-card.mid { border-left-color: #b45309; }
    .candidate-card.low { border-left-color: #b91c1c; }

    .candidate-rank { font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #64748b; margin-bottom: 0.2rem; }
    .candidate-name { font-size: 1.1rem; font-weight: 600; color: #1a2332; margin-bottom: 0.5rem; }
    .candidate-reason { font-size: 0.88rem; color: #475569; line-height: 1.55; }

    .score-badge { font-size: 1.8rem; font-weight: 700; line-height: 1; }
    .score-label { font-size: 0.75rem; color: #94a3b8; margin-top: 0.1rem; }

    .summary-bar {
        background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px;
        padding: 0.9rem 1.2rem; margin-bottom: 1.2rem; display: flex; gap: 2rem;
    }
    .summary-stat { text-align: center; }
    .summary-stat-value { font-size: 1.4rem; font-weight: 700; color: #1a2332; }
    .summary-stat-label { font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; }

    .file-item { font-size: 0.85rem; color: #334155; padding: 0.3rem 0; border-bottom: 1px solid #f1f5f9; }

    .tag { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.75rem; margin: 0.1rem; }
    .tag-green { background: #dcfce7; color: #166534; }
    .tag-red { background: #fee2e2; color: #991b1b; }
    .tag-orange { background: #ffedd5; color: #9a3412; }
    .tag-blue { background: #dbeafe; color: #1e40af; }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

RANKING_PROMPT = """You are an expert recruiter and HR analyst evaluating candidates for an open role.

JOB DESCRIPTION:
{{$job_description}}

CANDIDATE PROFILES:
{{$retrieved_cvs}}

INSTRUCTIONS:
Evaluate each candidate and return a JSON array. For every candidate provide ALL of the following fields:

1. rank - integer, 1 = best fit
2. candidate_name - string
3. score - integer 0-100, overall fit score
4. reason - 2-3 sentence overall assessment

5. ai_verdict - one of: "Human", "Likely AI", "Uncertain"
   Signals of AI-generated CV: overly generic buzzword-heavy language, no specific personal anecdotes or numbers, perfectly uniform sentence structure, templated sections with no authentic voice, suspiciously comprehensive skill lists with no depth.
6. ai_confidence - integer 0-100, confidence in the ai_verdict
7. ai_reason - 1 sentence explaining the ai_verdict

8. red_flags - array of strings, each a concise red flag (e.g. "Gap in employment 2021-2022", "Vague job titles", "No quantified achievements", "Dates inconsistent"). Empty array if none.

9. skills_match - array of strings, key JD requirements this candidate clearly meets
10. skills_missing - array of strings, key JD requirements this candidate lacks or does not explicitly demonstrate

11. seniority_fit - one of: "Underqualified", "Good Fit", "Overqualified" followed by a dash and a brief note (e.g. "Overqualified - 12 years vs 3-5 required")

Return ONLY a valid JSON array, no markdown, no preamble:
[
  {
    "rank": 1,
    "candidate_name": "Name",
    "score": 85,
    "reason": "Overall assessment.",
    "ai_verdict": "Human",
    "ai_confidence": 80,
    "ai_reason": "Contains specific project details and authentic personal voice.",
    "red_flags": ["No quantified achievements"],
    "skills_match": ["Advanced Excel", "Power BI", "SOP development"],
    "skills_missing": ["Power Automate", "RAID log ownership"],
    "seniority_fit": "Good Fit - 4 years aligns with 3-5 year requirement"
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
def _ai_badge(verdict: str, confidence: int) -> str:
    color = {"Human": "#dcfce7", "Likely AI": "#fee2e2", "Uncertain": "#fef9c3"}.get(verdict, "#f1f5f9")
    text_color = {"Human": "#166534", "Likely AI": "#991b1b", "Uncertain": "#854d0e"}.get(verdict, "#334155")
    return (
        f'<span style="background:{color};color:{text_color};padding:0.2rem 0.6rem;'
        f'border-radius:4px;font-size:0.75rem;font-weight:600;">'
        f'{verdict} ({confidence}%)</span>'
    )


def _seniority_badge(seniority_fit: str) -> str:
    label = seniority_fit.split(" - ")[0] if " - " in seniority_fit else seniority_fit
    color = {"Good Fit": "#dcfce7", "Overqualified": "#ffedd5", "Underqualified": "#fee2e2"}.get(label, "#f1f5f9")
    text_color = {"Good Fit": "#166534", "Overqualified": "#9a3412", "Underqualified": "#991b1b"}.get(label, "#334155")
    return (
        f'<span style="background:{color};color:{text_color};padding:0.2rem 0.6rem;'
        f'border-radius:4px;font-size:0.75rem;font-weight:600;">{seniority_fit}</span>'
    )


def show_results(results: list, role: str):
    avg_score = round(sum(r.score for r in results) / len(results))
    top = results[0]

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
        <div class="summary-stat">
            <div class="summary-stat-value">{sum(1 for r in results if r.ai_verdict == "Likely AI")}</div>
            <div class="summary-stat-label">Likely AI CVs</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    for r in results:
        tier = "top" if r.score >= 75 else "mid" if r.score >= 50 else "low"
        score_color = "#0f766e" if r.score >= 75 else "#b45309" if r.score >= 50 else "#b91c1c"

        col_main, col_score = st.columns([5, 1])

        with col_main:
            # Header row: name + badges
            st.markdown(f"""
            <div class="candidate-card {tier}">
                <div class="candidate-rank">Rank {r.rank}</div>
                <div class="candidate-name">{r.candidate_name}
                    &nbsp;{_ai_badge(r.ai_verdict, r.ai_confidence)}
                    &nbsp;{_seniority_badge(r.seniority_fit)}
                </div>
                <div class="candidate-reason">{r.reason}</div>
            </div>
            """, unsafe_allow_html=True)

            # Expandable details
            with st.expander("Details — Skills, Red Flags & AI Analysis"):
                d1, d2, d3 = st.columns(3)

                with d1:
                    st.markdown("**Skills Match**")
                    if r.skills_match:
                        tags = "".join(f'<span class="tag tag-green">✓ {s}</span>' for s in r.skills_match)
                        st.markdown(tags, unsafe_allow_html=True)
                    else:
                        st.caption("None identified")

                    st.markdown("**Skills Missing**")
                    if r.skills_missing:
                        tags = "".join(f'<span class="tag tag-red">✗ {s}</span>' for s in r.skills_missing)
                        st.markdown(tags, unsafe_allow_html=True)
                    else:
                        st.caption("No gaps identified")

                with d2:
                    st.markdown("**Red Flags**")
                    if r.red_flags:
                        for flag in r.red_flags:
                            st.markdown(f'<span class="tag tag-orange">⚠ {flag}</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="tag tag-green">✓ No red flags</span>', unsafe_allow_html=True)

                with d3:
                    st.markdown("**AI Detection**")
                    st.markdown(_ai_badge(r.ai_verdict, r.ai_confidence), unsafe_allow_html=True)
                    st.caption(r.ai_reason)

        with col_score:
            st.markdown(f"""
            <div style="text-align:center; padding-top:1.2rem;">
                <div class="score-badge" style="color:{score_color}">{r.score}</div>
                <div class="score-label">out of 100</div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(r.score / 100)

    # Download
    st.divider()
    df = pd.DataFrame([{
        "Rank": r.rank,
        "Candidate": r.candidate_name,
        "Score": r.score,
        "Reason": r.reason,
        "Seniority Fit": r.seniority_fit,
        "AI Verdict": r.ai_verdict,
        "AI Confidence": r.ai_confidence,
        "AI Reason": r.ai_reason,
        "Red Flags": " | ".join(r.red_flags),
        "Skills Match": " | ".join(r.skills_match),
        "Skills Missing": " | ".join(r.skills_missing),
    } for r in results])

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
