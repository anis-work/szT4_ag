"""Streamlit web UI for CV Ranking Agent — designed for HR use."""

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
from semantic_kernel.functions import KernelFunctionFromPrompt

from config import GOOGLE_API_KEY, GEMINI_MODEL
from models import CV, JobDescription, RankedResult
from embedder import embed_text
from vector_store import VectorStore
from plugins.cv_retrieval_plugin import CVRetrievalPlugin
from pdf_loader import _extract_text, _chunk_text
from validator import validate_results

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="CV Ranking Agent", page_icon="🎯", layout="wide")

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


def build_cvs(tmp_dir: str, filenames: list) -> list:
    cvs, skipped = [], []
    for fname in sorted(filenames):
        path = os.path.join(tmp_dir, fname)
        try:
            text = _extract_text(path).strip()
        except Exception as e:
            skipped.append((fname, str(e)))
            continue
        if not text:
            skipped.append((fname, "No text found — may be a scanned/image PDF"))
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
                st.toast(f"API busy — retrying in {delay}s… (attempt {attempt}/{retries})")
                await asyncio.sleep(delay)
            else:
                raise


async def run_pipeline(kernel: Kernel, cvs: list, jd: JobDescription) -> list:
    # Embed
    bar = st.progress(0, text="Embedding resumes…")
    for i, cv in enumerate(cvs):
        cv.embedding = embed_text(cv.raw_text)
        bar.progress((i + 1) / len(cvs), text=f"Embedding: {cv.candidate_name}")
    bar.empty()

    # Vector store
    vs = VectorStore()
    for cv in cvs:
        vs.add(cv)

    # Retrieval plugin — replace if already registered
    kernel.plugins["retrieval"] = CVRetrievalPlugin(vs) if "retrieval" in kernel.plugins \
        else kernel.add_plugin(CVRetrievalPlugin(vs), plugin_name="retrieval") or kernel.plugins["retrieval"]

    # Retrieve
    retrieve_fn = kernel.get_function(plugin_name="retrieval", function_name="retrieve")
    retrieved = await _invoke_with_retry(kernel, retrieve_fn, query=jd.requirements, top_k=len(cvs))

    # Rank
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
def show_results(results: list):
    st.success(f"✅ Ranked {len(results)} candidates")

    for r in results:
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(r.rank, f"**#{r.rank}**")
        color = "#2ecc71" if r.score >= 75 else "#f39c12" if r.score >= 50 else "#e74c3c"
        with st.container(border=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"### {medal} {r.candidate_name}")
                st.caption(r.reason)
            with col2:
                st.markdown(
                    f"<div style='text-align:center;padding-top:8px'>"
                    f"<span style='font-size:2rem;font-weight:700;color:{color}'>{r.score}</span>"
                    f"<span style='color:gray'>/100</span></div>",
                    unsafe_allow_html=True,
                )
                st.progress(r.score / 100)

    df = pd.DataFrame([
        {"Rank": r.rank, "Candidate": r.candidate_name, "Score": r.score, "Reason": r.reason}
        for r in results
    ])
    st.download_button(
        "⬇️ Download results as CSV",
        data=df.to_csv(index=False),
        file_name="cv_ranking_results.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🎯 CV Ranking Agent")
st.caption("Upload resumes, describe the role — get an AI-powered shortlist in seconds.")
st.divider()

left, right = st.columns(2, gap="large")

with left:
    st.subheader("📄 Resumes")
    uploaded = st.file_uploader(
        "Upload PDF or DOCX files",
        type=["pdf", "docx", "doc"],
        accept_multiple_files=True,
        help="Name files as firstname_lastname.pdf — the filename becomes the candidate name.",
    )
    if uploaded:
        st.caption(f"{len(uploaded)} file(s) ready")
        for f in uploaded:
            st.markdown(f"- `{f.name}`")

with right:
    st.subheader("📋 Job Description")
    role = st.text_input("Job Title", placeholder="e.g. Senior Python Backend Engineer")
    jd_text = st.text_area(
        "Full job description",
        height=260,
        placeholder="Paste the complete JD here — required skills, experience, responsibilities…",
    )

st.divider()

ready = bool(uploaded and role.strip() and jd_text.strip())
run = st.button("🚀 Rank Candidates", type="primary", use_container_width=True, disabled=not ready)

if not uploaded:
    st.info("👆 Upload at least one resume to get started.")
elif not role.strip() or not jd_text.strip():
    st.info("👆 Enter the job title and description to continue.")

if run:
    with st.spinner("Running pipeline…"):
        try:
            kernel = get_kernel()
            tmp_dir, filenames = save_uploads(uploaded)
            cvs, skipped = build_cvs(tmp_dir, filenames)

            if skipped:
                for fname, reason in skipped:
                    st.warning(f"⚠️ Skipped **{fname}**: {reason}")

            if not cvs:
                st.error("No readable resumes found. Use text-based PDFs or DOCX files.")
                st.stop()

            jd = JobDescription(role=role.strip(), requirements=jd_text.strip())
            results = asyncio.run(run_pipeline(kernel, cvs, jd))

            st.divider()
            st.subheader("🏆 Rankings")
            show_results(results)

        except Exception as e:
            st.error(f"Error: {e}")
