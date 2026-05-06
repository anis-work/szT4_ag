"""Business logic and pipeline functions for CV Ranking Agent."""

import asyncio
import json
import tempfile
import os
import uuid
import logging
from pathlib import Path

import streamlit as st
from werkzeug.utils import secure_filename

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.google.google_ai import GoogleAIChatCompletion
from semantic_kernel.connectors.ai.google.google_ai.google_ai_prompt_execution_settings import GoogleAIPromptExecutionSettings
from semantic_kernel.functions import KernelFunctionFromPrompt

from config import GOOGLE_API_KEY, GEMINI_MODEL
from models import CV, JobDescription, RankedResult
from embedder import embed_text
from vector_store import VectorStore
from plugins.cv_retrieval_plugin import CVRetrievalPlugin
from pdf_loader import _extract_text, _chunk_text, extract_experience_years, extract_candidate_name
from taxonomy.resolver import build_enriched_jd
from validator import validate_results

logger = logging.getLogger(__name__)

RANKING_PROMPT = """You are a STRICT technical recruiter. Evaluate each candidate against the job description.

JOB DESCRIPTION:
{{$job_description}}

CANDIDATE PROFILES:
{{$retrieved_cvs}}

MANDATORY EVALUATION PROCESS - follow every step in order:

STEP 1 - EXTRACT REQUIRED SKILLS FROM JD
List every required skill, technology, certification, and experience level explicitly mentioned.
Count them. This is your TOTAL_SKILLS number.

STEP 2 - FOR EACH CANDIDATE, DO A SKILL-BY-SKILL CHECKLIST
Go through every required skill one by one.
Mark each as PRESENT (explicit evidence in CV) or MISSING (not mentioned or no evidence).
Count PRESENT skills - this is skills_matched.
List MISSING skills in skills_missing.
Do NOT guess or infer - only count what is explicitly stated in the CV.

STEP 3 - COMPUTE SCORE FROM THE CHECKLIST
Base score = (skills_matched / TOTAL_SKILLS) * 100
Adjustments:
  Experience meets requirement: +0 to +5
  Experience below requirement: -10 to -20
  Each missing CRITICAL skill (mentioned 2+ times in JD): additional -5
  Exceptional bonus skills beyond JD: +0 to +5
Final score must be mathematically consistent with skills_matched / TOTAL_SKILLS ratio.

STEP 4 - WRITE THE REASON
Must reference exact skills_matched count and TOTAL_SKILLS.
Format: Matched X of Y required skills. Has: [matched skills]. Missing: [missing skills]. [1 sentence on experience fit].
Do NOT write a reason that contradicts the score or skills_matched count.

STEP 5 - EXPERIENCE
Use PRE_EXTRACTED_EXPERIENCE if present in the candidate profile - this is authoritative.
If not present, infer from date ranges in the CV text.

STEP 6 - CANDIDATE NAME
Use the VERIFIED_NAME tag if present in the candidate profile.
Otherwise use the exact name from the candidate profile header.
Do NOT use the filename or modify the name.

OUTPUT FORMAT:
Return ONLY a valid JSON array with no markdown, no explanation, no code fences.
Each object must have these exact fields:
  rank (integer, 1 = best),
  candidate_name (string),
  score (integer 0-100),
  reason (string),
  experience_years (float),
  key_strengths (string),
  skills_matched (integer),
  skills_missing (string, comma-separated)
"""


@st.cache_resource
def get_kernel() -> Kernel:
    """Initialize and return a Semantic Kernel instance with ranking function."""
    kernel = Kernel()
    kernel.add_service(GoogleAIChatCompletion(gemini_model_id=GEMINI_MODEL, api_key=GOOGLE_API_KEY))
    kernel.add_function(plugin_name="ranking", function=KernelFunctionFromPrompt(
        function_name="rank_candidates",
        plugin_name="ranking",
        prompt=RANKING_PROMPT,
        prompt_execution_settings=GoogleAIPromptExecutionSettings(temperature=0.0, seed=42),
    ))
    return kernel


def save_uploads(uploaded_files) -> tuple:
    """Save uploaded files to temporary directory and return paths."""
    tmp_dir = tempfile.mkdtemp()
    names = []
    for f in uploaded_files:
        safe_name = secure_filename(f.name)
        if not safe_name:
            continue
        dest = os.path.join(tmp_dir, safe_name)
        if not dest.startswith(tmp_dir):
            logger.warning(f"Path traversal attempt blocked: {f.name}")
            continue
        with open(dest, "wb") as out:
            out.write(f.read())
        names.append(safe_name)
    return tmp_dir, names


def build_cvs(tmp_dir: str, filenames: list) -> tuple:
    """Extract text from files and build CV objects."""
    cvs, skipped = [], []
    for fname in sorted(filenames):
        safe_fname = secure_filename(fname)
        path = os.path.join(tmp_dir, safe_fname)
        if not path.startswith(tmp_dir):
            logger.warning(f"Path traversal attempt blocked: {fname}")
            continue
        try:
            text = _extract_text(path).strip()
        except Exception as e:
            logger.exception(f"Failed to extract text from {fname}")
            skipped.append((fname, str(e)))
            continue
        if not text:
            ext = Path(fname).suffix.lower()
            reason = (
                "No text could be extracted — DOCX may use text boxes or an unsupported layout."
                if ext in (".docx", ".doc")
                else "No text could be extracted — PDF may be a scanned image."
            )
            skipped.append((fname, reason))
            continue
        full_text = "\n\n---\n\n".join(_chunk_text(text))
        # Use name extracted from CV text; fall back to cleaned filename
        filename_name = Path(fname).stem.replace("_", " ").replace("-", " ").title()
        name = extract_candidate_name(text) or filename_name
        exp_years = extract_experience_years(text)
        cvs.append(CV(
            id=str(uuid.uuid4()),
            candidate_name=name,
            raw_text=full_text,
            experience_years=exp_years
        ))
    return cvs, skipped


async def _invoke_with_retry(kernel, fn, retries=5, delay=30, **kwargs):
    """Invoke kernel function with retry logic for transient errors."""
    for attempt in range(1, retries + 1):
        try:
            return await kernel.invoke(fn, **kwargs)
        except Exception as e:
            msg = str(e) + str(getattr(e, '__cause__', ''))
            if attempt < retries and any(x in msg for x in ("503", "429", "UNAVAILABLE", "EXHAUSTED", "ServerError")):
                wait = delay * attempt
                st.toast(f"⏳ Service busy — retrying ({attempt}/{retries}) in {wait}s...")
                await asyncio.sleep(wait)
            else:
                raise


async def run_pipeline(kernel: Kernel, cvs: list, jd: JobDescription, status_placeholder) -> list:
    """Run the complete CV ranking pipeline."""
    # Step 1: Embed resumes
    status_placeholder.info("📄 Step 1/3: Extracting and embedding resumes...")
    bar = st.progress(0)
    for i, cv in enumerate(cvs):
        cv.embedding = embed_text(cv.raw_text)
        bar.progress((i + 1) / len(cvs))
    bar.empty()

    # Step 2: Retrieve relevant candidates
    status_placeholder.info("🔍 Step 2/3: Retrieving relevant candidates...")
    # Enrich JD with Sulzer taxonomy implied skills (zero API calls)
    enriched_requirements = build_enriched_jd(jd.role, jd.requirements)
    vs = VectorStore()
    for cv in cvs:
        vs.add(cv)

    kernel.add_plugin(CVRetrievalPlugin(vs), plugin_name="retrieval")
    retrieve_fn = kernel.get_function(plugin_name="retrieval", function_name="retrieve")
    retrieved = await _invoke_with_retry(kernel, retrieve_fn, query=enriched_requirements, top_k=len(cvs))

    # Inject pre-extracted experience and verified name into retrieved text
    retrieved_str = str(retrieved).strip()
    for cv in cvs:
        old = f"{cv.candidate_name}\n"
        annotations = []
        if cv.experience_years is not None:
            annotations.append(f"PRE_EXTRACTED_EXPERIENCE: {cv.experience_years} years")
        annotations.append(f"VERIFIED_NAME: {cv.candidate_name}")
        new = f"{cv.candidate_name} [{', '.join(annotations)}]\n"
        retrieved_str = retrieved_str.replace(old, new, 1)

    # Step 3: Rank candidates
    status_placeholder.info("🤖 Step 3/3: AI ranking candidates...")
    rank_fn = kernel.get_function(plugin_name="ranking", function_name="rank_candidates")
    result = await _invoke_with_retry(kernel, rank_fn,
                                      job_description=enriched_requirements,
                                      retrieved_cvs=retrieved_str)
    ranking_json = str(result).strip()
    if ranking_json.startswith("```"):
        ranking_json = ranking_json.split("```")[1]
        if ranking_json.startswith("json"):
            ranking_json = ranking_json[4:]
        ranking_json = ranking_json.strip()

    items = json.loads(ranking_json)
    verified_names = {cv.candidate_name.lower(): cv.candidate_name for cv in cvs}

    results = []
    for item in items:
        try:
            result = RankedResult(**item)
            matched = verified_names.get(result.candidate_name.lower())
            if matched:
                result = result.model_copy(update={"candidate_name": matched})
            results.append(result)
        except Exception as e:
            logger.warning(f"Failed to parse result item: {e}")
    
    status_placeholder.empty()
    return validate_results(results)
