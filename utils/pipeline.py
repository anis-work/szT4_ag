"""Business logic and pipeline functions for CV Ranking Agent."""

import asyncio
import json
import tempfile
import os
import uuid
import logging
import re
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
from pdf_loader import _extract_text, _chunk_text, extract_experience_years
from validator import validate_results

logger = logging.getLogger(__name__)

RANKING_PROMPT = """You are a fair and balanced technical recruiter. Evaluate each candidate holistically against the job description.

JOB DESCRIPTION:
{{$job_description}}

CANDIDATE PROFILES:
{{$retrieved_cvs}}

EVALUATION APPROACH:

1. SKILLS ASSESSMENT
   - Identify required skills from the JD
   - Check each candidate for explicit evidence of each skill
   - Count matched skills as skills_matched, list missing ones in skills_missing

2. HOLISTIC SCORING (do not use a rigid formula)
   - 90-100: Exceptional fit — meets all requirements, has bonus skills
   - 75-89: Strong fit — meets most required skills and experience
   - 60-74: Good fit — meets majority of skills, minor gaps
   - 40-59: Partial fit — meets some skills, notable gaps
   - 0-39: Weak fit — missing critical skills or significantly under-experienced
   - Consider both skill breadth AND depth, not just keyword count
   - A candidate strong in 7 of 8 critical skills is better than one weak in all 10

3. EXPERIENCE
   - Use PRE_EXTRACTED_EXPERIENCE if present in the candidate profile — this is authoritative
   - If not present, infer from date ranges in the CV text

4. CANDIDATE NAME
   - Use the VERIFIED_NAME tag if present in the candidate profile
   - Otherwise use the exact name from the candidate profile header
   - Do NOT use the filename or modify the name

5. REASON
   - Write 2-3 clear sentences covering: skills matched, skills missing, experience fit
   - Be specific — name actual skills, not vague statements

OUTPUT FORMAT:
Return ONLY a valid JSON array with no markdown, no explanation, no code fences.
Each object must have these exact fields:
  rank (integer, 1 = best),
  cv_id (string, copy exactly from the CV_ID tag in the candidate profile),
  candidate_name (string, copy exactly from VERIFIED_NAME tag),
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
        # Use filename as candidate name - simple and reliable
        stem = Path(fname).stem
        stem = re.sub(r'^(Naukri|LinkedIn|Indeed|Resume|CV)[_\-\s]+', '', stem, flags=re.IGNORECASE)
        stem = re.sub(r'[_\-\s]*\d+[Yy][_\-\s]*\d*[Mm]?\s*$', '', stem)  # strip "6y_4m", "3Y 0M"
        stem = re.sub(r'[_\-\s]*\d+$', '', stem)  # strip trailing numbers
        name = stem.replace('_', ' ').replace('-', ' ').strip().title()
        exp_years = extract_experience_years(text)
        cvs.append(CV(
            id=str(uuid.uuid4()),
            candidate_name=name,
            raw_text=full_text,
            experience_years=exp_years
        ))
    # Deduplicate CVs by candidate name (keep first occurrence)
    seen_names: set = set()
    unique_cvs = []
    for cv in cvs:
        key = cv.candidate_name.lower().strip()
        if key not in seen_names:
            seen_names.add(key)
            unique_cvs.append(cv)
        else:
            logger.warning(f"Duplicate CV skipped: {cv.candidate_name}")
    return unique_cvs, skipped


async def _invoke_with_retry(kernel, fn, retries=5, delay=15, **kwargs):
    """Invoke kernel function with retry logic for transient errors."""
    from semantic_kernel.exceptions.kernel_exceptions import KernelInvokeException
    from semantic_kernel.exceptions.function_exceptions import FunctionExecutionException

    for attempt in range(1, retries + 1):
        try:
            return await kernel.invoke(fn, **kwargs)
        except (KernelInvokeException, FunctionExecutionException, Exception) as e:
            # Walk full exception cause chain to find transient signal
            cause = e
            msg_parts = []
            while cause:
                msg_parts.append(str(cause))
                cause = getattr(cause, '__cause__', None) or getattr(cause, '__context__', None)
            full_msg = ' '.join(msg_parts)

            is_transient = any(x in full_msg for x in (
                "503", "429", "UNAVAILABLE", "EXHAUSTED", "ServerError",
                "high demand", "temporarily", "overloaded", "quota"
            ))

            if attempt < retries and is_transient:
                wait = delay * attempt
                st.toast(f"⏳ Google API busy — retrying ({attempt}/{retries}) in {wait}s...")
                logger.warning(f"Transient error on attempt {attempt}, retrying in {wait}s: {str(e)[:120]}")
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
    vs = VectorStore()
    for cv in cvs:
        vs.add(cv)

    kernel.add_plugin(CVRetrievalPlugin(vs), plugin_name="retrieval")
    retrieve_fn = kernel.get_function(plugin_name="retrieval", function_name="retrieve")
    retrieved = await _invoke_with_retry(kernel, retrieve_fn, query=jd.requirements, top_k=len(cvs))

    # Inject pre-extracted experience, verified name, and a stable CV_ID into retrieved text
    retrieved_str = str(retrieved).strip()
    cv_id_map = {}  # cv_id -> CV object for authoritative name lookup
    for cv in cvs:
        cv_id_map[cv.id] = cv
        old = f"{cv.candidate_name}\n"
        annotations = [f"CV_ID: {cv.id}", f"VERIFIED_NAME: {cv.candidate_name}"]
        if cv.experience_years is not None:
            annotations.append(f"PRE_EXTRACTED_EXPERIENCE: {cv.experience_years} years")
        new = f"{cv.candidate_name} [{', '.join(annotations)}]\n"
        retrieved_str = retrieved_str.replace(old, new, 1)

    # Step 3: Rank candidates
    status_placeholder.info("🤖 Step 3/3: AI ranking candidates...")
    rank_fn = kernel.get_function(plugin_name="ranking", function_name="rank_candidates")
    result = await _invoke_with_retry(kernel, rank_fn,
                                      job_description=jd.requirements,
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
            # Use cv_id for authoritative name — most reliable
            if result.cv_id and result.cv_id in cv_id_map:
                result = result.model_copy(update={"candidate_name": cv_id_map[result.cv_id].candidate_name})
            else:
                # Fallback: fuzzy match by lowercased name
                matched = verified_names.get(result.candidate_name.lower())
                if matched:
                    result = result.model_copy(update={"candidate_name": matched})
            results.append(result)
        except Exception as e:
            logger.warning(f"Failed to parse result item: {e}")
    
    status_placeholder.empty()
    return validate_results(results)
