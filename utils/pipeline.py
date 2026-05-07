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

RANKING_PROMPT = """You are a strict and precise technical recruiter. Your job is to score and rank candidates against a job description with accuracy and consistency.

JOB DESCRIPTION:
{{$job_description}}

CANDIDATE PROFILES:
{{$retrieved_cvs}}

SCORING RULES — follow these exactly, no exceptions:

STEP 1 — IDENTIFY REQUIRED SKILLS
List every required skill, tool, and experience from the JD. Count the total. This is your baseline.

STEP 2 — SCORE EACH CANDIDATE (0–100 integer)
   - 90-100: Exceeds ALL requirements, has bonus skills
   - 75-89: Meets ALL required skills, meets experience requirement
   - 60-74: Meets MOST required skills (80%+)
   - 40-59: Meets SOME required skills (50-79%)
   - 0-39: Missing critical skills (<50%)

STEP 3 — RANK
Rank strictly by score descending. If two candidates have the same score, rank the one with more relevant experience higher.

STEP 4 — EXPERIENCE
Always use PRE_EXTRACTED_EXPERIENCE from the candidate profile if present — do not override it.
If not present, calculate from date ranges in the CV text.

STEP 5 — CANDIDATE NAME
Copy exactly from VERIFIED_NAME tag. Do not modify, shorten, or add titles.

STEP 6 — REASON
Write exactly 2–3 sentences: (1) what skills matched, (2) what is missing, (3) experience fit.
Be specific — name actual skills and tools, not vague statements like "good fit".

OUTPUT FORMAT:
Return ONLY a valid JSON array. No markdown, no explanation, no code fences.
Each object must have these exact fields:
  rank (integer, 1 = best),
  cv_id (string, copy exactly from CV_ID tag),
  candidate_name (string, copy exactly from VERIFIED_NAME tag),
  score (integer 0-100),
  reason (string),
  experience_years (float, from PRE_EXTRACTED_EXPERIENCE if available),
  key_strengths (string, comma-separated list of matched skills),
  skills_matched (integer),
  skills_missing (string, comma-separated list of missing skills)
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
        # Strip job title suffixes — match partial words too (e.g. "Transitin", "Prog")
        name = re.sub(r'\s+(?:Senior|Junior|Lead|Head|Director|Manager|Analyst|Engineer|Developer|Consultant|Coordinator|Specialist|Associate|Intern|Officer|Transition|Transit|Program|Prog|PMO)\w*.*$', '', name, flags=re.IGNORECASE).strip()
        exp_years = extract_experience_years(text)
        # Use deterministic ID based on filename for consistency across runs
        cv_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, fname))
        # Embed metadata directly into raw_text so it's always present in retrieved output
        annotations = f"CV_ID: {cv_id} | VERIFIED_NAME: {name}"
        if exp_years is not None:
            annotations += f" | PRE_EXTRACTED_EXPERIENCE: {exp_years} years"
        full_text = f"[{annotations}]\n{full_text}"
        cvs.append(CV(
            id=cv_id,
            candidate_name=name,
            raw_text=full_text,
            experience_years=exp_years
        ))
    # Deduplicate CVs by candidate name (keep first occurrence)
    seen_names: set = set()
    unique_cvs = []
    for cv in cvs:
        # Normalize name: remove job titles and extra whitespace for comparison
        normalized = re.sub(r'\s+(Transition|Senior|Junior|Lead|Manager|Analyst|Engineer|Developer|Consultant|Coordinator|Specialist|Associate).*$', '', cv.candidate_name, flags=re.IGNORECASE).strip().lower()
        if normalized not in seen_names:
            seen_names.add(normalized)
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
    retrieved_str = str(retrieved).strip()
    cv_id_map = {cv.id: cv for cv in cvs}

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

    # Unescape HTML entities the LLM may have introduced
    import html as html_lib
    ranking_json = html_lib.unescape(ranking_json)

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
