"""Business logic and pipeline functions for CV Ranking Agent."""

import asyncio
import json
import html as html_lib
import tempfile
import os
import time
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

RANKING_PROMPT = """You are a principal-level technical recruiter with deep domain knowledge across industries. \
Your judgment must go far beyond keyword matching — you are evaluating genuine fit, \
transferable expertise, and real evidence of skill from each candidate's work history.

═══════════════════════════════════════════════════════
ROLE & JOB DESCRIPTION
═══════════════════════════════════════════════════════
{{$job_description}}

═══════════════════════════════════════════════════════
CANDIDATE PROFILES
═══════════════════════════════════════════════════════
{{$retrieved_cvs}}

═══════════════════════════════════════════════════════
EVALUATION INSTRUCTIONS — follow every step in order
═══════════════════════════════════════════════════════

─── STEP 1 · PARSE THE JD INTO A WEIGHTED REQUIREMENT SET ───────────────────
Classify every distinct requirement from the JD into one of three tiers:

  CRITICAL [C] — The role cannot function without this. Missing it is a
                   dealbreaker unless a strong semantic equivalent exists.
                   Identify these from words like "required", "must have",
                   "essential", or from the core function of the role itself.

  IMPORTANT [I] — Strongly preferred. Absence hurts the score significantly
                   but does not eliminate the candidate.
                   Identify these from words like "preferred", "strong plus",
                   "ideally", or from tools/methods central to day-to-day work.

  BONUS [B] — Nice-to-have. Adds value but is not expected.
                   Identify these from words like "familiarity", "exposure",
                   "beneficial", or additional certifications not listed as mandatory.

─── STEP 2 · EVALUATE EACH CANDIDATE — SEMANTIC, NOT LEXICAL ────────────────
For every requirement identified in Step 1, search the candidate's profile for
EVIDENCE. Apply the following semantic reasoning rules:

  FULL CREDIT — Exact skill OR a well-established equivalent demonstrated in
  real work experience or a shipped project. Examples of semantic equivalence:
    • "PostgreSQL" or "MySQL" or "Oracle" → satisfies "relational database"
    • "AWS" or "GCP" or "Azure" → satisfies "cloud platform"
    • "Pandas" or "Polars" or "dplyr" → satisfies "data manipulation"
    • "led a 6-person cross-functional team" → satisfies "team leadership"
    • "reduced customer churn by 18% via segmentation model" → satisfies "data-driven decision making"
    • "ANSYS CFX" or "OpenFOAM" or "Fluent" → satisfies "CFD simulation tool"
    • "centrifugal pump impeller design" → satisfies "pump hydraulics design"

  PARTIAL CREDIT (50–75 %) — Adjacent skill with clear, plausible transferability.
    • Experience in a closely related tool, method, or domain where the core
      competency transfers with modest upskilling.

  NO CREDIT — Fundamentally different domain despite surface similarity.
    • The skill appears on the CV but context shows it is unrelated to the requirement.
    • Vague self-assessments with no supporting evidence:
      "familiar with", "exposure to", "basic knowledge of" — no project cited.

  IMPLICIT CREDIT — Skills logically implied by demonstrated adjacent work.
  Do not penalise a candidate for not listing skills a competent practitioner
  would obviously possess.

─── STEP 3 · EXPERIENCE GATE (NON-NEGOTIABLE) ───────────────────────────────
  a. Extract the minimum years of experience stated in the JD (e.g. "5+ years",
     "minimum 3 years"). If none is stated, skip this gate entirely.
  b. Use PRE_EXTRACTED_EXPERIENCE from the candidate's tag if available.
  c. If the candidate's experience is below the JD minimum:
       → Their final score is capped at 60. No exceptions.
  d. This cap is applied AFTER the raw score is calculated — not instead of it.
     A well-qualified but under-experienced candidate still outranks a poorly-
     qualified under-experienced one within the cap.

─── STEP 4 · CALCULATE RAW SCORE (0–100) ────────────────────────────────────
Start at 0. Add points strictly on the basis of EVIDENCE found in Step 2.

  Per CRITICAL skill with FULL evidence in work history:   +15 pts
  Per CRITICAL skill with PARTIAL evidence:                +8 pts
  Per CRITICAL skill with NO evidence:                     +0 pts
  Per IMPORTANT skill with FULL evidence:                  +8 pts
  Per IMPORTANT skill with PARTIAL evidence:               +4 pts
  Per BONUS skill with evidence:                           +3 pts
  Experience meets JD minimum:                             +10 pts
  Experience exceeds JD minimum by 3+ years:               +5 pts
  Outstanding achievement relevant to the role
    (quantified business impact, publication, award,
     patent, top ranking):                                 +5 pts (max once)

  Hard cap: 100. Apply the experience gate from Step 3 last.

  Score bands for reference:
    90–100  All critical skills evidenced, exceeds experience requirement
    75–89   All critical skills evidenced, meets experience requirement
    60–74   Most critical skills evidenced, minor gaps or at experience floor
    40–59   Some critical skills evidenced, notable gaps
    0–39    Missing most critical skills

─── STEP 5 · RANK ───────────────────────────────────────────────────────────
  Sort all candidates by final score descending.
  Tiebreaker: higher experience_years wins.
  Assign rank 1 to the best candidate.

─── STEP 6 · WRITE THE REASON ───────────────────────────────────────────────
Exactly 3 sentences per candidate — no more, no fewer:
  Sentence 1: Which CRITICAL skills were evidenced and from which specific
              role, project, or achievement in their CV (cite it by name).
  Sentence 2: Which critical or important skills are absent or only partially
              evidenced, and why that matters for this role.
  Sentence 3: Experience verdict — state their extracted years vs the JD
              requirement and explicitly note if the gate was applied.

─── STEP 7 · VALIDATE BEFORE OUTPUTTING ─────────────────────────────────────
  • Every cv_id must be copied verbatim from the CV_ID tag in the profile.
  • Every candidate_name must be copied verbatim from the VERIFIED_NAME tag.
  • Scores must be integers in [0, 100].
  • ranks must be unique positive integers starting at 1.
  • skills_matched must equal the integer count of C+I skills with full evidence.
  • skills_missing must list every C and I skill with no or partial evidence.

═══════════════════════════════════════════════════════
OUTPUT — STRICT FORMAT
═══════════════════════════════════════════════════════
Return ONLY a valid JSON array. Absolutely no markdown, no code fences, \
no prose before or after. Any deviation will break the parser.

Schema (each element):
{
  "rank": <integer, 1 = best>,
  "cv_id": <string, verbatim from CV_ID tag>,
  "candidate_name": <string, verbatim from VERIFIED_NAME tag>,
  "score": <integer, 0-100>,
  "reason": <string, exactly 3 sentences>,
  "experience_years": <float, from PRE_EXTRACTED_EXPERIENCE or your best estimate>,
  "key_strengths": <string, comma-separated matched skills with evidence source>,
  "skills_matched": <integer, count of C+I skills with full evidence>,
  "skills_missing": <string, comma-separated missing or partial C+I skills>
}"""


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


def _safe_experience(text: str) -> float:
    """Wrapper around extract_experience_years with fallback for edge cases."""
    exp = extract_experience_years(text)
    if exp is None or exp <= 0 or exp > 40:
        m = re.search(r'\b(\d{1,2})\+?\s*years?\b', text.lower())
        if m:
            return float(m.group(1))
        m2 = re.search(r'(\d+)\s*years?\s*(\d+)\s*months?', text.lower())
        if m2:
            return round(float(m2.group(1)) + float(m2.group(2)) / 12, 1)
        return 0.0
    return round(exp, 1)


def save_uploads(uploaded_files) -> tuple:
    """Save uploaded files to temporary directory and return paths."""
    tmp_dir = os.path.realpath(tempfile.mkdtemp())
    names = []
    for f in uploaded_files:
        safe_name = secure_filename(f.name)
        if not safe_name:
            continue
        dest = os.path.realpath(os.path.join(tmp_dir, safe_name))
        if not dest.startswith(tmp_dir + os.sep):
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
        path = os.path.realpath(os.path.join(tmp_dir, safe_fname))
        if not path.startswith(os.path.realpath(tmp_dir) + os.sep):
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
        stem = Path(fname).stem
        stem = re.sub(r'^(Naukri|LinkedIn|Indeed|Resume|CV)[_\-\s]+', '', stem, flags=re.IGNORECASE)
        stem = re.sub(r'[_\-\s]*\d+[Yy][_\-\s]*\d*[Mm]?\s*$', '', stem)
        stem = re.sub(r'[_\-\s]*\d+$', '', stem)
        name = stem.replace('_', ' ').replace('-', ' ').strip().title()
        name = re.sub(
            r'\s+(?:Senior|Junior|Lead|Head|Director|Manager|Analyst|Engineer|Developer|Consultant|'
            r'Coordinator|Specialist|Associate|Intern|Officer|Transition|Transit|Program|Prog|PMO)\w*.*$',
            '', name, flags=re.IGNORECASE
        ).strip()
        exp_years = _safe_experience(text)
        cv_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, fname))
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

    seen_names: set = set()
    unique_cvs = []
    for cv in cvs:
        normalized = re.sub(
            r'\s+(Transition|Senior|Junior|Lead|Manager|Analyst|Engineer|Developer|Consultant|'
            r'Coordinator|Specialist|Associate)\w*.*$',
            '', cv.candidate_name, flags=re.IGNORECASE
        ).strip().lower()
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
        if i < len(cvs) - 1:
            time.sleep(1)  # avoid burst rate limiting on free tier
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

    ranking_json = html_lib.unescape(ranking_json)

    items = json.loads(ranking_json)
    verified_names = {cv.candidate_name.lower(): cv.candidate_name for cv in cvs}

    results = []
    for item in items:
        try:
            result = RankedResult(**item)
            if result.cv_id and result.cv_id in cv_id_map:
                result = result.model_copy(update={"candidate_name": cv_id_map[result.cv_id].candidate_name})
            else:
                matched = verified_names.get(result.candidate_name.lower())
                if matched:
                    result = result.model_copy(update={"candidate_name": matched})
            results.append(result)
        except Exception as e:
            logger.warning(f"Failed to parse result item: {e}")

    status_placeholder.empty()
    return validate_results(results)
