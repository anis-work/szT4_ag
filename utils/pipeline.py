"""Business logic and pipeline functions for CV Ranking Agent."""

import asyncio
import json
import html as html_lib
import tempfile
import os
import uuid
import logging
import re
from pathlib import Path

import numpy as np
import streamlit as st
from werkzeug.utils import secure_filename

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.google.google_ai import GoogleAIChatCompletion
from semantic_kernel.connectors.ai.google.google_ai.google_ai_prompt_execution_settings import GoogleAIPromptExecutionSettings
from semantic_kernel.functions import KernelFunctionFromPrompt

from config import GOOGLE_API_KEY, GEMINI_MODEL
from models import CV, JobDescription, RankedResult
from embedder import embed_text, embed_query
from vector_store import VectorStore
from plugins.cv_retrieval_plugin import CVRetrievalPlugin
from pdf_loader import _extract_text, _chunk_text, extract_experience_years
from validator import validate_results

logger = logging.getLogger(__name__)


# ── JD Parsing Prompt ─────────────────────────────────────────────────────────

JD_PARSE_PROMPT = """You are a job description analyst. Parse the job description below and extract requirements into three tiers.

JOB DESCRIPTION:
{{$job_description}}

Rules:
- CORE: Must-have skills/experience explicitly required. Missing these = candidate is unqualified.
- IMPORTANT: Strong preferences, tools, domain knowledge. Missing these = significant gap.
- OPTIONAL: Nice-to-have, soft skills, travel, environment exposure. Missing these = minor gap.
- min_experience: Extract minimum years required as integer. If not stated, return 0.

Return ONLY valid JSON, no markdown, no explanation:
{
  "core": ["skill1", "skill2"],
  "important": ["skill3", "skill4"],
  "optional": ["skill5", "skill6"],
  "min_experience": 0
}"""


# ── Reasoning Prompt ──────────────────────────────────────────────────────────

REASONING_PROMPT = """You are a technical recruiter providing written justification for pre-computed candidate scores.

JOB DESCRIPTION:
{{$job_description}}

CANDIDATE PROFILES WITH PRE-COMPUTED SCORES:
{{$retrieved_cvs}}

For each candidate, write a justification for their score. Do NOT change the score — it is already computed.

For each candidate write exactly 3 sentences:
1. Which core skills were evidenced and from which specific role/project
2. What is missing or weak
3. Experience fit relative to JD requirement

Also extract:
- key_strengths: comma-separated list of matched skills with evidence
- skills_missing: comma-separated list of missing core/important skills
- skills_matched: count of core + important skills evidenced

Return ONLY a valid JSON array. No markdown, no explanation, no code fences.
Each object must have:
  cv_id (string, copy exactly from CV_ID tag),
  candidate_name (string, copy exactly from VERIFIED_NAME tag),
  reason (string, exactly 3 sentences),
  key_strengths (string),
  skills_matched (integer),
  skills_missing (string)
"""


@st.cache_resource
def get_kernel() -> Kernel:
    """Initialize and return a Semantic Kernel instance."""
    kernel = Kernel()
    kernel.add_service(GoogleAIChatCompletion(gemini_model_id=GEMINI_MODEL, api_key=GOOGLE_API_KEY))

    kernel.add_function(plugin_name="jd_parser", function=KernelFunctionFromPrompt(
        function_name="parse_jd",
        plugin_name="jd_parser",
        prompt=JD_PARSE_PROMPT,
        prompt_execution_settings=GoogleAIPromptExecutionSettings(temperature=0.0, seed=42),
    ))

    kernel.add_function(plugin_name="reasoning", function=KernelFunctionFromPrompt(
        function_name="reason_candidates",
        plugin_name="reasoning",
        prompt=REASONING_PROMPT,
        prompt_execution_settings=GoogleAIPromptExecutionSettings(temperature=0.0, seed=42),
    ))

    return kernel


# ── Phase 1: Python Pre-Scoring ───────────────────────────────────────────────

def _safe_experience(text: str) -> float:
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


def _cosine_similarity(a: list, b: list) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


def _skill_present(skill_emb, cv_chunk_embs, threshold=0.72) -> bool:
    if not cv_chunk_embs:
        return False
    return max(_cosine_similarity(skill_emb, c) for c in cv_chunk_embs) >= threshold


def _compute_pre_score(
    cv: CV,
    core_embeddings: list,
    important_embeddings: list,
    optional_embeddings: list,
    min_experience: int,
    cv_chunk_embs: list,
) -> dict:
    if not cv_chunk_embs:
        return {"pre_score": 0, "core_matched": 0, "important_matched": 0}

    core_matched = sum(1 for emb in core_embeddings if _skill_present(emb, cv_chunk_embs))
    important_matched = sum(1 for emb in important_embeddings if _skill_present(emb, cv_chunk_embs))
    optional_matched = sum(1 for emb in optional_embeddings if _skill_present(emb, cv_chunk_embs))

    score = (core_matched * 30) + (important_matched * 12) + (optional_matched * 1)

    missing_core = len(core_embeddings) - core_matched
    if missing_core > 1:
        score -= 40

    exp = cv.experience_years or 0.0
    if min_experience > 0:
        if exp >= min_experience:
            score += 15
            if exp >= min_experience + 3:
                score += 5
        else:
            score = min(score, 55)
    else:
        if exp >= 5:
            score += 15
        elif exp >= 2:
            score += 8

    return {
        "pre_score": max(0, min(round(score), 100)),
        "core_matched": core_matched,
        "important_matched": important_matched,
    }


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


async def _parse_jd(kernel: Kernel, jd_text: str) -> dict:
    """Use LLM to parse JD into tiered requirements."""
    parse_fn = kernel.get_function(plugin_name="jd_parser", function_name="parse_jd")
    result = await kernel.invoke(parse_fn, job_description=jd_text)
    raw = html_lib.unescape(str(result).strip())
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    parsed = json.loads(raw)
    return {
        "core": parsed.get("core", []),
        "important": parsed.get("important", []),
        "optional": parsed.get("optional", []),
        "min_experience": int(parsed.get("min_experience", 0)),
    }


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
    """
    Two-phase CV ranking pipeline:
    Phase 1 — Python pre-scores candidates deterministically using embeddings + weighted skill matching
    Phase 2 — LLM provides reasoning/justification only (does not change scores)
    """

    # ── Step 1: Embed resumes ─────────────────────────────────────────────────
    status_placeholder.info("📄 Step 1/4: Embedding resumes...")
    bar = st.progress(0)
    for i, cv in enumerate(cvs):
        cv.embedding = embed_text(cv.raw_text)
        bar.progress((i + 1) / len(cvs))
    bar.empty()

    # ── Step 2: Parse JD into tiered requirements ─────────────────────────────
    status_placeholder.info("🔍 Step 2/4: Parsing job description requirements...")
    try:
        jd_tiers = await _parse_jd(kernel, jd.requirements)
        logger.info(f"JD parsed — Core: {jd_tiers['core']}, Important: {jd_tiers['important']}, "
                    f"Optional: {jd_tiers['optional']}, Min exp: {jd_tiers['min_experience']}")
    except Exception as e:
        logger.warning(f"JD parsing failed, falling back to single-tier: {e}")
        jd_tiers = {"core": [], "important": [], "optional": [], "min_experience": 0}

    # Embed each skill tier for similarity matching
    core_embeddings = [embed_query(s) for s in jd_tiers["core"]] if jd_tiers["core"] else []
    important_embeddings = [embed_query(s) for s in jd_tiers["important"]] if jd_tiers["important"] else []
    optional_embeddings = [embed_query(s) for s in jd_tiers["optional"]] if jd_tiers["optional"] else []

    # ── Step 3: Python pre-scoring ────────────────────────────────────────────
    status_placeholder.info("⚖️ Step 3/4: Computing weighted scores...")
    pre_scores = {}
    for cv in cvs:
        chunks = [c.strip() for c in cv.raw_text.split("\n\n---\n\n") if c.strip()]
        if not chunks:
            chunks = [cv.raw_text]
        # Store chunk embeddings on CV to avoid re-embedding later
        if not cv.chunk_embeddings:
            cv.chunk_embeddings = [embed_text(chunk) for chunk in chunks[:15]]

        pre_scores[cv.id] = _compute_pre_score(
            cv,
            core_embeddings,
            important_embeddings,
            optional_embeddings,
            jd_tiers["min_experience"],
            cv.chunk_embeddings,
        )
        logger.info(f"{cv.candidate_name}: pre_score={pre_scores[cv.id]['pre_score']}, "
                    f"core_matched={pre_scores[cv.id]['core_matched']}")

    # Sort by score desc, break ties by domain strength (core + important matched)
    cvs_sorted = sorted(
        cvs,
        key=lambda c: (
            pre_scores[c.id]["core_matched"] >= 1,
            pre_scores[c.id]["pre_score"],
            pre_scores[c.id]["core_matched"] + pre_scores[c.id]["important_matched"]
        ),
        reverse=True
    )

    # Build retrieved_cvs string with pre-scores injected
    vs = VectorStore()
    for cv in cvs:
        vs.add(cv)

    kernel.add_plugin(CVRetrievalPlugin(vs), plugin_name="retrieval")
    retrieve_fn = kernel.get_function(plugin_name="retrieval", function_name="retrieve")
    retrieved = await _invoke_with_retry(kernel, retrieve_fn, query=jd.requirements, top_k=len(cvs))
    retrieved_str = str(retrieved).strip()

    # Inject pre-scores into the retrieved text so LLM sees them
    for cv in cvs_sorted:
        ps = pre_scores[cv.id]["pre_score"]
        old = f"VERIFIED_NAME: {cv.candidate_name}"
        new = f"VERIFIED_NAME: {cv.candidate_name} | PRE_COMPUTED_SCORE: {ps}"
        retrieved_str = retrieved_str.replace(old, new, 1)

    cv_id_map = {cv.id: cv for cv in cvs}

    # ── Step 4: LLM reasoning only ────────────────────────────────────────────
    status_placeholder.info("🤖 Step 4/4: Generating candidate assessments...")
    reason_fn = kernel.get_function(plugin_name="reasoning", function_name="reason_candidates")
    result = await _invoke_with_retry(
        kernel, reason_fn,
        job_description=jd.requirements,
        retrieved_cvs=retrieved_str
    )

    reasoning_json = html_lib.unescape(str(result).strip())
    if reasoning_json.startswith("```"):
        reasoning_json = reasoning_json.split("```")[1]
        if reasoning_json.startswith("json"):
            reasoning_json = reasoning_json[4:]
        reasoning_json = reasoning_json.strip()

    reasoning_items = json.loads(reasoning_json)
    reasoning_map = {item.get("cv_id", ""): item for item in reasoning_items}

    # ── Assemble final results using pre-scores + LLM reasoning ──────────────
    verified_names = {cv.candidate_name.lower(): cv.candidate_name for cv in cvs}
    results = []

    for rank_idx, cv in enumerate(cvs_sorted, start=1):
        ps = pre_scores[cv.id]
        reasoning = reasoning_map.get(cv.id, {})

        # Authoritative name from cv_id_map
        candidate_name = cv_id_map[cv.id].candidate_name

        try:
            result = RankedResult(
                rank=rank_idx,
                cv_id=cv.id,
                candidate_name=candidate_name,
                score=ps["pre_score"],
                reason=reasoning.get("reason", ""),
                experience_years=cv.experience_years or 0.0,
                key_strengths=reasoning.get("key_strengths", ""),
                skills_matched=reasoning.get("skills_matched", ps["core_matched"] + ps["important_matched"]),
                skills_missing=reasoning.get("skills_missing", ""),
            )
            results.append(result)
        except Exception as e:
            logger.warning(f"Failed to assemble result for {cv.candidate_name}: {e}")

    status_placeholder.empty()
    return validate_results(results)
