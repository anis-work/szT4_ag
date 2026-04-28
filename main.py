"""Main entry point for CV Ranking Agent using Semantic Kernel."""

import argparse
import asyncio
import json
import logging
from typing import List

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.google.google_ai import GoogleAIChatCompletion
from semantic_kernel.functions import KernelFunctionFromPrompt

from config import GOOGLE_API_KEY, GEMINI_MODEL, TOP_K
from models import CV, JobDescription, RankedResult
from embedder import embed_text, embed_query
from vector_store import VectorStore
from plugins.cv_retrieval_plugin import CVRetrievalPlugin
from plugins.cv_ingestion_plugin import CVIngestionPlugin
from validator import validate_results
from sample_data import get_sample_cvs, get_sample_jd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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


def build_kernel() -> Kernel:
    kernel = Kernel()
    kernel.add_service(GoogleAIChatCompletion(gemini_model_id=GEMINI_MODEL, api_key=GOOGLE_API_KEY))
    kernel.add_function(plugin_name="ranking", function=KernelFunctionFromPrompt(
        function_name="rank_candidates",
        plugin_name="ranking",
        prompt=RANKING_PROMPT,
    ))
    logger.info(f"Kernel initialized with {GEMINI_MODEL}")
    return kernel


async def load_cvs_via_sk(kernel: Kernel, folder: str) -> List[CV]:
    """Use SK CVIngestionPlugin to load CVs from a folder."""
    ingestion_plugin = CVIngestionPlugin()
    kernel.add_plugin(ingestion_plugin, plugin_name="ingestion")

    load_fn = kernel.get_function(plugin_name="ingestion", function_name="load_from_folder")
    result = await kernel.invoke(load_fn, folder=folder)
    logger.info(str(result))

    return ingestion_plugin.cvs


def embed_cvs(cvs: List[CV]) -> List[CV]:
    logger.info(f"Embedding {len(cvs)} CVs...")
    for cv in cvs:
        cv.embedding = embed_text(cv.raw_text)
        logger.info(f"  ✓ Embedded: {cv.candidate_name}")
    return cvs


async def _invoke_with_retry(kernel: Kernel, fn, retries: int = 3, delay: int = 20, **kwargs):
    """Invoke a kernel function with retry on 503/429 transient errors."""
    import asyncio as _asyncio
    for attempt in range(1, retries + 1):
        try:
            return await kernel.invoke(fn, **kwargs)
        except Exception as e:
            msg = str(e)
            if attempt < retries and ("503" in msg or "429" in msg or "UNAVAILABLE" in msg or "EXHAUSTED" in msg):
                logger.warning(f"Attempt {attempt} failed ({msg[:80]}...). Retrying in {delay}s...")
                await _asyncio.sleep(delay)
            else:
                raise


async def run_pipeline(kernel: Kernel, vector_store: VectorStore, jd: JobDescription) -> List[RankedResult]:
    kernel.add_plugin(CVRetrievalPlugin(vector_store), plugin_name="retrieval")

    # Step 1: Retrieve top-K CVs via SK CVRetrievalPlugin
    logger.info(f"Retrieving top {TOP_K} CVs...")
    retrieve_fn = kernel.get_function(plugin_name="retrieval", function_name="retrieve")
    retrieved_result = await _invoke_with_retry(kernel, retrieve_fn, query=jd.requirements, top_k=TOP_K)
    retrieved_cvs_str = str(retrieved_result).strip()

    # Step 2: Rank via SK KernelFunctionFromPrompt
    logger.info("Ranking candidates with Gemini via Semantic Kernel...")
    rank_fn = kernel.get_function(plugin_name="ranking", function_name="rank_candidates")
    ranking_result = await _invoke_with_retry(
        kernel, rank_fn,
        job_description=jd.requirements,
        retrieved_cvs=retrieved_cvs_str,
    )
    ranking_json = str(ranking_result).strip()

    if ranking_json.startswith("```"):
        ranking_json = ranking_json.split("```")[1]
        if ranking_json.startswith("json"):
            ranking_json = ranking_json[4:]
        ranking_json = ranking_json.strip()

    ranking_data = json.loads(ranking_json)
    results = []
    for item in ranking_data:
        try:
            results.append(RankedResult(**item))
        except Exception as e:
            logger.warning(f"Skipping invalid item {item}: {e}")

    return validate_results(results)


def print_results(results: List[RankedResult]) -> None:
    print("\n" + "=" * 90)
    print("CV RANKING RESULTS".center(90))
    print("=" * 90)
    for r in results:
        print(f"\n[Rank {r.rank}] {r.candidate_name}")
        print(f"Score: {r.score}/100")
        print(f"Reason: {r.reason}")
        print("-" * 90)
    print()


async def main():
    parser = argparse.ArgumentParser(description="CV Ranking Agent")
    parser.add_argument("--folder", type=str, default=None,
                        help="Path to folder containing PDF/DOCX resumes")
    parser.add_argument("--role", type=str, default=None,
                        help="Job role title (used with --folder)")
    parser.add_argument("--requirements", type=str, default=None,
                        help="Path to a .txt file with job requirements (used with --folder)")
    args = parser.parse_args()

    kernel = build_kernel()

    if args.folder:
        # --- PDF/DOCX mode via SK ingestion plugin ---
        logger.info(f"Loading CVs from folder: {args.folder}")
        cvs = await load_cvs_via_sk(kernel, args.folder)

        role = args.role or "Software Engineer"
        if args.requirements:
            with open(args.requirements, "r", encoding="utf-8") as f:
                requirements = f.read()
        else:
            requirements = role  # minimal fallback
        jd = JobDescription(role=role, requirements=requirements)

        # Auto-scale TOP_K to number of CVs loaded so all are ranked
        import config as _cfg
        _cfg.TOP_K = len(cvs)
    else:
        # --- Sample data mode (default) ---
        logger.info("Using built-in sample data...")
        cvs = get_sample_cvs()
        jd = get_sample_jd()

    logger.info(f"Loaded {len(cvs)} CVs | Job: {jd.role}")

    cvs = embed_cvs(cvs)

    vector_store = VectorStore()
    for cv in cvs:
        vector_store.add(cv)
    logger.info(f"Vector store: {len(vector_store)} CVs")

    results = await run_pipeline(kernel, vector_store, jd)
    print_results(results)
    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
