"""Simplified main entry point for CV Ranking Agent.

Direct RAG pipeline without complex Semantic Kernel plugin infrastructure.
"""

import json
import logging
import google.generativeai as genai
from config import GOOGLE_API_KEY, GEMINI_MODEL, TOP_K
from models import RankedResult
from embedder import embed_text, embed_query
from vector_store import VectorStore
from validator import validate_results
from sample_data import get_sample_cvs, get_sample_jd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_and_embed_cvs():
    """Load CVs and embed them."""
    logger.info("Loading sample CVs...")
    cvs = get_sample_cvs()
    
    logger.info(f"Embedding {len(cvs)} CVs...")
    for cv in cvs:
        embedding = embed_text(cv.raw_text)
        cv.embedding = embedding
        logger.info(f"  ✓ Embedded: {cv.candidate_name}")
    
    return cvs


def build_vector_store(cvs):
    """Build vector store with CVs."""
    logger.info("Building vector store...")
    vector_store = VectorStore()
    for cv in cvs:
        vector_store.add(cv)
    logger.info(f"  ✓ Added {len(vector_store)} CVs to vector store")
    return vector_store


def retrieve_candidates(vector_store, job_requirements):
    """Retrieve top-K candidates using semantic search."""
    logger.info(f"Embedding job requirements...")
    query_embedding = embed_query(job_requirements)
    
    logger.info(f"Searching for top {TOP_K} candidates...")
    relevant_cvs = vector_store.search(query_embedding, top_k=TOP_K)
    
    # Format for ranking prompt
    candidates_text = "RETRIEVED CANDIDATES:\n\n"
    for i, cv in enumerate(relevant_cvs, 1):
        candidates_text += f"Candidate {i}: {cv.candidate_name}\n"
        candidates_text += f"ID: {cv.id}\n"
        candidates_text += f"Profile:\n{cv.raw_text}\n"
        candidates_text += "-" * 80 + "\n\n"
    
    logger.info(f"  ✓ Retrieved {len(relevant_cvs)} candidates")
    return candidates_text


def rank_candidates(job_requirements, candidates_text):
    """Rank candidates using Gemini."""
    logger.info("Ranking candidates with Gemini...")
    
    genai.configure(api_key=GOOGLE_API_KEY)
    
    ranking_prompt = f"""You are an expert recruiter evaluating candidates for a software engineering position.

Your task is to rank the provided candidates based on how well their profiles match the job requirements.

JOB DESCRIPTION:
{job_requirements}

CANDIDATE PROFILES:
{candidates_text}

EVALUATION CRITERIA:
- Match with required skills (Python, FastAPI, PostgreSQL)
- Years of relevant experience (target: 4+)
- Cloud platform experience (AWS preferred)
- System design and architecture knowledge
- Leadership and mentoring experience (preferred)
- Seniority level alignment

INSTRUCTIONS:
1. Evaluate each candidate carefully against the job requirements
2. Assign a score from 0-100 based on fit and match
3. Provide a brief reason for each ranking (1-2 sentences)
4. Return ONLY a valid JSON array with no additional text, markdown, or explanation

Return ONLY this JSON array (no preamble, no markdown):
[
  {{
    "rank": 1,
    "candidate_name": "Name",
    "score": 95,
    "reason": "Exact match with 5+ years experience and all required skills."
  }}
]
"""
    
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(ranking_prompt)
        ranking_json = response.text.strip()
        
        # Try to extract JSON if wrapped in markdown
        if ranking_json.startswith("```"):
            ranking_json = ranking_json.split("```")[1]
            if ranking_json.startswith("json"):
                ranking_json = ranking_json[4:]
        ranking_json = ranking_json.strip()
        
        logger.info(f"  ✓ Received ranking response")
        return ranking_json
    
    except Exception as e:
        logger.error(f"Failed to rank candidates: {str(e)}")
        raise


def main():
    """Run the simplified CV ranking pipeline."""
    try:
        logger.info("=" * 80)
        logger.info("CV RANKING AGENT - SIMPLIFIED PIPELINE")
        logger.info("=" * 80)
        
        # Step 1: Load and embed CVs
        cvs = load_and_embed_cvs()
        
        # Step 2: Build vector store
        vector_store = build_vector_store(cvs)
        
        # Step 3: Load job description
        logger.info("Loading job description...")
        jd = get_sample_jd()
        logger.info(f"  ✓ Job: {jd.role}")
        
        # Step 4: Retrieve relevant candidates
        candidates_text = retrieve_candidates(vector_store, jd.requirements)
        
        # Step 5: Rank candidates with Gemini
        ranking_json = rank_candidates(jd.requirements, candidates_text)
        
        # Step 6: Parse and validate results
        logger.info("Parsing and validating results...")
        try:
            ranking_data = json.loads(ranking_json)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {ranking_json}")
            raise
        
        # Convert to RankedResult objects
        results = []
        for item in ranking_data:
            try:
                result = RankedResult(**item)
                results.append(result)
            except Exception as e:
                logger.warning(f"Invalid ranking item: {item} - {str(e)}")
        
        # Validate
        validated_results = validate_results(results)
        logger.info(f"  ✓ {len(validated_results)} valid results")
        
        # Step 7: Display results
        print("\n" + "=" * 90)
        print("FINAL RANKING RESULTS".center(90))
        print("=" * 90)
        
        if validated_results:
            for result in validated_results:
                print(f"\n[Rank {result.rank}] {result.candidate_name}")
                print(f"Score: {result.score}/100")
                print(f"Reason: {result.reason}")
                print("-" * 90)
        else:
            print("No valid results to display")
        
        print("\n" + "=" * 90)
        logger.info("Pipeline completed successfully!")
    
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
