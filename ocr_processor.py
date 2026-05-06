"""Local OCR processor for scanned PDFs using EasyOCR + PyMuPDF with post-correction."""

import re
import logging
from typing import Optional
try:
    import fitz  # PyMuPDF
    import easyocr
    from PIL import Image
    import io
    import numpy as np
    OCR_AVAILABLE = True
    _reader = None  # Lazy load EasyOCR reader
except ImportError:
    OCR_AVAILABLE = False
    _reader = None

logger = logging.getLogger(__name__)

# Prompt is tightly constrained: fix only OCR noise, never rewrite or add content
_OCR_CORRECTION_PROMPT = """You are an OCR error corrector. Your ONLY job is to fix character-level mistakes introduced by optical character recognition scanning.

STRICT RULES — you MUST follow all of these:
1. Fix ONLY characters that are clearly wrong due to OCR scanning (e.g. '5' mistaken for 'S', '0' for 'O', '1' for 'I' or 'l', 'rn' for 'm')
2. Fix broken words split across lines by the scanner
3. Fix garbled punctuation caused by scanning artifacts
4. Do NOT rephrase, reword, or restructure any sentence
5. Do NOT add any new words, skills, dates, or information that is not already present
6. Do NOT remove any existing words or information
7. Do NOT change formatting beyond fixing obvious scan-induced line breaks
8. If a word looks intentional and not like an OCR error, leave it exactly as-is
9. Return ONLY the corrected text — no explanations, no commentary, no preamble

OCR TEXT TO CORRECT:
{text}"""


def is_scanned_pdf(text: str, min_length: int = 50) -> bool:
    """Detect if extracted text is too short/empty (likely scanned)."""
    return len(text.strip()) < min_length


def ocr_pdf(pdf_path: str) -> Optional[str]:
    """Extract text from scanned PDF using PyMuPDF + EasyOCR."""
    if not OCR_AVAILABLE:
        return None

    global _reader
    if _reader is None:
        _reader = easyocr.Reader(['en'], gpu=False)

    try:
        doc = fitz.open(pdf_path)
        text_parts = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
            img_data = pix.tobytes("png")
            # Use context manager to ensure BytesIO is always closed
            with io.BytesIO(img_data) as buf:
                img = Image.open(buf)
                img.load()  # force load before buf closes
            img_array = np.array(img)
            results = _reader.readtext(img_array, detail=0, paragraph=True)
            text_parts.append("\n".join(results))
        doc.close()
        return "\n\n".join(text_parts)
    except Exception:
        return None


def _regex_preclean(text: str) -> str:
    """Fast regex pre-clean for obvious OCR noise before sending to Gemini."""
    # Fix excessive whitespace first — reduces tokens sent to Gemini
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    # Fix phone patterns where letter/digit swap is unambiguous in numeric context
    text = re.sub(r'(\d{3}[-.\s]?)O(\d{3})', r'\g<1>0\2', text)
    text = re.sub(r'(\d{3}[-.\s]?)l(\d{3})', r'\g<1>1\2', text)
    return text.strip()


def correct_ocr_errors(text: str) -> str:
    """Fix OCR errors: regex pre-clean first, then Gemini for context-aware correction."""
    # Step 1: cheap regex pass to reduce noise and token count
    text = _regex_preclean(text)

    # Step 2: Gemini correction (imported here to avoid circular imports)
    try:
        from google import genai
        from config import GOOGLE_API_KEY, GEMINI_MODEL
        client = genai.Client(api_key=GOOGLE_API_KEY)
        prompt = _OCR_CORRECTION_PROMPT.format(text=text)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={"temperature": 0.0, "max_output_tokens": 8192},
        )
        corrected = response.text.strip()
        # Safety fallback: if Gemini returns something suspiciously short or empty, keep original
        if len(corrected) < len(text) * 0.5:
            logger.warning("Gemini OCR correction returned suspiciously short text — using original")
            return text
        return corrected
    except Exception as e:
        logger.warning(f"Gemini OCR correction failed, using regex-only result: {e}")
        return text
