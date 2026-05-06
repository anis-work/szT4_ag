"""PDF/DOCX loader with chunking for CV Ranking Agent."""

import os
import re
import uuid
import logging
from datetime import date, timezone, datetime
from pathlib import Path
from typing import List, Optional

from pypdf import PdfReader
from docx import Document

from models import CV
from ocr_processor import is_scanned_pdf, ocr_pdf, correct_ocr_errors, OCR_AVAILABLE

logger = logging.getLogger(__name__)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


# ── PDF extraction ────────────────────────────────────────────────────────────

def _extract_text_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    text = "\n".join(pages)

    if is_scanned_pdf(text):
        logger.info(f"Detected scanned PDF: {path}")
        if OCR_AVAILABLE:
            ocr_text = ocr_pdf(path)
            if ocr_text:
                text = correct_ocr_errors(ocr_text)
                logger.info(f"OCR extracted {len(text)} characters")
            else:
                logger.warning("OCR failed, returning empty text")
        else:
            logger.warning("OCR libraries not installed. Install: pip install -r requirements-ocr.txt")

    return text


# ── DOCX extraction ───────────────────────────────────────────────────────────

def _iter_block_items(parent):
    """Yield paragraphs and tables in document order."""
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    for child in parent.element.body:
        if child.tag == qn('w:p'):
            yield Paragraph(child, parent)
        elif child.tag == qn('w:tbl'):
            yield Table(child, parent)


def _extract_table_text(table) -> list:
    """Recursively extract text from a table including nested tables."""
    from docx.table import Table
    parts = []
    for row in table.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                if p.text.strip():
                    parts.append(p.text)
            for nested in cell.tables:
                parts.extend(_extract_table_text(nested))
    return parts


def _extract_text_from_shapes(doc) -> list:
    """Extract text from drawing/shape text boxes (common in Naukri templates)."""
    WPS_TXBX = '{http://schemas.microsoft.com/office/word/2010/wordprocessingShape}txbx'
    MC_ALTERNATE = '{http://schemas.openxmlformats.org/markup-compatibility/2006}AlternateContent'
    W_P = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'
    W_T = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t'
    parts = []
    try:
        for shape in doc.element.body.iter():
            if shape.tag in (WPS_TXBX, MC_ALTERNATE):
                for p in shape.iter(W_P):
                    text = ''.join(r.text for r in p.iter(W_T) if r.text)
                    if text.strip():
                        parts.append(text)
    except Exception as e:
        logger.debug(f"Shape text extraction skipped: {e}")
    return parts


def _extract_text_docx(path: str) -> str:
    doc = Document(path)
    parts = []

    try:
        for block in _iter_block_items(doc):
            from docx.table import Table
            from docx.text.paragraph import Paragraph
            if isinstance(block, Paragraph):
                if block.text.strip():
                    parts.append(block.text)
            elif isinstance(block, Table):
                parts.extend(_extract_table_text(block))
    except Exception as e:
        logger.debug(f"Block extraction partial failure: {e}")

    parts.extend(_extract_text_from_shapes(doc))

    try:
        for section in doc.sections:
            for hf in (section.header, section.footer):
                if hf:
                    for p in hf.paragraphs:
                        if p.text.strip():
                            parts.append(p.text)
    except Exception as e:
        logger.debug(f"Header/footer extraction skipped: {e}")

    seen, unique = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    return "\n".join(unique)


def _extract_text(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return _extract_text_pdf(path)
    elif ext in (".docx", ".doc"):
        return _extract_text_docx(path)
    raise ValueError(f"Unsupported file type: {ext}")


# ── Name extraction ───────────────────────────────────────────────────────────

def extract_candidate_name(text: str) -> Optional[str]:
    """Extract the candidate's actual name from CV text.

    Strategy (in order of confidence):
    1. Explicit 'Name:' / 'Full Name:' label in first 30 lines
    2. First line in top 10 that matches a proper name pattern
    Returns None so caller can fall back to filename.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    name_re = re.compile(r'^[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+){1,3}$')
    label_re = re.compile(r'^(?:full\s*)?name\s*[:\-]\s*(.+)$', re.IGNORECASE)

    for line in lines[:30]:
        m = label_re.match(line)
        if m:
            candidate = m.group(1).strip()
            if name_re.match(candidate):
                return candidate

    for line in lines[:10]:
        if name_re.match(line):
            return line

    return None


# ── Experience extraction ─────────────────────────────────────────────────────

def extract_experience_years(text: str) -> Optional[float]:
    """Parse total years of experience from raw CV text.

    Handles formats from Naukri, LinkedIn, and standard resumes:
    - Naukri header: '6y 4m', '6 Years 4 Months'
    - Explicit: '6 years of experience', '6+ years'
    - Date ranges: 'Jan 2018 - Present', '2018 - 2024'
    Returns None if no experience can be reliably determined.
    """
    today = datetime.now(timezone.utc).date()

    # Pattern 1: Naukri style — "6y 4m" or "6 Years 4 Months"
    m = re.search(r'(\d+)\s*[Yy](?:ears?|rs?)?[\s,]*(\d+)?\s*[Mm](?:onths?)?', text)
    if m:
        years = int(m.group(1))
        months = int(m.group(2)) if m.group(2) else 0
        return round(years + months / 12, 1)

    # Pattern 2: Explicit statement — "X years of experience" / "X+ years"
    m = re.search(r'(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)', text, re.IGNORECASE)
    if m:
        return float(m.group(1))

    # Pattern 3: Sum date ranges
    MONTHS = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
               'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}

    def _parse_date(s: str) -> Optional[date]:
        s = s.strip().lower()
        if s in ('present', 'current', 'till date', 'till now', 'ongoing'):
            return today
        m2 = re.match(r'([a-z]+)[\s,]+(\d{4})', s)
        if m2:
            mon = MONTHS.get(m2.group(1)[:3])
            if mon:
                return date(int(m2.group(2)), mon, 1)
        m2 = re.match(r'^(\d{4})$', s)
        if m2:
            return date(int(m2.group(1)), 1, 1)
        m2 = re.match(r'(\d{1,2})[/-](\d{4})', s)
        if m2:
            return date(int(m2.group(2)), int(m2.group(1)), 1)
        return None

    date_range_pattern = re.compile(
        r'([A-Za-z]+[\s,]+\d{4}|\d{4}|\d{1,2}[/-]\d{4})'
        r'\s*[-\u2013\u2014to]+\s*'
        r'([A-Za-z]+[\s,]+\d{4}|\d{4}|present|current|till\s*date|till\s*now|ongoing)',
        re.IGNORECASE
    )
    total_days = 0
    for match in date_range_pattern.finditer(text):
        start = _parse_date(match.group(1))
        end = _parse_date(match.group(2))
        if start and end and end >= start:
            total_days += (end - start).days

    if total_days > 180:
        return round(total_days / 365.25, 1)

    return None


# ── Chunking ──────────────────────────────────────────────────────────────────

def _chunk_text(text: str) -> List[str]:
    """Split text into overlapping chunks."""
    chunks, start = [], 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ── CLI loader ────────────────────────────────────────────────────────────────

def load_cvs_from_folder(folder: str) -> List[CV]:
    """Load all PDF/DOCX files from a folder as CV objects."""
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    supported = {".pdf", ".docx", ".doc"}
    files = [f for f in folder_path.iterdir() if f.suffix.lower() in supported]
    if not files:
        raise ValueError(f"No PDF/DOCX files found in: {folder}")

    cvs = []
    for file in sorted(files):
        text = _extract_text(str(file)).strip()
        if not text:
            continue
        chunks = _chunk_text(text)
        full_text = "\n\n---\n\n".join(chunks)
        candidate_name = file.stem.replace("_", " ").replace("-", " ").title()
        cvs.append(CV(
            id=str(uuid.uuid4()),
            candidate_name=candidate_name,
            raw_text=full_text,
        ))

    return cvs
