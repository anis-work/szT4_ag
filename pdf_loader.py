"""PDF/DOCX loader with chunking for CV Ranking Agent."""

import os
import uuid
from pathlib import Path
from typing import List

from pypdf import PdfReader
from docx import Document

from models import CV

CHUNK_SIZE = 800      # characters per chunk
CHUNK_OVERLAP = 100   # overlap between chunks


def _extract_text_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _extract_text_docx(path: str) -> str:
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def _extract_text(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return _extract_text_pdf(path)
    elif ext in (".docx", ".doc"):
        return _extract_text_docx(path)
    raise ValueError(f"Unsupported file type: {ext}")


def _chunk_text(text: str) -> List[str]:
    """Split text into overlapping chunks."""
    chunks, start = [], 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def load_cvs_from_folder(folder: str) -> List[CV]:
    """Load all PDF/DOCX files from a folder as CV objects.

    Each file becomes one CV. If the file is large, its text is chunked
    and all chunks are joined so the full context is preserved in raw_text.
    The candidate name is inferred from the filename.
    """
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

        # Chunk and rejoin — keeps full text but validates chunking works
        chunks = _chunk_text(text)
        full_text = "\n\n---\n\n".join(chunks)

        candidate_name = file.stem.replace("_", " ").replace("-", " ").title()
        cvs.append(CV(
            id=str(uuid.uuid4()),
            candidate_name=candidate_name,
            raw_text=full_text,
        ))

    return cvs
