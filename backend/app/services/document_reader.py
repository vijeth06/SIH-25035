"""
Utility functions to extract text from uploaded documents (PDF, DOCX, TXT).
Optimized for administrative documents used in eConsultation workflows.
"""

from typing import Tuple
from fastapi import UploadFile

def _read_txt(content: bytes, encoding: str = "utf-8") -> str:
    try:
        return content.decode(encoding, errors="ignore")
    except Exception:
        return content.decode("latin-1", errors="ignore")

def _read_docx(file_bytes: bytes) -> str:
    try:
        from docx import Document  # python-docx
        import io
        doc = Document(io.BytesIO(file_bytes))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
        return "\n".join(paragraphs)
    except Exception as e:
        return ""

def _read_pdf(file_bytes: bytes) -> str:
    try:
        import pdfplumber
        import io
        text_parts = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                if t:
                    text_parts.append(t)
        return "\n".join(text_parts)
    except Exception:
        # Fallback to pdfminer.six
        try:
            from pdfminer.high_level import extract_text
            import io
            return extract_text(io.BytesIO(file_bytes)) or ""
        except Exception:
            return ""

def read_text_from_upload(file: UploadFile) -> Tuple[str, str]:
    """
    Read text from an uploaded file and return (text, detected_type).
    detected_type is one of: 'pdf', 'docx', 'txt', 'unknown'.
    """
    filename = (file.filename or "").lower()
    content = file.file.read()  # bytes
    detected_type = "unknown"

    if filename.endswith(".pdf"):
        detected_type = "pdf"
        text = _read_pdf(content)
    elif filename.endswith(".docx"):
        detected_type = "docx"
        text = _read_docx(content)
    elif filename.endswith(".txt") or not filename:
        detected_type = "txt"
        text = _read_txt(content)
    else:
        # Try by MIME type hint
        ct = (file.content_type or "").lower()
        if "pdf" in ct:
            detected_type = "pdf"
            text = _read_pdf(content)
        elif "word" in ct or "docx" in ct:
            detected_type = "docx"
            text = _read_docx(content)
        elif "text" in ct:
            detected_type = "txt"
            text = _read_txt(content)
        else:
            # Final fallback: try text
            text = _read_txt(content)
            detected_type = "txt"

    # Basic cleanup
    text = (text or "").strip()
    return text, detected_type
