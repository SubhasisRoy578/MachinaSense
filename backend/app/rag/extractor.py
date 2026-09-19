import os
import io
import re
from typing import Dict, List, Any, Tuple
from pypdf import PdfReader
from docx import Document

ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.txt'}
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB

class DocumentExtractionError(Exception):
    pass

def validate_file(filename: str, file_bytes: bytes) -> str:
    ext = os.path.splitext(filename.lower())[1]
    if ext not in ALLOWED_EXTENSIONS:
        raise DocumentExtractionError(f"Unsupported file extension '{ext}'. Allowed extensions: .pdf, .docx, .txt")
    if len(file_bytes) == 0:
        raise DocumentExtractionError("File is empty (0 bytes).")
    if len(file_bytes) > MAX_FILE_SIZE:
        raise DocumentExtractionError(f"File size exceeds limit of 25MB ({len(file_bytes) / 1024 / 1024:.1f} MB).")
    return ext

def extract_text(filename: str, file_bytes: bytes) -> Tuple[List[Dict[str, Any]], str]:
    """
    Extracts text from PDF, DOCX, or TXT file bytes.
    Returns (pages_list, full_text).
    pages_list items: {"page": page_num, "section": section_name, "text": text_content}
    """
    ext = validate_file(filename, file_bytes)
    pages: List[Dict[str, Any]] = []
    full_text_parts: List[str] = []

    try:
        if ext == '.pdf':
            reader = PdfReader(io.BytesIO(file_bytes))
            if len(reader.pages) == 0:
                raise DocumentExtractionError("PDF file contains no readable pages.")

            current_section = "General Specifications"
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                text = text.strip()
                if not text:
                    continue

                # Simple section header detection heuristic (e.g. lines like "Section 2.1 - Bearing Assembly")
                header_match = re.search(r'^(Section \d+[\.\d+]*[^\n]+|[A-Z0-9\s]{4,30}\n)', text, re.MULTILINE)
                if header_match:
                    current_section = header_match.group(0).strip()

                pages.append({
                    "page": i + 1,
                    "section": current_section,
                    "text": text
                })
                full_text_parts.append(text)

        elif ext == '.docx':
            doc = Document(io.BytesIO(file_bytes))
            current_section = "General Information"
            current_text: List[str] = []
            page_num = 1

            for p in doc.paragraphs:
                txt = p.text.strip()
                if not txt:
                    continue

                if p.style and ('Heading' in p.style.name or 'Title' in p.style.name):
                    if current_text:
                        pages.append({
                            "page": page_num,
                            "section": current_section,
                            "text": "\n".join(current_text)
                        })
                        full_text_parts.append("\n".join(current_text))
                        current_text = []
                        page_num += 1
                    current_section = txt

                current_text.append(txt)

            if current_text:
                pages.append({
                    "page": page_num,
                    "section": current_section,
                    "text": "\n".join(current_text)
                })
                full_text_parts.append("\n".join(current_text))

        elif ext == '.txt':
            try:
                decoded = file_bytes.decode('utf-8')
            except UnicodeDecodeError:
                decoded = file_bytes.decode('latin-1')

            decoded = decoded.strip()
            if not decoded:
                raise DocumentExtractionError("TXT file contains no text.")

            # Split by double newlines into pseudo-pages
            paragraphs = [p.strip() for p in decoded.split('\n\n') if p.strip()]
            for i, p_text in enumerate(paragraphs):
                pages.append({
                    "page": i + 1,
                    "section": f"Paragraph {i + 1}",
                    "text": p_text
                })
                full_text_parts.append(p_text)

        full_text = "\n\n".join(full_text_parts).strip()
        if not full_text:
            raise DocumentExtractionError("Failed to extract readable text content from document.")

        return pages, full_text

    except DocumentExtractionError:
        raise
    except Exception as e:
        raise DocumentExtractionError(f"Error parsing document '{filename}': {str(e)}")
