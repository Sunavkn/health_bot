import fitz  # PyMuPDF
from pathlib import Path

def read_pdf_text(path: Path) -> str:
    doc = fitz.open(path)
    text_chunks = []

    for page in doc:
        page_text = page.get_text().strip()
        if page_text:
            text_chunks.append(page_text)

    return "\n\n".join(text_chunks)
