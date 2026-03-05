"""Thin wrapper kept for backward compatibility. Use DocumentReader directly."""
from health_ai.utils.document_reader import DocumentReader

def extract_text_from_pdf(path) -> str:
    return DocumentReader().extract_text(str(path))
