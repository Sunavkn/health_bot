"""Thin wrapper kept for backward compatibility. Use DocumentReader directly."""
from health_ai.utils.document_reader import DocumentReader

def extract_text_from_image(path) -> str:
    return DocumentReader()._extract_image(str(path))
