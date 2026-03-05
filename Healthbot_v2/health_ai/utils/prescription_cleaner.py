"""
prescription_cleaner.py
Cleans raw OCR/PDF text from prescription images before indexing.
"""
import re


def clean_prescription_text(text: str) -> str:
    if not text:
        return ""

    # Normalise whitespace
    text = " ".join(text.split())

    # Remove common OCR noise characters
    text = re.sub(r"[|}{~`]", " ", text)

    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)

    return text.strip()
