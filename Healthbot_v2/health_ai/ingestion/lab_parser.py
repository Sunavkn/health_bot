"""
lab_parser.py
Unified lab report parser: tries table extraction first, falls back to regex.
Exposes `parse_lab_report` and `format_results` for use in the ingestion pipeline.
"""

from .table_lab_parser import TableLabParser
from .regex_lab_parser import RegexLabParser
import pdfplumber
from typing import List, Dict


class LabParser:
    """
    Unified Lab Parser.
    - Uses table extraction first.
    - Falls back to regex parsing.
    - Returns structured numeric lab tests.
    """

    def __init__(self):
        self.table_parser = TableLabParser()
        self.regex_parser = RegexLabParser()

    def parse_pdf(self, pdf_path: str) -> List[Dict]:
        # 1. Try structured table parsing
        structured_tests = self.table_parser.parse_pdf(pdf_path)

        # 2. Fallback to regex if nothing found
        if not structured_tests:
            raw_text = self.extract_text(pdf_path)
            structured_tests = self.regex_parser.parse(raw_text)

        return structured_tests

    def extract_text(self, pdf_path: str) -> str:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text


# ── Module-level helpers used by ingest.py ───────────────────────────────────

def parse_lab_report(pdf_path: str) -> List[Dict]:
    """Parse a lab report PDF and return a list of structured test dicts."""
    parser = LabParser()
    return parser.parse_pdf(pdf_path)


def format_results(structured_tests: List[Dict]) -> str:
    """
    Format structured lab test dicts into a readable text block.
    Output format: 'Test Name: value unit (Ref: min-max)'
    Grouped by panel.
    """
    if not structured_tests:
        return ""

    lines = []
    current_panel = None

    for test in structured_tests:
        panel = test.get("panel", "Unknown")
        if panel != current_panel:
            lines.append(f"\n{panel}:")
            current_panel = panel

        name = test.get("name", "Unknown")
        value = test.get("value", "")
        unit = test.get("unit") or ""
        ref_min = test.get("ref_min")
        ref_max = test.get("ref_max")

        if ref_min is not None and ref_max is not None:
            ref_str = f" (Ref: {ref_min}–{ref_max})"
        elif ref_max is not None:
            ref_str = f" (Ref: <{ref_max})"
        elif ref_min is not None:
            ref_str = f" (Ref: >{ref_min})"
        else:
            ref_str = ""

        lines.append(f"  {name}: {value} {unit}{ref_str}".rstrip())

    return "\n".join(lines)
