import pdfplumber
import re
from typing import List, Dict
from .range_utils import parse_reference_range


class TableLabParser:

    def parse_pdf(self, pdf_path: str) -> List[Dict]:

        results = []

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()

                for table in tables:
                    results.extend(self.parse_table(table))

        return results

    def parse_table(self, table) -> List[Dict]:

        structured = []

        for row in table:

            if not row or len(row) < 3:
                continue

            row = [cell.strip() if cell else "" for cell in row]

            name = row[0]
            value_text = row[2]
            unit = row[3] if len(row) > 3 else None
            ref_text = row[-1]

            value = self.extract_numeric(value_text)
            ref_min, ref_max = parse_reference_range(ref_text)

            if value is None:
                continue

            structured.append({
                "panel": "Unknown",
                "name": name,
                "value": value,
                "unit": unit,
                "ref_min": ref_min,
                "ref_max": ref_max,
            })

        return structured

    def extract_numeric(self, text):
        match = re.search(r"\d+\.?\d*", text)
        if match:
            return float(match.group())
        return None