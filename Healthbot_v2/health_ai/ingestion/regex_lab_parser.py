import re
from typing import List, Dict, Optional
from .range_utils import parse_reference_range


class RegexLabParser:

    def __init__(self):
        self.current_panel = "Unknown"

    def parse(self, raw_text: str) -> List[Dict]:
        lines = self.clean_text(raw_text)
        results = []

        for line in lines:
            panel = self.detect_panel(line)
            if panel:
                self.current_panel = panel
                continue

            parsed = self.parse_line(line)
            if parsed:
                results.append(parsed)

        return results

    def clean_text(self, text: str) -> List[str]:
        text = text.replace("\xa0", " ")
        text = re.sub(r"\s+", " ", text)
        return [l.strip() for l in text.split("\n") if l.strip()]

    def detect_panel(self, line: str) -> Optional[str]:
        panels = [
            "Complete Blood Count",
            "Biochemistry",
            "Lipid Profile",
            "Thyroid",
            "Iron",
            "Electrolytes",
            "Urine",
            "Immunoassay",
        ]
        for p in panels:
            if p.lower() in line.lower():
                return p
        return None

    def parse_line(self, line: str) -> Optional[Dict]:
        # Named group is "test_name" (was "n" — caused KeyError on .group("name"))
        pattern = re.compile(
            r"""
            (?P<test_name>[A-Za-z0-9\(\)\-\/\s]+)
            \s+
            (?P<value>\d+\.?\d*)
            \s*
            (?P<unit>[A-Za-z\/%µ]+)?
            \s+
            (?P<range>[\d\.\s\-\<\>\≤\≥]+)
            """,
            re.VERBOSE,
        )

        match = pattern.search(line)
        if not match:
            return None

        try:
            value = float(match.group("value"))
        except (ValueError, TypeError):
            return None

        ref_min, ref_max = parse_reference_range(match.group("range"))

        return {
            "panel": self.current_panel,
            "name": match.group("test_name").strip(),
            "value": value,
            "unit": match.group("unit"),
            "ref_min": ref_min,
            "ref_max": ref_max,
        }
