import re
from typing import Optional, Tuple


def parse_reference_range(range_text: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Parses reference range formats:
    - 74 - 106
    - <200
    - ≤200
    - >6.5
    - ≥6.5
    """

    if not range_text:
        return None, None

    range_text = range_text.strip()

    # Standard range
    match = re.search(r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)", range_text)
    if match:
        return float(match.group(1)), float(match.group(2))

    # Less than
    match = re.search(r"[<≤]\s*(\d+\.?\d*)", range_text)
    if match:
        return None, float(match.group(1))

    # Greater than
    match = re.search(r"[>≥]\s*(\d+\.?\d*)", range_text)
    if match:
        return float(match.group(1)), None

    return None, None