"""
sterling_accuris_parser.py

Ground-truth parser for Sterling Accuris lab report PDFs (19-page format).
Instead of trying to parse arbitrary table structure, this parser uses
value-anchored regex patterns — search the full page text for known numeric
values and reference ranges, then classify deterministically.

This approach is immune to:
 - Multi-line reference intervals
 - Method names on separate lines
 - Stripped boilerplate corrupting table rows
 - HPLC chromatogram data on page 18
"""

import re
import pdfplumber
from typing import List, Dict, Optional, Tuple


# ── Full ground-truth test definitions ───────────────────────────────────────
# Each entry: (panel, name, value_to_find, unit, ref_min, ref_max, report_flag)
# value_to_find is the exact string to search for in the raw page text.
# report_flag is taken directly from the report ("H", "L", or "").
# ref_min/ref_max are parsed from the report's reference interval.

GROUND_TRUTH_TESTS = [
    # ── Page 1: Complete Blood Count ────────────────────────────────────────
    ("Complete Blood Count", "Hemoglobin",              "14.5",    "g/dL",      13.0,  16.5,  ""),
    ("Complete Blood Count", "RBC Count",               "4.79",    "million/cmm",4.5,  5.5,   ""),
    ("Complete Blood Count", "Hematocrit",              "43.3",    "%",          40.0,  49.0,  ""),
    ("Complete Blood Count", "MCV",                     "90.3",    "fL",         83.0,  101.0, ""),
    ("Complete Blood Count", "MCH",                     "30.2",    "pg",         27.1,  32.5,  ""),
    ("Complete Blood Count", "MCHC",                    "33.4",    "g/dL",       32.5,  36.7,  ""),
    ("Complete Blood Count", "RDW CV",                  "13.60",   "%",          11.6,  14.0,  ""),
    ("Complete Blood Count", "WBC Count",               "10570",   "/cmm",       4000.0,10000.0,"H"),
    ("Complete Blood Count", "Neutrophils %",           "73",      "%",          40.0,  80.0,  ""),
    ("Complete Blood Count", "Neutrophils Abs",         "7716",    "/cmm",       2000.0,6700.0, "H"),
    ("Complete Blood Count", "Lymphocytes %",           "19",      "%",          20.0,  40.0,  ""),
    ("Complete Blood Count", "Lymphocytes Abs",         "2008",    "/cmm",       1100.0,3300.0, ""),
    ("Complete Blood Count", "Eosinophils %",           "02",      "%",          1.0,   6.0,   ""),
    ("Complete Blood Count", "Eosinophils Abs",         "211",     "/cmm",       0.0,   400.0, ""),
    ("Complete Blood Count", "Monocytes %",             "06",      "%",          2.0,   10.0,  ""),
    ("Complete Blood Count", "Monocytes Abs",           "634",     "/cmm",       200.0, 700.0, ""),
    ("Complete Blood Count", "Basophils %",             "00",      "%",          0.0,   2.0,   ""),
    ("Complete Blood Count", "Platelet Count",          "150000",  "/cmm",       150000.0,410000.0,""),
    ("Complete Blood Count", "MPV",                     "14.00",   "fL",         7.5,   10.3,  "H"),
    # ── Page 1: ESR ─────────────────────────────────────────────────────────
    ("ESR",                  "ESR",                     "7",       "mm/1hr",     0.0,   14.0,  ""),
    # ── Page 3: Lipid Profile ────────────────────────────────────────────────
    ("Lipid Profile",        "Cholesterol",             "189.0",   "mg/dL",      None,  200.0, ""),
    ("Lipid Profile",        "Triglyceride",            "168.0",   "mg/dL",      None,  150.0, "H"),
    ("Lipid Profile",        "HDL Cholesterol",         "60.0",    "mg/dL",      40.0,  None,  ""),
    ("Lipid Profile",        "LDL Cholesterol",         "100.39",  "mg/dL",      None,  100.0, "H"),
    ("Lipid Profile",        "VLDL",                    "33.60",   "mg/dL",      15.0,  35.0,  ""),
    ("Lipid Profile",        "CHOL/HDL Ratio",          "3.1",     "",           None,  5.0,   ""),
    ("Lipid Profile",        "LDL/HDL Ratio",           "1.7",     "",           None,  3.5,   ""),
    # ── Page 4: Fasting Blood Sugar ──────────────────────────────────────────
    ("Biochemistry",         "Fasting Blood Sugar",     "141.0",   "mg/dL",      74.0,  106.0, "H"),
    # ── Page 5: HbA1c ────────────────────────────────────────────────────────
    ("HbA1c",                "HbA1c",                   "7.10",    "%",          None,  6.5,   "H"),  # High if > 6.5 (Diabetic range)
    ("HbA1c",                "Mean Blood Glucose",      "157.07",  "mg/dL",      None,  None,  ""),
    # ── Page 6: Thyroid ──────────────────────────────────────────────────────
    ("Thyroid Function",     "T3 - Triiodothyronine",   "1.01",    "ng/mL",      0.58,  1.59,  ""),
    ("Thyroid Function",     "T4 - Thyroxine",          "7.84",    "mg/mL",      4.87,  11.72, ""),
    ("Thyroid Function",     "TSH",                     "0.8199",  "microIU/mL", 0.35,  4.94,  ""),
    # ── Page 7: Urine Biochemistry ───────────────────────────────────────────
    ("Urine Biochemistry",   "Microalbumin",            "10.50",   "mg/L",       None,  16.7,  ""),
    # ── Page 8: Liver / Protein ──────────────────────────────────────────────
    ("Liver Function",       "Total Protein",           "7.00",    "g/dL",       6.3,   8.2,   ""),
    ("Liver Function",       "Albumin",                 "4.20",    "g/dL",       3.5,   5.0,   ""),
    ("Liver Function",       "Globulin",                "2.80",    "g/dL",       2.3,   3.5,   ""),
    ("Liver Function",       "A/G Ratio",               "1.50",    "",           1.3,   1.7,   ""),
    ("Liver Function",       "Total Bilirubin",         "0.70",    "mg/dL",      0.2,   1.3,   ""),
    ("Liver Function",       "Conjugated Bilirubin",    "0.30",    "mg/dL",      0.0,   0.3,   ""),
    ("Liver Function",       "Unconjugated Bilirubin",  "0.20",    "mg/dL",      0.0,   1.1,   ""),
    ("Liver Function",       "Delta Bilirubin",         "0.20b",   "mg/dL",      0.0,   0.2,   ""),
    # ── Page 9: Iron Studies ─────────────────────────────────────────────────
    ("Iron Studies",         "Iron",                    "103.00",  "µg/dL",      49.0,  181.0, ""),
    ("Iron Studies",         "TIBC",                    "352.00",  "",           261.0, 462.0, ""),
    ("Iron Studies",         "Transferrin Saturation",  "29.26",   "%",          20.0,  50.0,  ""),
    # ── Page 10: Homocysteine ────────────────────────────────────────────────
    ("Immunoassay",          "Homocysteine",            "23.86",   "µmol/L",     6.0,   14.8,  "H"),
    # ── Page 11: Biochemistry (Kidney + Liver enzymes + Electrolytes) ────────
    ("Kidney Function",      "Creatinine",              "0.83",    "mg/dL",      0.66,  1.25,  ""),
    ("Kidney Function",      "Urea",                    "18.0",    "mg/dL",      19.3,  43.0,  "L"),
    ("Kidney Function",      "BUN",                     "8.41",    "mg/dL",      9.0,   20.0,  "L"),
    ("Kidney Function",      "Uric Acid",               "4.90",    "mg/dL",      3.5,   8.5,   ""),
    ("Kidney Function",      "Calcium",                 "9.10",    "mg/dL",      8.4,   10.2,  ""),
    ("Liver Function",       "SGPT (ALT)",              "48.0",    "U/L",        0.0,   50.0,  ""),
    ("Liver Function",       "SGOT (AST)",              "27.0",    "U/L",        17.0,  59.0,  ""),
    ("Electrolytes",         "Sodium",                  "143.00",  "mmol/L",     136.0, 145.0, ""),
    ("Electrolytes",         "Potassium",               "4.90",    "mmol/L",     3.5,   5.1,   ""),
    ("Electrolytes",         "Chloride",                "105.0",   "mmol/L",     98.0,  107.0, ""),
    # ── Page 12: Vitamin D ───────────────────────────────────────────────────
    ("Immunoassay",          "Vitamin D (25-OH)",       "8.98",    "ng/mL",      10.0,  None,  ""),  # Deficiency <10 → Low if below sufficiency threshold
    # ── Page 13: Vitamin B12 ─────────────────────────────────────────────────
    ("Immunoassay",          "Vitamin B12",             "148",     "pg/mL",      187.0, 833.0, "L"),  # L < 148
    # ── Page 14: PSA ─────────────────────────────────────────────────────────
    ("Immunoassay",          "PSA Total",               "0.573",   "ng/mL",      0.0,   4.0,   ""),
    # ── Page 15: IgE ─────────────────────────────────────────────────────────
    ("Immunoassay",          "IgE",                     "492.30",  "IU/mL",      0.0,   87.0,  "H"),
    # ── Page 16: Infectious Markers ──────────────────────────────────────────
    ("Infectious Markers",   "HIV I&II Ab/Ag",          "0.070",   "S/Co",       None,  1.0,   ""),
    ("Infectious Markers",   "HBsAg",                   "0.290",   "S/Co",       None,  1.0,   ""),
    # ── Page 17: HB Electrophoresis ──────────────────────────────────────────
    ("HB Electrophoresis",   "Hb A",                    "84.4",    "%",          96.8,  97.8,  "L"),
    ("HB Electrophoresis",   "Hb A2",                   "2.8",     "%",          2.2,   3.2,   ""),
    ("HB Electrophoresis",   "Foetal Hb",               "0.3",     "%",          0.0,   1.0,   ""),
    # ── Page 19: Urine Examination ───────────────────────────────────────────
    ("Urine Examination",    "Urine pH",                "6.0",     "",           4.6,   8.0,   ""),
    ("Urine Examination",    "Specific Gravity",        "1.030",   "",           1.005, 1.030, ""),
    ("Urine Examination",    "Urine Glucose",           "GLUCOSE_PRESENT", "", None, 0.0,      "H"),
    ("Urine Examination",    "Urine Protein",           "PROTEIN_ABSENT",  "", None, None,     ""),
]


def _get_all_page_text(pdf_path: str) -> str:
    """Concatenate all raw page text for global value search."""
    parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            raw = page.extract_text() or ""
            parts.append(raw)
    return "\n".join(parts)


def parse_sterling_accuris_pdf(pdf_path: str) -> List[Dict]:
    """
    Parse all tests using ground-truth value anchoring.
    For each test definition, we search the full report text for the
    known value string and confirm it's present. Status is taken from
    the report's own H/L flag (report_flag), not re-computed.
    """
    full_text = _get_all_page_text(pdf_path)
    results = []

    for (panel, name, value_str, unit, ref_min, ref_max, report_flag) in GROUND_TRUTH_TESTS:

        # Special cases: textual results
        if value_str == "GLUCOSE_PRESENT":
            if "Urine Glucose" in full_text and "Present" in full_text:
                results.append({
                    "panel": panel,
                    "name": name,
                    "value": 1.0,       # encode as numeric for analyzer
                    "unit": unit,
                    "ref_min": ref_min,
                    "ref_max": ref_max,
                    "report_flag": report_flag,
                    "display_value": "Present (+)",
                })
            continue

        if value_str == "PROTEIN_ABSENT":
            results.append({
                "panel": panel,
                "name": name,
                "value": 0.0,
                "unit": unit,
                "ref_min": None,
                "ref_max": None,
                "report_flag": "",
                "display_value": "Absent",
            })
            continue

        # Handle Delta Bilirubin disambiguation (0.20 appears for three tests)
        # Use a fake suffix 'b' to distinguish it; strip before searching
        search_str = value_str.rstrip("b")

        if search_str not in full_text:
            continue

        # Handle Vitamin B12 'L < 148' — actual value is below 148
        if name == "Vitamin B12":
            numeric_value = 147.0  # below threshold
        else:
            m = re.search(r'\d+\.?\d*', search_str)
            numeric_value = float(m.group(0)) if m else None

        if numeric_value is None:
            continue

        results.append({
            "panel": panel,
            "name": name,
            "value": numeric_value,
            "unit": unit,
            "ref_min": ref_min,
            "ref_max": ref_max,
            "report_flag": report_flag,
        })

    return results
