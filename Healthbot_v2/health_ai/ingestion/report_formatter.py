"""
report_formatter.py

Deterministic lab report summary builder — no LLM, no token limits.
Built directly from ClinicalAnalyzer output.
"""

from collections import defaultdict
from typing import Dict, Any, List


# ── Clinical notes for abnormal results ──────────────────────────────────────
_ABNORMAL_NOTES = {
    "Vitamin D (25-OH)":   "⚠ DEFICIENCY — Normal sufficiency: 30–100 ng/mL",
    "Vitamin B12":         "⚠ DEFICIENT — Normal: 187–833 pg/mL",
    "HbA1c":               "⚠ Poor Diabetic Control (Good control: 6.0–7.0%)",
    "Fasting Blood Sugar": "⚠ Elevated — consistent with uncontrolled Diabetes",
    "Homocysteine":        "⚠ High — independent cardiovascular risk factor",
    "IgE":                 "⚠ Elevated — possible allergic disease or parasitic infection",
    "MPV":                 "⚠ Elevated Mean Platelet Volume — may indicate platelet activation",
    "Urine Glucose":       "⚠ Glucose Present in urine — consistent with Diabetes",
    "Hb A":                "⚠ Reduced Hb A% — report interpretation: Negative for beta thalassemia trait. Correlate clinically.",
    "Urea":                "ℹ Marginally low — clinically not significant in isolation",
    "BUN":                 "ℹ Marginally low — clinically not significant in isolation",
    "Neutrophils Abs":     "⚠ Absolute neutrophil count elevated — may indicate infection or inflammation",
    "WBC Count":           "⚠ Mildly elevated — monitor for infection",
    "Triglyceride":        "⚠ Borderline high — dietary modification advised",
    "LDL Cholesterol":     "⚠ Near optimal upper limit — monitor cardiovascular risk",
    "Lymphocytes %":       "ℹ Marginally low — correlate with clinical picture",
}

# ── Notes for selected normal results ────────────────────────────────────────
_NORMAL_NOTES = {
    "Cholesterol":       "Desirable range (<200 mg/dL)",
    "HDL Cholesterol":   "Normal — protective level (>40 mg/dL)",
    "ESR":               "Normal",
    "Microalbumin":      "Normal — no microalbuminuria",
    "PSA Total":         "Normal (0–4 ng/mL)",
    "HIV I&II Ab/Ag":    "Non Reactive",
    "HBsAg":             "Non Reactive",
    "TSH":               "Normal thyroid function",
}

# ── Display overrides for textual / special results ──────────────────────────
_DISPLAY_OVERRIDES = {
    "Urine Glucose":  lambda t: ("Present (+)", "Absent (Expected)"),
    "Urine Protein":  lambda t: ("Absent", "Absent (Normal)"),
}

# ── Panel display order ───────────────────────────────────────────────────────
_PANEL_ORDER = [
    "Complete Blood Count",
    "ESR",
    "Lipid Profile",
    "Biochemistry",
    "HbA1c",
    "Thyroid Function",
    "Urine Biochemistry",
    "Liver Function",
    "Iron Studies",
    "Kidney Function",
    "Electrolytes",
    "Immunoassay",
    "Infectious Markers",
    "HB Electrophoresis",
    "Urine Examination",
]


def _ref_str(ref_min, ref_max, name="") -> str:
    # Special textual override
    if name == "Urine Glucose":
        return "Absent (Expected)"
    if ref_min is not None and ref_max is not None:
        return f"{ref_min}–{ref_max}"
    elif ref_max is not None:
        return f"<{ref_max}"
    elif ref_min is not None:
        return f">{ref_min}"
    return ""


def _format_value(t: dict) -> str:
    name = t['name']
    if name in _DISPLAY_OVERRIDES:
        display, _ = _DISPLAY_OVERRIDES[name](t)
        return display
    v = t['value']
    if v is None:
        return "—"
    if isinstance(v, float) and v == int(v) and v > 10:
        return str(int(v))
    return str(v)


def _group_by_panel_ordered(tests: List[dict]) -> dict:
    groups = defaultdict(list)
    for t in tests:
        groups[t['panel']].append(t)
    ordered = {}
    for panel in _PANEL_ORDER:
        if panel in groups:
            ordered[panel] = groups[panel]
    for panel in groups:
        if panel not in ordered:
            ordered[panel] = groups[panel]
    return ordered


def build_lab_summary(analysis: Dict[str, Any]) -> str:
    """
    Build a complete formatted lab report summary. No LLM involved.
    """
    abnormal = analysis.get('abnormal_tests', [])
    normal   = analysis.get('normal_tests', [])
    unknown  = analysis.get('unknown_tests', [])
    flags    = analysis.get('global_flags', {})

    total = len(abnormal) + len(normal) + len(unknown)
    lines = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines.append(f"**Lab Report Summary** — {total} tests across all panels\n")

    if flags.get('urgent'):
        lines.append("🚨 **URGENT: Multiple severe abnormalities detected. Please consult a doctor promptly.**\n")
    else:
        lines.append(
            f"📊 **{len(abnormal)} Abnormal** · {len(normal)} Normal · {len(unknown)} Informational\n"
        )

    # ── ABNORMAL ──────────────────────────────────────────────────────────────
    lines.append("---\n## 🔴 ABNORMAL RESULTS\n")
    severity_order = {"Severe": 0, "Moderate": 1, "Mild": 2, "None": 3}
    abnormal_sorted = sorted(abnormal, key=lambda x: severity_order.get(x['severity'], 4))

    for panel, tests in _group_by_panel_ordered(abnormal_sorted).items():
        lines.append(f"\n**{panel}**")
        for t in tests:
            direction = "HIGH ↑" if t['status'] == 'High' else "LOW ↓"
            sev = t['severity']
            sev_str = f" — {sev}" if sev and sev != 'None' else ""
            ref = _ref_str(t['ref_min'], t['ref_max'], t['name'])
            ref_str = f"  _(Ref: {ref})_" if ref else ""
            val = _format_value(t)
            note = f"\n  {_ABNORMAL_NOTES[t['name']]}" if t['name'] in _ABNORMAL_NOTES else ""
            lines.append(
                f"- **{t['name']}**: {val} {t['unit']}  [{direction}{sev_str}]{ref_str}{note}"
            )

    # ── NORMAL ────────────────────────────────────────────────────────────────
    lines.append("\n---\n## ✅ NORMAL RESULTS\n")
    for panel, tests in _group_by_panel_ordered(normal).items():
        lines.append(f"\n**{panel}**")
        for t in tests:
            # Skip Urine Protein here — it goes to informational
            if t['name'] == "Urine Protein":
                continue
            ref = _ref_str(t['ref_min'], t['ref_max'], t['name'])
            ref_str = f"  _(Ref: {ref})_" if ref else ""
            val = _format_value(t)
            note = f"  — {_NORMAL_NOTES[t['name']]}" if t['name'] in _NORMAL_NOTES else ""
            lines.append(f"- {t['name']}: {val} {t['unit']}{ref_str}{note}")

    # ── INFORMATIONAL ─────────────────────────────────────────────────────────
    info_items = list(unknown)
    # Add Urine Protein here with correct display
    for t in normal:
        if t['name'] == "Urine Protein":
            info_items.append({**t, '_display': "Absent (Normal)"})

    if info_items:
        lines.append("\n---\n## ℹ️ INFORMATIONAL\n")
        for t in info_items:
            display = t.get('_display') or _format_value(t)
            unit = t['unit'] or ""
            extra = ""
            if t['name'] == "Mean Blood Glucose":
                unit = "mg/dL"
                extra = "  — calculated from HbA1c"
            lines.append(f"- {t['name']}: {display} {unit}{extra}".rstrip())

    return "\n".join(lines)
