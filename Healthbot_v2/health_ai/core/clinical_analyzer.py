from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any


@dataclass
class LabTest:
    panel: str
    name: str
    value: Optional[float]
    unit: Optional[str]
    ref_min: Optional[float]
    ref_max: Optional[float]
    status: str = "Unknown"
    severity: str = "None"


class ClinicalAnalyzer:

    def __init__(self):
        self.tests: List[LabTest] = []

    def analyze(self, structured_tests: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.tests = []
        for item in structured_tests:
            test = LabTest(
                panel=item.get("panel", "Unknown"),
                name=item.get("name", ""),
                value=item.get("value"),
                unit=item.get("unit"),
                ref_min=item.get("ref_min"),
                ref_max=item.get("ref_max"),
            )
            test.status = self.classify_value(
                test.value, test.ref_min, test.ref_max
            )
            test.severity = self.assign_severity(
                test.value, test.ref_min, test.ref_max, test.status
            )
            self.tests.append(test)
        return self.build_output()

    def classify_value(
        self,
        value: Optional[float],
        ref_min: Optional[float],
        ref_max: Optional[float],
    ) -> str:
        """
        Classify a test result as High, Low, or Normal.

        Handles one-sided reference ranges:
          - ref_max only (e.g. Cholesterol <200):  High if value > ref_max
          - ref_min only (e.g. HDL >40):           Low  if value < ref_min
          - both present:                           standard High/Low/Normal
          - neither present:                        Unknown
        """
        if value is None:
            return "Unknown"

        try:
            value = float(value)
        except (ValueError, TypeError):
            return "Unknown"

        has_min = ref_min is not None
        has_max = ref_max is not None

        if not has_min and not has_max:
            return "Unknown"

        if has_min and has_max:
            try:
                if value < float(ref_min):
                    return "Low"
                elif value > float(ref_max):
                    return "High"
                else:
                    return "Normal"
            except (ValueError, TypeError):
                return "Unknown"

        if has_max:
            try:
                return "High" if value > float(ref_max) else "Normal"
            except (ValueError, TypeError):
                return "Unknown"

        if has_min:
            try:
                return "Low" if value < float(ref_min) else "Normal"
            except (ValueError, TypeError):
                return "Unknown"

        return "Unknown"

    def assign_severity(
        self,
        value: Optional[float],
        ref_min: Optional[float],
        ref_max: Optional[float],
        status: str,
    ) -> str:
        if status in ("Normal", "Unknown") or value is None:
            return "None"

        try:
            value = float(value)
        except (ValueError, TypeError):
            return "None"

        if status == "Low" and ref_min is not None:
            try:
                deviation = (float(ref_min) - value) / float(ref_min)
            except (ValueError, ZeroDivisionError):
                return "None"
        elif status == "High" and ref_max is not None:
            try:
                deviation = (value - float(ref_max)) / float(ref_max)
            except (ValueError, ZeroDivisionError):
                return "None"
        else:
            return "None"

        if deviation < 0.1:
            return "Mild"
        elif deviation < 0.3:
            return "Moderate"
        else:
            return "Severe"

    def build_output(self) -> Dict[str, Any]:
        abnormal, normal, unknown = [], [], []
        for test in self.tests:
            d = asdict(test)
            if test.status in ("High", "Low"):
                abnormal.append(d)
            elif test.status == "Normal":
                normal.append(d)
            else:
                unknown.append(d)

        severity_order = {"Severe": 0, "Moderate": 1, "Mild": 2, "None": 3}
        abnormal.sort(key=lambda x: severity_order.get(x["severity"], 4))

        panel_summary = self._group_by_panel(self.tests)
        global_flags = self._detect_global_flags(abnormal)

        return {
            "abnormal_tests": abnormal,
            "normal_tests": normal,
            "unknown_tests": unknown,
            "panel_summary": panel_summary,
            "global_flags": global_flags,
            "requires_medical_attention": global_flags["urgent"],
        }

    def _group_by_panel(self, tests: List[LabTest]) -> Dict[str, Any]:
        panels: Dict[str, Any] = {}
        for test in tests:
            panels.setdefault(test.panel, {"total": 0, "abnormal": 0, "tests": []})
            panels[test.panel]["total"] += 1
            if test.status in ("High", "Low"):
                panels[test.panel]["abnormal"] += 1
            panels[test.panel]["tests"].append(asdict(test))
        return panels

    def _detect_global_flags(self, abnormal_tests: List[Dict]) -> Dict[str, Any]:
        severe = sum(1 for t in abnormal_tests if t["severity"] == "Severe")
        moderate = sum(1 for t in abnormal_tests if t["severity"] == "Moderate")
        urgent = severe >= 2 or (severe >= 1 and moderate >= 2)
        return {
            "total_abnormal": len(abnormal_tests),
            "severe_count": severe,
            "moderate_count": moderate,
            "urgent": urgent,
        }
