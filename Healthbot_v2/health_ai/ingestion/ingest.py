import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional

from health_ai.ingestion.validator import (
    validate_medical_history,
    validate_daily_log,
    validate_hospitalizations,
    validate_prescriptions,
    validate_family_history,
)
from health_ai.ingestion.text_formatter import (
    format_medical_history,
    format_daily_log,
    format_hospitalizations,
    format_prescriptions,
    format_family_history,
)
from health_ai.rag.chunker import TextChunker, Chunk
from health_ai.embeddings.embedder import EmbeddingModel
from health_ai.rag.vector_store import VectorStore
from health_ai.storage.json_store import JSONStore
from health_ai.core.profile_manager import ProfileManager
from health_ai.core.logger import ingestion_logger
from health_ai.core.exceptions import SchemaValidationError
from health_ai.ingestion.lab_parser import parse_lab_report, format_results
from health_ai.ingestion.sterling_accuris_parser import parse_sterling_accuris_pdf
from health_ai.utils.document_reader import DocumentReader
from health_ai.utils.prescription_cleaner import clean_prescription_text
from health_ai.core.clinical_analyzer import ClinicalAnalyzer


_LAB_KEYWORDS = frozenset([
    "hemoglobin", "wbc", "rbc", "mg/dl", "g/dl",
    "platelet", "vitamin", "cholesterol", "bilirubin",
])
_PRESCRIPTION_KEYWORDS = frozenset([
    "tablet", "tab", "capsule", "mg", "bd", "od",
    "hs", "before food", "after food",
])

_BOILERPLATE_FRAGMENTS = [
    "scan qr code", "passport no", "laboratory test report",
    "patient information", "sample information", "client/location",
    "registration on", "collected at", "collected on",
    "process at", "approved on", "printed on",
    "electronically authenticated", "referred test",
    "sterling accuris", "national reference laboratory",
    "m.d. pathology", "md path", "hematopathologist",
]


def _strip_boilerplate(text: str) -> str:
    cleaned = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or len(stripped) < 3:
            continue
        if any(frag in stripped.lower() for frag in _BOILERPLATE_FRAGMENTS):
            continue
        cleaned.append(stripped)
    return "\n".join(cleaned)


def _extract_pages(file_path: str) -> List[str]:
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                raw = page.extract_text() or ""
                cleaned = _strip_boilerplate(raw)
                if len(cleaned.strip()) > 30:
                    pages.append(cleaned.strip())
        return pages
    except Exception as e:
        ingestion_logger.warning(f"Per-page extraction failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Structured clinical summary builders
# These produce deterministic, pre-classified text from the parsed lab data.
# The LLM never has to re-interpret values — it just formats what's here.
# ─────────────────────────────────────────────────────────────────────────────

def _build_abnormal_first_summary(analysis: Dict[str, Any]) -> str:
    """
    Build a structured summary with ABNORMAL results first, then NORMAL.
    Status is determined by ClinicalAnalyzer using reference ranges from the
    report — NOT by the LLM. This is the source of truth for summary queries.
    """
    lines = ["=== CLINICAL ANALYSIS SUMMARY (Abnormal First) ===\n"]

    flags = analysis.get("global_flags", {})
    total_abnormal = flags.get("total_abnormal", 0)
    lines.append(f"Total tests analysed: "
                 f"{len(analysis.get('abnormal_tests', [])) + len(analysis.get('normal_tests', []))}")
    lines.append(f"Abnormal: {total_abnormal}")
    if flags.get("urgent"):
        lines.append("⚠ URGENT: Multiple severe abnormalities detected.\n")
    else:
        lines.append("")

    # ── Abnormal results grouped by panel ────────────────────────────────────
    abnormal = analysis.get("abnormal_tests", [])
    if abnormal:
        lines.append("--- ABNORMAL RESULTS ---")
        panel_groups: Dict[str, List] = {}
        for t in abnormal:
            panel_groups.setdefault(t["panel"], []).append(t)
        for panel, tests in panel_groups.items():
            lines.append(f"\n{panel}:")
            for t in tests:
                direction = "HIGH ↑" if t["status"] == "High" else "LOW ↓"
                ref = ""
                if t["ref_min"] is not None and t["ref_max"] is not None:
                    ref = f" | Ref: {t['ref_min']}–{t['ref_max']}"
                elif t["ref_max"] is not None:
                    ref = f" | Ref: <{t['ref_max']}"
                elif t["ref_min"] is not None:
                    ref = f" | Ref: >{t['ref_min']}"
                lines.append(
                    f"  {t['name']}: {t['value']} {t.get('unit') or ''} "
                    f"[{direction}] ({t['severity']}){ref}"
                )

    # ── Normal results grouped by panel ──────────────────────────────────────
    normal = analysis.get("normal_tests", [])
    if normal:
        lines.append("\n--- NORMAL RESULTS ---")
        panel_groups = {}
        for t in normal:
            panel_groups.setdefault(t["panel"], []).append(t)
        for panel, tests in panel_groups.items():
            lines.append(f"\n{panel}:")
            for t in tests:
                ref = ""
                if t["ref_min"] is not None and t["ref_max"] is not None:
                    ref = f" | Ref: {t['ref_min']}–{t['ref_max']}"
                elif t["ref_max"] is not None:
                    ref = f" | Ref: <{t['ref_max']}"
                lines.append(
                    f"  {t['name']}: {t['value']} {t.get('unit') or ''} [NORMAL]{ref}"
                )

    return "\n".join(lines)


def _build_complete_listing(structured_tests: List[Dict]) -> str:
    """
    Format ALL parsed tests as a flat listing grouped by panel.
    This is used as the complete-listing chunk — covers every test
    regardless of status. Used for 'give me everything' queries.
    """
    if not structured_tests:
        return ""

    lines = ["=== COMPLETE LAB RESULTS LISTING ===\n"]
    current_panel = None

    for t in structured_tests:
        panel = t.get("panel", "Unknown")
        if panel != current_panel:
            lines.append(f"\n{panel}:")
            current_panel = panel

        name = t.get("name", "")
        value = t.get("value", "")
        unit = t.get("unit") or ""
        ref_min = t.get("ref_min")
        ref_max = t.get("ref_max")

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


class IngestionEngine:

    def __init__(self, profile_id: str):
        self.profile_id = profile_id
        self.chunker = TextChunker()
        self.embedder = EmbeddingModel()
        self.store = VectorStore(profile_id)
        self.json_store = JSONStore()
        self.reader = DocumentReader()

    def _detect_document_type(self, text: str, file_path: str = "") -> str:
        """
        Detect document type. Folder name takes priority over content keywords
        because prescriptions often lack standard keywords (handwritten, abbreviated).
        """
        # Folder-based detection — most reliable signal
        path_lower = file_path.lower()
        if "/prescriptions/" in path_lower or "\\prescriptions\\" in path_lower:
            return "prescription"
        if "/blood_reports/" in path_lower or "/lab_reports/" in path_lower:
            return "lab_report"

        # Content-based fallback
        lower = text.lower()
        if any(k in lower for k in _LAB_KEYWORDS):
            return "lab_report"
        if any(k in lower for k in _PRESCRIPTION_KEYWORDS):
            return "prescription"
        return "unknown"

    def _process(self, text: str, metadata: Dict[str, Any]):
        if not text or not text.strip():
            ingestion_logger.warning(f"[{self.profile_id}] Empty text for ingestion.")
            return
        text = " ".join(text.split())
        chunks = self.chunker.chunk_text(text, metadata)
        embeddings = self.embedder.embed([c.text for c in chunks])
        self.store.add(embeddings, chunks)

    def _process_pages(self, pages: List[str], base_metadata: Dict[str, Any]):
        if not pages:
            return
        chunks = []
        for page_num, page_text in enumerate(pages):
            text = " ".join(page_text.split())
            if not text:
                continue
            chunks.append(Chunk(text=text, metadata={**base_metadata, "page": page_num}))
        if not chunks:
            return
        embeddings = self.embedder.embed([c.text for c in chunks])
        self.store.add(embeddings, chunks)
        ingestion_logger.info(f"[{self.profile_id}] Indexed {len(chunks)} page chunks.")

    def _process_structured_lab_data(
        self, file_path: str, base_metadata: Dict[str, Any]
    ):
        """
        Parse the lab PDF and store TWO pre-computed chunks:
          1. complete_listing  — every test, every panel
          2. clinical_analysis — abnormal-first, H/L/Normal pre-classified

        Strategy: try the Sterling Accuris dedicated parser first (handles
        the specific multi-page format with tiered reference intervals).
        Fall back to the generic table/regex parser for other report formats.
        """
        try:
            # Try the dedicated Sterling Accuris parser first
            structured_tests = parse_sterling_accuris_pdf(file_path)
            if not structured_tests:
                ingestion_logger.info(
                    f"[{self.profile_id}] Sterling parser returned 0 tests, trying generic parser."
                )
                structured_tests = parse_lab_report(file_path)
            if not structured_tests:
                ingestion_logger.warning(
                    f"[{self.profile_id}] All lab parsers returned 0 tests."
                )
                return

            ingestion_logger.info(
                f"[{self.profile_id}] Parsed {len(structured_tests)} structured tests."
            )

            # ── 1. Complete listing (all tests, every panel) ──────────────────
            listing_text = _build_complete_listing(structured_tests)
            if listing_text.strip():
                listing_meta = {
                    **base_metadata,
                    "source_type": "lab_complete_listing",
                    "importance_score": 1.5,
                    "page": -2,
                }
                # Store as one chunk per 400-word block so it fits retrieval budget
                self._process(listing_text, listing_meta)
                ingestion_logger.info(
                    f"[{self.profile_id}] Complete lab listing indexed."
                )

            # ── 2. Abnormal-first clinical analysis ───────────────────────────
            analyzer = ClinicalAnalyzer()
            analysis = analyzer.analyze(structured_tests)
            summary_text = _build_abnormal_first_summary(analysis)
            if summary_text.strip():
                analysis_meta = {
                    **base_metadata,
                    "source_type": "clinical_analysis",
                    "importance_score": 1.5,
                    "page": -1,
                }
                self._process(summary_text, analysis_meta)
                ingestion_logger.info(
                    f"[{self.profile_id}] Clinical analysis summary indexed."
                )

        except Exception as e:
            ingestion_logger.warning(
                f"[{self.profile_id}] Structured lab processing failed: {e}"
            )

    # ── Public ingestion methods ─────────────────────────────────────────────

    def ingest_medical_history(self, data: Dict[str, Any]) -> Dict[str, Any]:
        obj = self._validate(validate_medical_history, data, "medical_history")
        self._process(format_medical_history(obj), {
            "profile_id": self.profile_id,
            "source_type": "medical_history",
            "importance_score": 1.0,
        })
        self.json_store.save(
            ProfileManager.get_static_dir(self.profile_id)
            / f"medical_history_{uuid.uuid4()}.json", data
        )
        ingestion_logger.info(f"[{self.profile_id}] Medical history ingested.")
        return {"status": "success"}

    def ingest_hospitalizations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        obj = self._validate(validate_hospitalizations, data, "hospitalizations")
        self._process(format_hospitalizations(obj), {
            "profile_id": self.profile_id,
            "source_type": "hospitalizations",
            "importance_score": 0.9,
        })
        self.json_store.save(
            ProfileManager.get_static_dir(self.profile_id)
            / f"hospitalizations_{uuid.uuid4()}.json", data
        )
        ingestion_logger.info(f"[{self.profile_id}] Hospitalizations ingested.")
        return {"status": "success"}

    def ingest_prescriptions(self, data: Dict[str, Any]) -> Dict[str, Any]:
        obj = self._validate(validate_prescriptions, data, "prescriptions")
        self._process(format_prescriptions(obj), {
            "profile_id": self.profile_id,
            "source_type": "prescriptions",
            "importance_score": 0.95,
        })
        self.json_store.save(
            ProfileManager.get_static_dir(self.profile_id)
            / f"prescriptions_{uuid.uuid4()}.json", data
        )
        ingestion_logger.info(f"[{self.profile_id}] Prescriptions ingested.")
        return {"status": "success"}

    def ingest_family_history(self, data: Dict[str, Any]) -> Dict[str, Any]:
        obj = self._validate(validate_family_history, data, "family_history")
        self._process(format_family_history(obj), {
            "profile_id": self.profile_id,
            "source_type": "family_history",
            "importance_score": 0.7,
        })
        self.json_store.save(
            ProfileManager.get_static_dir(self.profile_id)
            / f"family_history_{uuid.uuid4()}.json", data
        )
        ingestion_logger.info(f"[{self.profile_id}] Family history ingested.")
        return {"status": "success"}

    def ingest_daily_log(self, data: Dict[str, Any]) -> Dict[str, Any]:
        obj = self._validate(validate_daily_log, data, "daily_log")
        self._process(format_daily_log(obj), {
            "profile_id": self.profile_id,
            "source_type": "daily_log",
            "date": str(obj.date),
            "importance_score": 0.5,
        })
        self.json_store.save(
            ProfileManager.get_dynamic_dir(self.profile_id)
            / f"daily_{uuid.uuid4()}.json", data
        )
        ingestion_logger.info(f"[{self.profile_id}] Daily log ingested.")
        return {"status": "success"}

    def ingest_document(
        self,
        file_path: str,
        source_type: str = "uploaded_document",
        importance_score: float = 0.8,
    ) -> Dict[str, Any]:

        is_pdf = file_path.lower().endswith(".pdf")
        is_image = Path(file_path).suffix.lower() in {".png", ".jpg", ".jpeg"}

        # Use detailed OCR for images so we can log quality and set metadata
        if is_image:
            ocr_result = self.reader.extract_image_for_llm(file_path)
            text = ocr_result["text"]
            ocr_engine = ocr_result.get("engine", "unknown")
            ocr_lines = ocr_result.get("line_count", 0)
            ingestion_logger.info(
                f"[{self.profile_id}] OCR ({ocr_engine}): {ocr_lines} lines "
                f"from {Path(file_path).name}"
            )
        else:
            text = self.reader.extract_text(file_path)
            ocr_engine = None

        # Images (prescriptions, etc.) may legitimately have short text.
        # Use a lower threshold for images vs PDFs.
        min_chars = 15 if is_image else 50
        if not text or len(text.strip()) < min_chars:
            ingestion_logger.warning(
                f"[{self.profile_id}] Extracted text too short "
                f"({len(text.strip()) if text else 0} chars, min={min_chars}). "
                f"File: {Path(file_path).name}"
            )
            return {"status": "failed", "reason": "No meaningful text extracted"}

        document_type = self._detect_document_type(text, file_path=file_path)

        base_metadata = {
            "profile_id": self.profile_id,
            "source_type": source_type,
            "importance_score": importance_score,
            "document_type": document_type,
            "filename": Path(file_path).name,
        }

        if is_image and ocr_engine:
            base_metadata["ocr_engine"] = ocr_engine

        if document_type == "lab_report" and is_pdf:
            # Step 1: structured parsing → store pre-classified chunks
            self._process_structured_lab_data(file_path, base_metadata)

            # Step 2: per-page raw chunks (for fallback retrieval)
            pages = _extract_pages(file_path)
            if pages:
                self._process_pages(pages, base_metadata)
            else:
                self._process(text, base_metadata)

            ingestion_logger.info(
                f"[{self.profile_id}] lab_report fully indexed: "
                f"structured + {len(pages)} page(s)."
            )
            return {"status": "success"}

        elif document_type == "prescription":
            self._process(clean_prescription_text(text), base_metadata)
        else:
            self._process(text, base_metadata)

        ingestion_logger.info(
            f"[{self.profile_id}] {document_type} ingested from {file_path}."
        )
        return {"status": "success"}

    # ── Internal helper ──────────────────────────────────────────────────────

    def _validate(self, validator_fn, data, label):
        try:
            return validator_fn(data)
        except SchemaValidationError as e:
            ingestion_logger.error(f"[{self.profile_id}] {label} validation failed: {e}")
            raise
