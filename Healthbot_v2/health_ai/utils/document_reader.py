"""
document_reader.py
Extracts text from PDFs (via pdfplumber) and images (via PaddleOCR / pytesseract).

OCR pipeline:
  1. PaddleOCR  – best accuracy, supports angle correction
  2. pytesseract – fallback if PaddleOCR is unavailable
  3. Empty string – caller decides what to do

Key fixes vs original:
  - Confidence threshold lowered 0.60 → 0.40 (keeps handwritten/low-contrast text)
  - Low-confidence lines are appended instead of discarded
  - `extract_image_for_llm()` exposes OCR details for callers to audit quality
"""
import logging
from pathlib import Path
from threading import Lock
from typing import Optional

logger = logging.getLogger(__name__)

_HEADER_LINE_COUNT = 14

_FOOTER_FRAGMENTS = [
    "dr.tejaswini", "dr. sanjeev", "dr.yash", "dr. purvish",
    "dr. hardik", "dr. siddharth", "m.d. pathology", "md path",
    "hematopathologist", "electronically authenticated", "referred test",
    "sterling accuris pathology laboratory", "national reference laboratory",
    "b/s. jalaram", "email:", "page ",
]

# Lowered from 0.60 to 0.40 — keeps handwritten prescription text
_OCR_MIN_CONFIDENCE = 0.40
_OCR_MIN_LINE_CHARS = 2


def _clean_page(raw: str) -> str:
    lines = raw.split("\n")[_HEADER_LINE_COUNT:]
    cleaned = []
    for line in lines:
        s = line.strip()
        if not s or len(s) < 3:
            continue
        if any(f in s.lower() for f in _FOOTER_FRAGMENTS):
            continue
        cleaned.append(s)
    return "\n".join(cleaned).strip()


class DocumentReader:
    """Singleton text extractor supporting PDF and image inputs."""

    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._ocr = None
        return cls._instance

    def _get_ocr(self):
        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR
                self._ocr = PaddleOCR(use_angle_cls=True, lang="en")
                logger.info("PaddleOCR loaded successfully.")
            except Exception as e:
                logger.warning(f"PaddleOCR unavailable: {e}")
                self._ocr = False
        return self._ocr if self._ocr else None

    # ── Public API ────────────────────────────────────────────────────────────

    def extract_text(self, file_path: str) -> str:
        """Return full text from a PDF or image file."""
        path = Path(file_path)
        ext = path.suffix.lower()
        if ext == ".pdf":
            return self._extract_pdf(file_path)
        elif ext in {".png", ".jpg", ".jpeg"}:
            return self._extract_image(file_path)
        logger.warning(f"Unsupported extension '{ext}' for {file_path}")
        return ""

    def extract_pages(self, file_path: str) -> list[str]:
        """Return per-page cleaned text list (PDFs only)."""
        pages = []
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    raw = page.extract_text() or ""
                    cleaned = _clean_page(raw)
                    if len(cleaned) > 40:
                        pages.append(" ".join(cleaned.split()))
        except Exception as e:
            logger.warning(f"extract_pages failed: {e}")
        return pages

    def extract_image_for_llm(self, file_path: str) -> dict:
        """
        Run OCR and return structured result:
            { "text": str, "lines": [str], "engine": str, "line_count": int }
        Useful when callers want to inspect quality before indexing.
        """
        return self._extract_image_detailed(file_path)

    # ── Internal extraction ───────────────────────────────────────────────────

    def _extract_pdf(self, file_path: str) -> str:
        try:
            import pdfplumber
            parts = []
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    raw = page.extract_text() or ""
                    parts.append(_clean_page(raw))
            return "\n\n".join(parts)
        except Exception as e:
            logger.error(f"PDF extraction failed for {file_path}: {e}")
            return ""

    def _extract_image(self, file_path: str) -> str:
        return self._extract_image_detailed(file_path)["text"]

    def _extract_image_detailed(self, file_path: str) -> dict:
        """
        Full OCR pipeline. Returns dict with text, lines, engine, line_count.
        - High-confidence lines (≥0.40) collected first
        - Low-confidence lines appended after (not discarded)
        - pytesseract used as fallback with PSM 6 for prescription layouts
        """
        # ── 1. PaddleOCR ─────────────────────────────────────────────────────
        try:
            from paddleocr import PaddleOCR
            ocr_engine = self._get_ocr()
            if ocr_engine is None:
                ocr_engine = PaddleOCR(use_angle_cls=True, lang="en")

            raw_result = ocr_engine.ocr(file_path, cls=True)
            kept_lines = []
            low_conf_lines = []

            for page_result in (raw_result or []):
                for det in (page_result or []):
                    if not det or len(det) < 2:
                        continue
                    text_conf = det[1]
                    if not isinstance(text_conf, (list, tuple)) or len(text_conf) < 2:
                        continue
                    text, conf = text_conf
                    text = str(text).strip()
                    if len(text) < _OCR_MIN_LINE_CHARS:
                        continue
                    try:
                        conf_f = float(conf)
                    except Exception:
                        conf_f = 0.0

                    if conf_f >= _OCR_MIN_CONFIDENCE:
                        kept_lines.append(text)
                    else:
                        # Keep low-conf lines too — better to have them
                        low_conf_lines.append(text)

            all_lines = kept_lines + low_conf_lines
            joined = " ".join(all_lines).strip()

            if joined:
                logger.info(
                    f"PaddleOCR: {len(all_lines)} lines "
                    f"({len(kept_lines)} high-conf, {len(low_conf_lines)} low-conf) "
                    f"from {file_path}"
                )
                return {"text": joined, "lines": all_lines, "engine": "paddleocr",
                        "line_count": len(all_lines)}

            logger.info(f"PaddleOCR: no lines from {file_path}, trying pytesseract.")

        except Exception as e:
            logger.warning(f"PaddleOCR failed for {file_path}: {e}")

        # ── 2. pytesseract fallback ───────────────────────────────────────────
        try:
            import pytesseract
            from PIL import Image
            img = Image.open(file_path)
            # PSM 6 = uniform block of text — good for prescription layouts
            custom_cfg = r"--oem 3 --psm 6"
            text = pytesseract.image_to_string(img, config=custom_cfg).strip()

            if text:
                lines = [l.strip() for l in text.splitlines()
                         if len(l.strip()) >= _OCR_MIN_LINE_CHARS]
                logger.info(f"pytesseract: {len(lines)} lines from {file_path}")
                return {"text": text, "lines": lines, "engine": "pytesseract",
                        "line_count": len(lines)}

            logger.warning(f"pytesseract: empty text for {file_path}")

        except Exception as e:
            logger.error(f"pytesseract fallback failed for {file_path}: {e}")

        # ── 3. Both failed ────────────────────────────────────────────────────
        logger.error(
            f"All OCR engines failed for {file_path}. "
            "Install paddleocr or pytesseract."
        )
        return {"text": "", "lines": [], "engine": "failed", "line_count": 0}
