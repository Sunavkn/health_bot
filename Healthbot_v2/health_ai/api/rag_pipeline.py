from typing import List, Optional

from health_ai.rag.retriever import Retriever
from health_ai.rag.context_builder import ContextBuilder
from health_ai.model.llm_loader import LLMEngine
from health_ai.rag.vector_store import VectorStore
from health_ai.safety.red_flag import detect_red_flags
from health_ai.safety.disclaimer import DISCLAIMER, URGENT_NOTICE
from health_ai.core.profile_manager import ProfileManager
from health_ai.ingestion.sterling_accuris_parser import parse_sterling_accuris_pdf
from health_ai.core.clinical_analyzer import ClinicalAnalyzer
from health_ai.ingestion.report_formatter import build_lab_summary

import json
from pathlib import Path


# ── System prompts ────────────────────────────────────────────────────────────

GENERAL_SYSTEM_PROMPT = """
You are a medical information assistant.
- Answer in 3–6 clear sentences.
- Do not reference any patient records.
- Do not provide a diagnosis.
- Recommend consulting a doctor for personal concerns.
""".strip()

PERSONAL_SYSTEM_PROMPT = """
You are a clinical lab report reader. The patient's lab data is in [PATIENT DATA] below.

RULES:
- ONLY use values that appear word-for-word in [PATIENT DATA].
- NEVER invent or estimate values from memory.
- A result is HIGH only if [PATIENT DATA] marks it HIGH ↑.
- A result is LOW only if [PATIENT DATA] marks it LOW ↓.
- A result is NORMAL only if [PATIENT DATA] marks it NORMAL or no flag is present.
- Use the reference interval from the data — do NOT apply your own knowledge of ranges.
- Group results by panel.
- Do not skip any test present in the data.
""".strip()

PRESCRIPTION_SYSTEM_PROMPT = """
You are reading a patient's prescription(s). The prescription text is in [PATIENT DATA].

RULES:
- List every medicine/drug prescribed, its dosage, frequency, and duration.
- Include doctor name, clinic, date if present.
- Include diagnosis or complaints if mentioned.
- Format clearly: one medicine per line.
- Do NOT add advice beyond what is written.
- Do NOT skip any medicine.
""".strip()

SYMPTOM_SYSTEM_PROMPT = """
You are a medical guidance assistant.
- Provide safe lifestyle and dietary advice for the reported symptoms.
- Suggest hydration, rest, and basic home care where relevant.
- List warning signs that require urgent care.
- Always recommend consulting a doctor.
- Keep the response concise and actionable.
- Do not give a diagnosis.
""".strip()


# ── Query classifiers ─────────────────────────────────────────────────────────

_PERSONAL_KEYWORDS = frozenset([
    "my", "me", "mine", "i have", "my report", "my prescription",
    "my medication", "my test", "my blood", "my lab", "my result",
    "report", "prescription", "medication", "test result",
    "blood report", "lab",
])

_PRESCRIPTION_KEYWORDS = frozenset([
    "prescription", "prescriptions", "medicine", "medicines",
    "medication", "medications", "tablet", "tablets", "drug", "drugs",
    "prescribed", "doctor prescribed", "what did doctor", "dosage",
    "what do my prescription", "what does my prescription",
])
_SYMPTOM_KEYWORDS = frozenset([
    "i feel", "i am feeling", "i have been", "i've been",
    "pain", "fever", "cough", "headache", "vomiting",
    "nausea", "dizziness", "fatigue", "weakness",
])
_SUMMARY_KEYWORDS = frozenset([
    "summary", "all", "everything", "complete", "full",
    "entire", "list all", "show all", "all results",
    "all numbers", "all tests", "give me all", "overview",
])


class RAGPipeline:

    def __init__(self, profile_id: str):
        self.profile_id = profile_id
        self.store = VectorStore(profile_id)
        self.retriever = Retriever(profile_id, store=self.store)
        self.context_builder = ContextBuilder()
        self.llm = LLMEngine()
        self._cached_summary: Optional[str] = None

    # ── Query classification ──────────────────────────────────────────────────

    def classify_query(self, query: str) -> str:
        q = query.lower()
        if any(k in q for k in _SYMPTOM_KEYWORDS):
            return "symptom"
        is_personal = any(k in q for k in _PERSONAL_KEYWORDS)
        if is_personal:
            if any(k in q for k in _SUMMARY_KEYWORDS):
                return "summary"
            # Prescription-specific query → dedicated path
            if any(k in q for k in _PRESCRIPTION_KEYWORDS):
                return "prescription"
            return "personal"
        return "general"

    # ── Prescription retrieval ───────────────────────────────────────────────

    def _build_prescription_response(
        self, query: str, history: list
    ) -> str:
        """
        Retrieve ALL prescription chunks from the vector store and feed them
        to the LLM for formatting. Uses source_type filter so we only get
        prescription chunks, not lab report chunks.
        """
        # Get all stored chunks and filter for prescriptions
        all_chunks = []
        for key in sorted(self.store.metadata.keys(), key=lambda x: int(x)):
            entry = self.store.metadata[key]
            src = entry["metadata"].get("source_type", "")
            doc_type = entry["metadata"].get("document_type", "")
            if src == "prescription" or doc_type == "prescription" or "prescription" in src:
                all_chunks.append(entry["text"])

        if not all_chunks:
            # Fallback: semantic search in case source_type wasn't set correctly
            retrieved = self.retriever.retrieve(
                "prescription medicine tablet dosage", top_k=10
            )
            all_chunks = [c["text"] for c in retrieved if c.get("text", "").strip()]

        if not all_chunks:
            return (
                "No prescription data found in your records. "
                "Please upload prescription images via the /upload/{profile_id} endpoint."
            )

        prescription_text = "\n\n---\n\n".join(all_chunks)
        context = f"[PATIENT DATA]\n{prescription_text}\n\n[QUESTION]\n{query}"

        return self.llm.generate(
            system_prompt=PRESCRIPTION_SYSTEM_PROMPT,
            user_prompt=context,
            max_tokens=600,
        )

    # ── Deterministic summary (no LLM) ───────────────────────────────────────

    def _build_deterministic_summary(self) -> str:
        """
        Build the lab summary entirely in Python from the parsed + analysed data.
        No LLM involved → no truncation, no misinterpretation, no hallucination.

        Strategy:
        1. Find all PDF lab reports in the profile's raw_documents directory.
        2. Parse each with the Sterling Accuris parser (falls back to generic).
        3. Run ClinicalAnalyzer to classify every test.
        4. Format with report_formatter.build_lab_summary().
        """
        if self._cached_summary:
            return self._cached_summary

        profile_dir = ProfileManager.get_profile_dir(self.profile_id)
        raw_dir = profile_dir / "raw_documents"

        all_tests = []
        found_reports = []

        for pdf_path in sorted(raw_dir.rglob("*.pdf")):
            try:
                tests = parse_sterling_accuris_pdf(str(pdf_path))
                if tests:
                    all_tests.extend(tests)
                    found_reports.append(pdf_path.name)
            except Exception as e:
                pass

        if not all_tests:
            return "No lab report data found. Please upload a lab report PDF first."

        analysis = ClinicalAnalyzer().analyze(all_tests)
        summary = build_lab_summary(analysis)

        if len(found_reports) == 1:
            header = f"_Report: {found_reports[0]}_\n\n"
        else:
            header = f"_Reports: {', '.join(found_reports)}_\n\n"

        self._cached_summary = header + summary
        return self._cached_summary

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def run(self, query: str, history: Optional[List[str]] = None) -> str:
        if history is None:
            history = []

        intent = self.classify_query(query)

        if intent == "general":
            response = self.llm.generate(
                system_prompt=GENERAL_SYSTEM_PROMPT,
                user_prompt=query,
                max_tokens=350,
            )

        elif intent == "symptom":
            response = self.llm.generate(
                system_prompt=SYMPTOM_SYSTEM_PROMPT,
                user_prompt=query,
                max_tokens=400,
            )

        elif intent == "prescription":
            response = self._build_prescription_response(query, history)

        elif intent == "summary":
            # Pure Python — no LLM, no token limits, guaranteed complete
            response = self._build_deterministic_summary()

        else:  # personal — standard RAG with LLM
            retrieved = self.retriever.retrieve(query, top_k=8)
            context_text = self.context_builder.build_context(
                retrieved, history, query
            )
            response = self.llm.generate(
                system_prompt=PERSONAL_SYSTEM_PROMPT,
                user_prompt=context_text,
                max_tokens=500,
            )

        # ── Safety layer ──────────────────────────────────────────────────────
        if detect_red_flags(query):
            response += "\n\n" + URGENT_NOTICE

        response += "\n\n" + DISCLAIMER

        return response
