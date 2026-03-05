"""
server.py
FastAPI application for the Health AI backend.

Changes vs original:
  - Startup now auto-detects LAN IP and prints a connection URL for the mobile app
  - StartupIndexer called with force_reindex_images=True on first run (env-flag controlled)
  - New endpoint: POST /ocr/analyze/{profile_id}
      Accepts an image upload, runs OCR, sends OCR text + query to the LLM,
      returns tailored AI response + raw OCR lines.  No indexing — pure real-time.
  - New endpoint: GET /server/info
      Returns host IP, port, profile, and vector count — useful for the mobile app
      to verify connectivity and current state.
  - Upload endpoint now saves prescription images to the correct profile folder
    AND marks them for re-indexing so subsequent startup picks them up.
"""

import os
import shutil
import socket
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from health_ai.ingestion.ingest import IngestionEngine
from health_ai.api.rag_pipeline import RAGPipeline
from health_ai.core.exceptions import SchemaValidationError
from health_ai.ingestion.startup_indexer import StartupIndexer
from health_ai.rag.vector_store import VectorStore
from health_ai.utils.document_reader import DocumentReader
from health_ai.core.profile_manager import ProfileManager

PROFILE_ID = "profile-1"

# Set FORCE_REINDEX_IMAGES=1 in the environment to re-run OCR on all images.
_FORCE_REINDEX = os.environ.get("FORCE_REINDEX_IMAGES", "0") == "1"

app = FastAPI(
    title="Health AI Backend",
    description="Offline medical AI — RAG over personal health documents.",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pipeline cache — loaded once at startup, reused per request.
_pipeline_cache: Dict[str, RAGPipeline] = {}


def _get_local_ip() -> str:
    """Best-effort detection of LAN IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


@app.on_event("startup")
def startup_event():
    port = int(os.environ.get("PORT", 8000))
    lan_ip = _get_local_ip()

    print("\n" + "=" * 55)
    print("  Health AI Backend starting up")
    print("=" * 55)
    print(f"  Local:   http://localhost:{port}")
    print(f"  Network: http://{lan_ip}:{port}   ← use this on the phone")
    print("=" * 55)

    print("\n---- STARTUP INDEXING ----")
    indexer = StartupIndexer(PROFILE_ID, force_reindex_images=_FORCE_REINDEX)
    indexer.scan_and_index()
    _pipeline_cache[PROFILE_ID] = RAGPipeline(PROFILE_ID)
    print("---- STARTUP COMPLETE ----\n")


def _get_pipeline(profile_id: str) -> RAGPipeline:
    if profile_id not in _pipeline_cache:
        _pipeline_cache[profile_id] = RAGPipeline(profile_id)
    return _pipeline_cache[profile_id]


def _invalidate(profile_id: str):
    """Drop cached pipeline so next query picks up freshly indexed data."""
    _pipeline_cache.pop(profile_id, None)


# ── Server info (for mobile app discovery) ───────────────────────────────────

@app.get("/server/info")
def server_info():
    """
    Returns server metadata useful for the mobile app to display status
    and verify it's talking to the right backend.
    """
    port = int(os.environ.get("PORT", 8000))
    store = VectorStore(PROFILE_ID)
    return {
        "server": "Health AI Backend",
        "version": "2.1.0",
        "profile_id": PROFILE_ID,
        "lan_ip": _get_local_ip(),
        "port": port,
        "vectors_indexed": store.size(),
    }


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


# ── Profile status ────────────────────────────────────────────────────────────

@app.get("/profiles/{profile_id}/status")
def profile_status(profile_id: str):
    """Return how many vectors are stored for this profile."""
    store = VectorStore(profile_id)
    return {"profile_id": profile_id, "vectors_indexed": store.size()}


# ── Ingest: structured JSON data ──────────────────────────────────────────────

@app.post("/ingest/medical-history/{profile_id}")
def ingest_medical_history(profile_id: str, data: Dict[str, Any]):
    try:
        result = IngestionEngine(profile_id).ingest_medical_history(data)
        _invalidate(profile_id)
        return result
    except SchemaValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/hospitalizations/{profile_id}")
def ingest_hospitalizations(profile_id: str, data: Dict[str, Any]):
    try:
        result = IngestionEngine(profile_id).ingest_hospitalizations(data)
        _invalidate(profile_id)
        return result
    except SchemaValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/prescriptions/{profile_id}")
def ingest_prescriptions(profile_id: str, data: Dict[str, Any]):
    try:
        result = IngestionEngine(profile_id).ingest_prescriptions(data)
        _invalidate(profile_id)
        return result
    except SchemaValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/family-history/{profile_id}")
def ingest_family_history(profile_id: str, data: Dict[str, Any]):
    try:
        result = IngestionEngine(profile_id).ingest_family_history(data)
        _invalidate(profile_id)
        return result
    except SchemaValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/daily-log/{profile_id}")
def ingest_daily_log(profile_id: str, data: Dict[str, Any]):
    try:
        result = IngestionEngine(profile_id).ingest_daily_log(data)
        _invalidate(profile_id)
        return result
    except SchemaValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/document/{profile_id}")
def ingest_document(profile_id: str, data: Dict[str, Any]):
    """Ingest a document by server-side file path."""
    if "file_path" not in data:
        raise HTTPException(status_code=400, detail="Missing 'file_path' field")
    try:
        result = IngestionEngine(profile_id).ingest_document(data["file_path"])
        _invalidate(profile_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Ingest: file upload ───────────────────────────────────────────────────────

@app.post("/upload/{profile_id}")
async def upload_document(
    profile_id: str,
    file: UploadFile = File(...),
    doc_type: Optional[str] = None,
):
    """
    Upload a PDF, PNG, JPG, or JPEG document directly.

    For prescription images the file is ALSO saved to the profile's
    raw_documents/prescriptions/ folder so future startups re-index it.

    Query param `doc_type`: "prescription" | "lab_report" | auto-detect
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".png", ".jpg", ".jpeg"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: pdf, png, jpg, jpeg.",
        )

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # Determine document type for folder routing
        detected_type = doc_type or (
            "prescription" if suffix in {".png", ".jpg", ".jpeg"} else "lab_report"
        )

        # For prescription images, save to the persistent prescriptions folder
        # so startup_indexer picks them up on future restarts too
        if detected_type == "prescription":
            profile_dir = ProfileManager.get_profile_dir(profile_id)
            presc_dir = profile_dir / "raw_documents" / "prescriptions"
            presc_dir.mkdir(parents=True, exist_ok=True)
            dest_path = presc_dir / file.filename
            shutil.copy2(tmp_path, dest_path)

        result = IngestionEngine(profile_id).ingest_document(
            tmp_path,
            source_type=detected_type,
            importance_score=0.9 if detected_type == "prescription" else 0.8,
        )
        _invalidate(profile_id)
        return {**result, "filename": file.filename, "doc_type": detected_type}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


# ── OCR: real-time prescription analysis ─────────────────────────────────────

@app.post("/ocr/analyze/{profile_id}")
async def ocr_analyze(
    profile_id: str,
    file: UploadFile = File(...),
    query: str = "What medicines are prescribed? List all with dosage and instructions.",
):
    """
    Real-time OCR + AI analysis of an uploaded prescription image.

    This endpoint does NOT index the document — it just:
      1. Runs OCR on the uploaded image
      2. Sends the OCR text + query to the LLM with the prescription prompt
      3. Returns the AI response AND the raw OCR lines for the frontend to display

    Use this for instant prescription reading without touching the vector store.
    Use /upload/{profile_id} if you want the document permanently indexed.

    Body (multipart/form-data):
        file  – image file (PNG, JPG, JPEG)
        query – optional question to ask about the prescription
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg"}:
        raise HTTPException(
            status_code=400,
            detail=f"Only image files supported here (png/jpg/jpeg). Got '{suffix}'.",
        )

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        reader = DocumentReader()
        ocr_result = reader.extract_image_for_llm(tmp_path)
        ocr_text = ocr_result["text"]
        ocr_engine = ocr_result["engine"]
        ocr_lines = ocr_result["lines"]
        line_count = ocr_result["line_count"]

        if not ocr_text or len(ocr_text.strip()) < 10:
            return {
                "status": "ocr_failed",
                "ocr_engine": ocr_engine,
                "ocr_lines": [],
                "ocr_line_count": 0,
                "ai_response": (
                    "OCR could not extract readable text from this image. "
                    "Please ensure the image is clear and well-lit, then try again."
                ),
            }

        # Build a tailored AI response for THIS specific image
        pipeline = _get_pipeline(profile_id)
        from health_ai.api.rag_pipeline import PRESCRIPTION_SYSTEM_PROMPT
        from health_ai.safety.disclaimer import DISCLAIMER

        user_prompt = (
            f"[PRESCRIPTION IMAGE OCR TEXT]\n{ocr_text}\n\n"
            f"[QUESTION]\n{query}"
        )

        ai_response = pipeline.llm.generate(
            system_prompt=PRESCRIPTION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=600,
        )
        ai_response += "\n\n" + DISCLAIMER

        return {
            "status": "success",
            "filename": file.filename,
            "ocr_engine": ocr_engine,
            "ocr_line_count": line_count,
            "ocr_lines": ocr_lines,
            "ai_response": ai_response,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


# ── Query ─────────────────────────────────────────────────────────────────────

@app.post("/query/{profile_id}")
def query(profile_id: str, data: Dict[str, Any]):
    """
    Run a RAG query.
    Body: { "query": "...", "history": ["prev Q", "prev A", ...] }
    """
    if "query" not in data:
        raise HTTPException(status_code=400, detail="Missing 'query' field")

    history: List[str] = data.get("history", [])

    try:
        pipeline = _get_pipeline(profile_id)
        response = pipeline.run(data["query"], history=history)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── On-device RAG support endpoints ──────────────────────────────────────────
# These three endpoints power the mobile app's on-device RAG:
#   1. /upload-and-embed  → OCR + chunk + embed on server, return vectors to phone
#   2. /embed-query       → embed a query string, return vector to phone
#   3. /generate          → LLM-only with phone-provided chunks as context

@app.post("/upload-and-embed/{profile_id}")
async def upload_and_embed(
    profile_id: str,
    file: UploadFile = File(...),
    doc_type: Optional[str] = None,
):
    """
    Upload a document. The server:
      1. Runs OCR / text extraction
      2. Chunks the text
      3. Computes embeddings
      4. Indexes in the server-side FAISS store (so server RAG still works)
      5. Returns all chunks + embeddings as JSON to the phone

    The phone stores these locally and uses them for on-device cosine similarity.
    """
    from health_ai.rag.chunker import TextChunker
    from health_ai.embeddings.embedder import EmbeddingModel
    from health_ai.utils.prescription_cleaner import clean_prescription_text

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".png", ".jpg", ".jpeg"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: pdf, png, jpg, jpeg.",
        )

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # ── 1. Extract text ───────────────────────────────────────────────────
        reader = DocumentReader()
        is_image = suffix in {".png", ".jpg", ".jpeg"}

        if is_image:
            ocr_result = reader.extract_image_for_llm(tmp_path)
            raw_text = ocr_result["text"]
            ocr_engine = ocr_result.get("engine", "unknown")
            ocr_lines = ocr_result.get("line_count", 0)
        else:
            raw_text = reader.extract_text(tmp_path)
            ocr_engine = "pdfplumber"
            ocr_lines = len(raw_text.splitlines())

        min_chars = 15 if is_image else 50
        if not raw_text or len(raw_text.strip()) < min_chars:
            return {
                "status": "failed",
                "reason": "Could not extract readable text from this file.",
                "ocr_engine": ocr_engine,
            }

        # ── 2. Detect type + clean ────────────────────────────────────────────
        detected_type = doc_type or ("prescription" if is_image else "lab_report")
        text = clean_prescription_text(raw_text) if detected_type == "prescription" else raw_text

        # ── 3. Chunk ──────────────────────────────────────────────────────────
        base_metadata = {
            "profile_id": profile_id,
            "source_type": detected_type,
            "document_type": detected_type,
            "filename": file.filename,
            "importance_score": 0.9 if detected_type == "prescription" else 1.0,
        }
        chunker = TextChunker()
        chunks = chunker.chunk_text(text, base_metadata)

        if not chunks:
            return {"status": "failed", "reason": "No chunks produced from extracted text."}

        # ── 4. Embed ──────────────────────────────────────────────────────────
        embedder = EmbeddingModel()
        embeddings = embedder.embed([c.text for c in chunks])

        # ── 5. Also index server-side so /query still works ───────────────────
        try:
            engine = IngestionEngine(profile_id)
            engine.ingest_document(
                tmp_path,
                source_type=detected_type,
                importance_score=base_metadata["importance_score"],
            )
            _invalidate(profile_id)

            # Save prescription images persistently
            if detected_type == "prescription":
                profile_dir = ProfileManager.get_profile_dir(profile_id)
                presc_dir = profile_dir / "raw_documents" / "prescriptions"
                presc_dir.mkdir(parents=True, exist_ok=True)
                dest = presc_dir / file.filename
                import shutil as _sh
                _sh.copy2(tmp_path, dest)
        except Exception as idx_err:
            # Non-fatal — phone still gets the chunks even if server index fails
            print(f"[warn] Server-side indexing failed: {idx_err}")

        # ── 6. Return chunks + embeddings to phone ────────────────────────────
        chunk_data = [
            {
                "text": chunk.text,
                "embedding": emb.tolist(),
                "metadata": chunk.metadata,
            }
            for chunk, emb in zip(chunks, embeddings)
        ]

        return {
            "status": "success",
            "filename": file.filename,
            "doc_type": detected_type,
            "ocr_engine": ocr_engine,
            "ocr_line_count": ocr_lines,
            "chunk_count": len(chunks),
            "chunks": chunk_data,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


@app.post("/embed-query")
def embed_query(data: Dict[str, Any]):
    """
    Embed a query string using the same model as the server.
    Returns a 384-dim normalized float vector.
    Body: { "query": "..." }
    """
    if "query" not in data:
        raise HTTPException(status_code=400, detail="Missing 'query' field")
    try:
        from health_ai.embeddings.embedder import EmbeddingModel
        embedder = EmbeddingModel()
        emb = embedder.embed([data["query"]])[0]
        return {"embedding": emb.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/{profile_id}")
def generate_response(profile_id: str, data: Dict[str, Any]):
    """
    LLM-only endpoint. The phone supplies retrieved chunks (already selected
    by on-device cosine similarity). The server just runs the LLM.

    Body: {
        "query":   "...",
        "chunks":  ["chunk text 1", "chunk text 2", ...],
        "history": ["prev question", "prev answer", ...]
    }
    """
    if "query" not in data:
        raise HTTPException(status_code=400, detail="Missing 'query' field")

    query   = data["query"]
    chunks  = data.get("chunks", [])
    history = data.get("history", [])

    try:
        from health_ai.api.rag_pipeline import (
            GENERAL_SYSTEM_PROMPT,
            PERSONAL_SYSTEM_PROMPT,
            PRESCRIPTION_SYSTEM_PROMPT,
            SYMPTOM_SYSTEM_PROMPT,
        )
        from health_ai.safety.disclaimer import DISCLAIMER, URGENT_NOTICE
        from health_ai.safety.red_flag import detect_red_flags

        pipeline = _get_pipeline(profile_id)
        intent   = pipeline.classify_query(query)

        if intent == "general" or not chunks:
            response = pipeline.llm.generate(
                system_prompt=GENERAL_SYSTEM_PROMPT,
                user_prompt=query,
                max_tokens=350,
            )
        elif intent == "symptom":
            response = pipeline.llm.generate(
                system_prompt=SYMPTOM_SYSTEM_PROMPT,
                user_prompt=query,
                max_tokens=400,
            )
        else:
            context_text = "\n---\n".join(chunks)
            system_prompt = (
                PRESCRIPTION_SYSTEM_PROMPT
                if intent == "prescription"
                else PERSONAL_SYSTEM_PROMPT
            )
            user_prompt = (
                f"[PATIENT DATA]\n{context_text}\n\n"
                f"[QUESTION]\n{query}"
            )
            response = pipeline.llm.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=500,
            )

        if detect_red_flags(query):
            response += "\n\n" + URGENT_NOTICE
        response += "\n\n" + DISCLAIMER

        return {"response": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
