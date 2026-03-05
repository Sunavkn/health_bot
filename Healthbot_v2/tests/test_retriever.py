from health_ai.rag.retriever import Retriever
from health_ai.rag.chunker import TextChunker
from health_ai.embeddings.embedder import EmbeddingModel
from health_ai.rag.vector_store import VectorStore
import tempfile, os


def _temp_store():
    """Create a VectorStore backed by a temp directory."""
    tmp_dir = tempfile.mkdtemp()
    store = VectorStore.__new__(VectorStore)
    store.profile_id = "test"
    store.dimension = 384
    from pathlib import Path
    store.vector_dir = Path(tmp_dir)
    store.index_path = store.vector_dir / "index.faiss"
    store.meta_path = store.vector_dir / "metadata.json"
    store._load()
    return store


def test_retrieval_pipeline():
    model = EmbeddingModel()
    chunker = TextChunker(chunk_size=50, overlap=10)
    store = _temp_store()

    text = "Patient has severe hypertension and high blood pressure." * 10
    chunks = chunker.chunk_text(
        text,
        {
            "source_type": "medical_history",
            "date": "2024-01-01",
            "importance_score": 1.0,
        },
    )

    embeddings = model.embed([c.text for c in chunks])
    store.add(embeddings, chunks)

    retriever = Retriever("test", store=store)
    results = retriever.retrieve("blood pressure problem")

    assert len(results) > 0
    assert "hypertension" in results[0]["text"].lower()
