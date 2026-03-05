import numpy as np
import tempfile
from pathlib import Path
from health_ai.rag.vector_store import VectorStore
from health_ai.rag.chunker import TextChunker
from health_ai.embeddings.embedder import EmbeddingModel


def _temp_store():
    tmp_dir = tempfile.mkdtemp()
    store = VectorStore.__new__(VectorStore)
    store.profile_id = "test"
    store.dimension = 384
    store.vector_dir = Path(tmp_dir)
    store.index_path = store.vector_dir / "index.faiss"
    store.meta_path = store.vector_dir / "metadata.json"
    store._load()
    return store


def test_vector_add_and_search():
    model = EmbeddingModel()
    chunker = TextChunker(chunk_size=50, overlap=10)
    store = _temp_store()

    text = "Hypertension is a chronic condition affecting blood pressure. " * 20
    chunks = chunker.chunk_text(text, {"source_type": "test"})

    embeddings = model.embed([c.text for c in chunks])
    store.add(embeddings, chunks)

    query_emb = model.embed(["blood pressure condition"])[0]
    results = store.search(query_emb, top_k=3)

    assert len(results) > 0
    assert "blood pressure" in results[0]["text"].lower()
