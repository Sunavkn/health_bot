from health_ai.rag.chunker import TextChunker

def test_chunking():
    text = "word " * 1200
    chunker = TextChunker(chunk_size=200, overlap=50)
    chunks = chunker.chunk_text(text, {"source_type": "test"})
    assert len(chunks) > 1
    assert chunks[0].metadata["source_type"] == "test"
