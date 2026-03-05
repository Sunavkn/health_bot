from typing import List
import uuid

class Chunk:
    def __init__(self, text: str, metadata: dict):
        self.id = str(uuid.uuid4())
        self.text = text
        self.metadata = metadata

class TextChunker:

    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str, base_metadata: dict) -> List[Chunk]:
        words = text.split()
        chunks = []

        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            metadata = base_metadata.copy()
            chunk = Chunk(chunk_text, metadata)
            chunks.append(chunk)

            start += self.chunk_size - self.overlap

        return chunks
