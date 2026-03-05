import faiss
import json
import numpy as np
from pathlib import Path
from typing import List

from health_ai.core.profile_manager import ProfileManager


class VectorStore:

    def __init__(self, profile_id: str):
        self.profile_id = profile_id
        self.dimension = 384  # must match embedding model dimension

        # Profile-scoped directory
        self.vector_dir = ProfileManager.get_vector_dir(profile_id)
        self.vector_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.vector_dir / "index.faiss"
        self.meta_path = self.vector_dir / "metadata.json"

        self._load()

    def _create_index(self):
        # Using cosine similarity (inner product)
        self.index = faiss.IndexFlatIP(self.dimension)

    def _load(self):
        # Load or create FAISS index
        print("VectorStore loading for profile:", self.profile_id)
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        else:
            self._create_index()

        # Load metadata
        if self.meta_path.exists():
            with open(self.meta_path, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

    def _save(self):
        faiss.write_index(self.index, str(self.index_path))

        with open(self.meta_path, "w") as f:
            json.dump(self.metadata, f, indent=4)

    def add(self, embeddings: np.ndarray, chunks: List):

        if len(embeddings) == 0:
            return

        embeddings = np.array(embeddings).astype("float32")

        start_id = len(self.metadata)

        self.index.add(embeddings)

        for i, chunk in enumerate(chunks):
            vector_id = str(start_id + i)
            self.metadata[vector_id] = {
                "text": chunk.text,
                "metadata": chunk.metadata
            }

        self._save()

    def search(self, query_embedding: np.ndarray, top_k: int = 5):

        if self.index.ntotal == 0:
            return []

        query_embedding = np.array([query_embedding]).astype("float32")

        scores, indices = self.index.search(query_embedding, top_k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            vector_id = str(idx)

            if vector_id in self.metadata:
                entry = self.metadata[vector_id]
                results.append({
                    "score": float(score),
                    "text": entry["text"],
                    "metadata": entry["metadata"]
                })

        return results

    def size(self):
        return self.index.ntotal
