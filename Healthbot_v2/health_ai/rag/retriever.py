from datetime import datetime
from typing import List, Optional
from health_ai.embeddings.embedder import EmbeddingModel
from health_ai.rag.vector_store import VectorStore


class Retriever:

    def __init__(self, profile_id: str, store: Optional[VectorStore] = None):
        self.profile_id = profile_id
        # Accept an injected store to avoid double-loading the FAISS index
        self.store = store if store is not None else VectorStore(profile_id)
        self.embedder = EmbeddingModel()

    def _recency_boost(self, metadata: dict) -> float:
        if "date" not in metadata:
            return 0.0
        try:
            doc_date = datetime.fromisoformat(metadata["date"])
            days_old = (datetime.now() - doc_date).days
            return max(0.0, 1.0 - (days_old / 365))
        except Exception:
            return 0.0

    def _importance_boost(self, metadata: dict) -> float:
        return metadata.get("importance_score", 0.0)

    def retrieve(self, query: str, top_k: int = 6) -> List[dict]:
        query_embedding = self.embedder.embed([query])[0]
        raw_results = self.store.search(query_embedding, top_k=top_k + 4)

        reranked = []
        for r in raw_results:
            score = r["score"]
            metadata = r["metadata"]
            score += self._recency_boost(metadata)
            score += self._importance_boost(metadata)
            reranked.append({
                "score": score,
                "text": r["text"],
                "metadata": metadata,
            })

        reranked.sort(key=lambda x: x["score"], reverse=True)
        return reranked[:top_k]
