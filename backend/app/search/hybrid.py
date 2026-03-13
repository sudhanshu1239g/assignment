"""Hybrid retriever that merges BM25 and Vector search results."""

from typing import List, Dict, Any

from .bm25 import BM25Index
from .vector import VectorIndex


def _min_max_normalize(scores: List[float]) -> List[float]:
    if not scores:
        return []
    min_s = min(scores)
    max_s = max(scores)
    if max_s == min_s:
        return [0.0 for _ in scores]
    return [(s - min_s) / (max_s - min_s) for s in scores]


class HybridRetriever:
    def __init__(self, alpha: float = 0.5, bm25: BM25Index | None = None, vector: VectorIndex | None = None):
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be between 0.0 and 1.0")
        self.alpha = alpha
        self.bm25 = bm25 or BM25Index()
        self.vector = vector or VectorIndex()

    def query(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if top_k <= 0:
            return []

        bm25_results = self.bm25.query(text, top_k=top_k)
        vector_results = self.vector.query(text, top_k=top_k)

        bm25_scores = [r.get("score", 0.0) for r in bm25_results]
        vector_scores = [r.get("score", 0.0) for r in vector_results]

        bm25_norm = _min_max_normalize(bm25_scores)
        vector_norm = _min_max_normalize(vector_scores)

        bm25_map = {}
        for r, s in zip(bm25_results, bm25_norm):
            key = r.get("doc_id") or r.get("title")
            if key is None:
                continue
            bm25_map[key] = {"doc": r, "norm": s}

        vector_map = {}
        for r, s in zip(vector_results, vector_norm):
            key = r.get("doc_id") or r.get("title")
            if key is None:
                continue
            vector_map[key] = {"doc": r, "norm": s}

        all_keys = set(bm25_map) | set(vector_map)
        merged: List[Dict[str, Any]] = []

        for key in all_keys:
            bm25_entry = bm25_map.get(key)
            vector_entry = vector_map.get(key)

            bm25_norm_score = bm25_entry["norm"] if bm25_entry else 0.0
            vector_norm_score = vector_entry["norm"] if vector_entry else 0.0

            hybrid_score = self.alpha * bm25_norm_score + (1.0 - self.alpha) * vector_norm_score

            base_doc = None
            if bm25_entry is not None:
                base_doc = bm25_entry["doc"].copy()
            elif vector_entry is not None:
                base_doc = vector_entry["doc"].copy()
            else:
                continue

            base_doc["score"] = float(hybrid_score)
            base_doc["score_breakdown"] = {
                "bm25": float(bm25_norm_score),
                "vector": float(vector_norm_score),
                "alpha": float(self.alpha),
            }
            merged.append(base_doc)

        merged.sort(key=lambda d: d.get("score", 0.0), reverse=True)
        return merged[:top_k]


if __name__ == "__main__":
    retriever = HybridRetriever(alpha=0.5)
    results = retriever.query("example query", top_k=5)
    print(f"Returned {len(results)} results")
