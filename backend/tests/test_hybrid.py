from backend.app.search.hybrid import HybridRetriever


class DummyIndex:
    def __init__(self, results):
        self._results = results

    def query(self, text: str, top_k: int = 5):
        return self._results[:top_k]


def test_hybrid_merges_and_orders():
    bm25_results = [
        {"doc_id": "1", "score": 2.0},
        {"doc_id": "2", "score": 1.0},
    ]
    vector_results = [
        {"doc_id": "2", "score": 0.9},
        {"doc_id": "3", "score": 0.1},
    ]

    retriever = HybridRetriever(alpha=0.5, bm25=DummyIndex(bm25_results), vector=DummyIndex(vector_results))
    results = retriever.query("test", top_k=3)

    ids = [r["doc_id"] for r in results]
    assert set(ids) == {"1", "2", "3"}
    assert results[0]["score"] >= results[-1]["score"]
    assert "score_breakdown" in results[0]
