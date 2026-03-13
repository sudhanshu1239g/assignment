"""BM25 indexing and querying using rank-bm25."""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any

from rank_bm25 import BM25Okapi

INDEX_DIR = Path("data/index/bm25")
INDEX_PATH = INDEX_DIR / "bm25.pkl"


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


class BM25Index:
    def __init__(self, index_path: Path = INDEX_PATH):
        self.index_path = index_path
        self._bm25 = None
        self._docs = None
        self._corpus_tokens = None

    def build(self, docs_path: Path) -> int:
        docs_path = Path(docs_path)
        if not docs_path.exists():
            raise FileNotFoundError(f"Docs file not found: {docs_path}")

        docs: List[Dict[str, Any]] = []
        corpus_tokens: List[List[str]] = []

        with docs_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                text = doc.get("text", "")
                tokens = _tokenize(text)
                docs.append(doc)
                corpus_tokens.append(tokens)

        bm25 = BM25Okapi(corpus_tokens)

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with self.index_path.open("wb") as f:
            pickle.dump({"bm25": bm25, "docs": docs, "corpus_tokens": corpus_tokens}, f)

        self._bm25 = bm25
        self._docs = docs
        self._corpus_tokens = corpus_tokens
        return len(docs)

    def _load(self) -> None:
        if self.index_path.exists():
            with self.index_path.open("rb") as f:
                data = pickle.load(f)
            self._bm25 = data["bm25"]
            self._docs = data["docs"]
            self._corpus_tokens = data.get("corpus_tokens")
        else:
            raise FileNotFoundError(
                f"BM25 index not found at {self.index_path}. Run build() first."
            )

    def query(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self._bm25 is None or self._docs is None:
            self._load()

        tokens = _tokenize(text)
        scores = self._bm25.get_scores(tokens)

        if top_k <= 0:
            return []

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            doc = self._docs[idx].copy()
            doc["score"] = float(scores[idx])
            results.append(doc)
        return results


if __name__ == "__main__":
    index = BM25Index()
    count = index.build(Path("data/processed/docs.jsonl"))
    print(f"Built BM25 index with {count} documents at {index.index_path}")
