"""Vector indexing and querying using sentence-transformers + faiss-cpu."""

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

import faiss
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path("data/index/vector")
INDEX_PATH = INDEX_DIR / "index.faiss"
DOCS_PATH = INDEX_DIR / "docs.pkl"
METADATA_PATH = INDEX_DIR / "metadata.json"

MODEL_NAME = "all-MiniLM-L6-v2"


def _created_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_docs(docs_path: Path) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    with Path(docs_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs


class VectorIndex:
    def __init__(self, index_dir: Path = INDEX_DIR, model_name: str = MODEL_NAME):
        self.index_dir = Path(index_dir)
        self.index_path = self.index_dir / INDEX_PATH.name
        self.docs_path = self.index_dir / DOCS_PATH.name
        self.metadata_path = self.index_dir / METADATA_PATH.name
        self.model_name = model_name
        self._model = None
        self._index = None
        self._docs = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def build(self, docs_path: Path) -> int:
        docs = _load_docs(docs_path)
        texts = [d.get("text", "") for d in docs]

        model = self._get_model()
        embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)
        # Ensure float32 for faiss compatibility.
        if embeddings.dtype != "float32":
            embeddings = embeddings.astype("float32")

        # Normalize for cosine similarity with inner product.
        faiss.normalize_L2(embeddings)
        dims = embeddings.shape[1]

        index = faiss.IndexFlatIP(dims)
        index.add(embeddings)

        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_path))
        with self.docs_path.open("wb") as f:
            pickle.dump(docs, f)

        metadata = {
            "model": self.model_name,
            "dimensions": int(dims),
            "created_at": _created_at(),
        }
        with self.metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=True, indent=2)

        self._index = index
        self._docs = docs
        return len(docs)

    def _load(self) -> None:
        if not self.index_path.exists() or not self.docs_path.exists():
            raise FileNotFoundError(
                f"Vector index not found at {self.index_dir}. Run build() first."
            )

        self._index = faiss.read_index(str(self.index_path))
        with self.docs_path.open("rb") as f:
            self._docs = pickle.load(f)

    def query(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if top_k <= 0:
            return []
        if self._index is None or self._docs is None:
            self._load()

        model = self._get_model()
        embedding = model.encode([text], show_progress_bar=False)
        if embedding.dtype != "float32":
            embedding = embedding.astype("float32")
        faiss.normalize_L2(embedding)

        scores, indices = self._index.search(embedding, top_k)
        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._docs):
                continue
            doc = self._docs[idx].copy()
            doc["score"] = float(score)
            results.append(doc)
        return results


if __name__ == "__main__":
    index = VectorIndex()
    count = index.build(Path("data/processed/docs.jsonl"))
    print(f"Built vector index with {count} documents at {index.index_dir}")
