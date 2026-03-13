"""Evaluation harness for hybrid search."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any

from .search.hybrid import HybridRetriever

DEFAULT_QRELS = Path("data/qrels.json")
OUT_PATH = Path("data/metrics/experiments.csv")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_qrels(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"qrels.json not found at {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("qrels.json must be a list of query objects")
    return data


def _dcg_at_k(rels: List[int], k: int) -> float:
    dcg = 0.0
    for i, rel in enumerate(rels[:k]):
        # rank i is 0-based; use log2(i+2)
        dcg += (2 ** rel - 1) / (1 if i == 0 else __import__("math").log2(i + 2))
    return dcg


def _ndcg_at_k(rels: List[int], k: int) -> float:
    if not rels:
        return 0.0
    dcg = _dcg_at_k(rels, k)
    ideal = sorted(rels, reverse=True)
    idcg = _dcg_at_k(ideal, k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def _recall_at_k(rels: List[int], k: int) -> float:
    if not rels:
        return 0.0
    total_rel = sum(1 for r in rels if r > 0)
    if total_rel == 0:
        return 0.0
    retrieved_rel = sum(1 for r in rels[:k] if r > 0)
    return retrieved_rel / total_rel


def _extract_doc_id(doc: Dict[str, Any]) -> str | None:
    return doc.get("doc_id") or doc.get("title")


def evaluate(qrels_path: Path = DEFAULT_QRELS, top_k: int = 10, alpha: float = 0.5) -> Dict[str, float]:
    qrels = _load_qrels(qrels_path)
    retriever = HybridRetriever(alpha=alpha)

    ndcgs: List[float] = []
    recalls: List[float] = []

    for q in qrels:
        query = q.get("query")
        relevant = q.get("relevant", [])
        if not query or not isinstance(relevant, list):
            continue

        results = retriever.query(query, top_k=top_k)
        result_ids = [_extract_doc_id(r) for r in results]

        rel_set = set(str(r) for r in relevant)
        rels = [1 if (rid is not None and str(rid) in rel_set) else 0 for rid in result_ids]

        ndcgs.append(_ndcg_at_k(rels, top_k))
        recalls.append(_recall_at_k(rels, top_k))

    avg_ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0

    return {"ndcg@10": avg_ndcg, "recall@10": avg_recall}


def _append_experiment(row: Dict[str, Any]) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = OUT_PATH.exists()

    with OUT_PATH.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    metrics = evaluate()
    row = {
        "timestamp": _timestamp(),
        "ndcg@10": metrics["ndcg@10"],
        "recall@10": metrics["recall@10"],
    }
    _append_experiment(row)
    print(json.dumps(row, ensure_ascii=True))


if __name__ == "__main__":
    main()
