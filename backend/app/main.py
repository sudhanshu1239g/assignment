"""FastAPI app exposing health, search, and metrics endpoints."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import List, Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .search.hybrid import HybridRetriever

VERSION = "0.1.0"
LOG_DB_PATH = "data/metrics/logs.db"
GIT_HEAD_PATH = ".git/HEAD"

app = FastAPI(title="Hybrid Search API", version=VERSION)

_metrics_lock = Lock()
_metrics = {
    "search_count": 0,
    "search_latency_sum": 0.0,
    "search_latency_last": 0.0,
}


def _update_metrics(latency_seconds: float) -> None:
    with _metrics_lock:
        _metrics["search_count"] += 1
        _metrics["search_latency_sum"] += latency_seconds
        _metrics["search_latency_last"] = latency_seconds


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()

def _read_commit_hash() -> str:
    try:
        head = Path(GIT_HEAD_PATH).read_text(encoding="utf-8").strip()
        if head.startswith("ref:"):
            ref_path = head.split(" ", 1)[1].strip()
            return Path(".git", ref_path).read_text(encoding="utf-8").strip()
        return head
    except Exception:
        return "unknown"


def _ensure_log_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS search_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT NOT NULL,
            query TEXT NOT NULL,
            top_k INTEGER NOT NULL,
            alpha REAL NOT NULL,
            latency_ms REAL NOT NULL,
            result_count INTEGER NOT NULL,
            error TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    existing = {row[1] for row in conn.execute("PRAGMA table_info(search_logs)").fetchall()}
    required = {
        "request_id": "TEXT NOT NULL DEFAULT ''",
        "result_count": "INTEGER NOT NULL DEFAULT 0",
        "latency_ms": "REAL NOT NULL DEFAULT 0",
        "error": "TEXT",
    }
    for col, decl in required.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE search_logs ADD COLUMN {col} {decl}")


def _log_search(
    query: str,
    top_k: int,
    alpha: float,
    latency_seconds: float,
    result_count: int,
    error: str | None = None,
) -> None:
    Path(LOG_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(LOG_DB_PATH)
    try:
        _ensure_log_schema(conn)
        request_id = str(uuid.uuid4())
        latency_ms = latency_seconds * 1000.0
        conn.execute(
            "INSERT INTO search_logs (request_id, query, top_k, alpha, latency_ms, result_count, error, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (request_id, query, top_k, alpha, latency_ms, result_count, error, _utc_timestamp()),
        )
        conn.commit()

        log_payload = {
            "request_id": request_id,
            "query": query,
            "top_k": top_k,
            "alpha": alpha,
            "latency_ms": latency_ms,
            "result_count": result_count,
            "error": error,
        }
        print(json.dumps(log_payload, ensure_ascii=True))
    finally:
        conn.close()


def _tokenize_query(text: str) -> List[str]:
    return [t for t in text.lower().split() if t]


def _build_highlight(text: str, query: str, window: int = 160) -> str:
    if not text:
        return ""
    terms = _tokenize_query(query)
    if not terms:
        return text[:window]

    lower_text = text.lower()
    best_idx = None
    best_term = None
    for term in terms:
        idx = lower_text.find(term)
        if idx != -1 and (best_idx is None or idx < best_idx):
            best_idx = idx
            best_term = term

    if best_idx is None:
        return text[:window]

    start = max(0, best_idx - window // 2)
    end = min(len(text), best_idx + len(best_term) + window // 2)
    snippet = text[start:end]

    # Emphasize the first matched term in the snippet.
    lower_snippet = snippet.lower()
    rel_idx = lower_snippet.find(best_term)
    if rel_idx != -1:
        rel_end = rel_idx + len(best_term)
        snippet = (
            snippet[:rel_idx]
            + "<em>"
            + snippet[rel_idx:rel_end]
            + "</em>"
            + snippet[rel_end:]
        )

    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    return snippet


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=100)
    alpha: float = Field(0.5, ge=0.0, le=1.0)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "OK", "version": VERSION, "commit": _read_commit_hash()}


@app.post("/search")
def search(req: SearchRequest) -> Dict[str, Any]:
    start = time.perf_counter()
    retriever = HybridRetriever(alpha=req.alpha)
    error = None
    try:
        results = retriever.query(req.query, top_k=req.top_k)
    except Exception as exc:
        results = []
        error = str(exc)

    # Add highlights to results.
    for r in results:
        r["highlight"] = _build_highlight(r.get("text", ""), req.query)

    latency = time.perf_counter() - start
    _update_metrics(latency)
    _log_search(req.query, req.top_k, req.alpha, latency, len(results), error=error)
    return {
        "query": req.query,
        "top_k": req.top_k,
        "alpha": req.alpha,
        "results": results,
        "latency_seconds": latency,
    }


@app.get("/metrics")
def metrics() -> str:
    with _metrics_lock:
        count = _metrics["search_count"]
        total = _metrics["search_latency_sum"]
        last = _metrics["search_latency_last"]

    lines = [
        "# HELP search_latency_seconds Search request latency in seconds.",
        "# TYPE search_latency_seconds summary",
        f"search_latency_seconds_count {count}",
        f"search_latency_seconds_sum {total}",
        f"search_latency_seconds_last {last}",
    ]
    return "\n".join(lines) + "\n"
