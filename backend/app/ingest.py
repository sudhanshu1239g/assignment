"""Ingest raw text/markdown files into JSONL."""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/docs.jsonl")
MAX_CHARS = 10_000


def _load_text(path: Path) -> str:
    # Read as UTF-8; replace invalid bytes to avoid crashes.
    return path.read_text(encoding="utf-8", errors="replace")


def _normalize_text(text: str) -> str:
    cleaned = text.strip()
    if len(cleaned) > MAX_CHARS:
        cleaned = cleaned[:MAX_CHARS]
    return cleaned


def _created_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def ingest() -> int:
    if not RAW_DIR.exists():
        return 0

    files = sorted(
        [p for p in RAW_DIR.rglob("*") if p.is_file() and p.suffix.lower() in {".txt", ".md"}]
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for path in files:
            text = _normalize_text(_load_text(path))
            record = {
                "doc_id": str(uuid.uuid4()),
                "title": path.name,
                "text": text,
                "source": str(path),
                "created_at": _created_at(),
            }
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
            count += 1
    return count


if __name__ == "__main__":
    total = ingest()
    print(f"Wrote {total} documents to {OUT_PATH}")
