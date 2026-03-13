#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV_DIR="$ROOT_DIR/.venv"
REQ_FILE="$ROOT_DIR/requirements.txt"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtual environment at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Limit threads to avoid OpenMP-related crashes on some systems.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONPATH="$ROOT_DIR"

if [[ ! -f "$REQ_FILE" ]]; then
  echo "requirements.txt not found at $REQ_FILE"
  echo "Create it with dependencies (fastapi, uvicorn, streamlit, pandas, rank-bm25, sentence-transformers, faiss-cpu)."
  exit 1
fi

HASH_FILE="$VENV_DIR/.requirements.sha256"
REQ_HASH="$(shasum -a 256 "$REQ_FILE" | awk '{print $1}')"

if [[ ! -f "$HASH_FILE" ]] || [[ "$(cat "$HASH_FILE")" != "$REQ_HASH" ]]; then
  echo "Installing dependencies from requirements.txt"
  pip install -r "$REQ_FILE"
  echo "$REQ_HASH" > "$HASH_FILE"
else
  echo "Dependencies already installed; skipping pip install"
fi

mkdir -p "$ROOT_DIR/data/raw" "$ROOT_DIR/data/processed" "$ROOT_DIR/data/index"

DOCS_JSONL="$ROOT_DIR/data/processed/docs.jsonl"
BM25_INDEX="$ROOT_DIR/data/index/bm25/bm25.pkl"
VECTOR_INDEX="$ROOT_DIR/data/index/vector/index.faiss"

if [[ ! -f "$DOCS_JSONL" ]]; then
  echo "Docs JSONL missing; running ingestion"
  python "$ROOT_DIR/backend/app/ingest.py"
fi

if [[ ! -f "$DOCS_JSONL" ]]; then
  echo "Docs JSONL still missing; please add .txt/.md files to data/raw and rerun."
else
  DOC_COUNT=$(wc -l < "$DOCS_JSONL" | tr -d ' ')
  if [[ "$DOC_COUNT" == "0" ]]; then
    echo "Docs JSONL is empty; add files to data/raw and rerun to build indexes."
  else
  if [[ ! -f "$BM25_INDEX" ]]; then
    echo "BM25 index missing; building"
    python "$ROOT_DIR/backend/app/search/bm25.py"
  fi

  if [[ ! -f "$VECTOR_INDEX" ]]; then
    echo "Vector index missing; building"
    python "$ROOT_DIR/backend/app/search/vector.py"
  fi
  fi
fi

echo "Starting services..."

uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

streamlit run "$ROOT_DIR/frontend/dashboard.py" --server.port 8501 &
FRONTEND_PID=$!

echo "Backend:   http://127.0.0.1:8000"
echo "Frontend:  http://127.0.0.1:8501"

cleanup() {
  echo "Stopping services..."
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
}

trap cleanup EXIT
wait
