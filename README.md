# Hybrid Search + KPI Dashboard

This project provides an end-to-end hybrid search system with:
- Data ingestion from local files
- BM25 + vector retrieval and hybrid scoring
- FastAPI backend with metrics + logging
- Streamlit dashboard for search, KPIs, and evaluation trends

## Quick start (recommended)

```bash
git clone https://github.com/sudhanshu1239g/assignment.git
cd assignment
mkdir -p data/raw
echo "Demo content for hybrid search." > data/raw/demo.txt
chmod +x up.sh
./up.sh
```

Open:
- Dashboard: `http://127.0.0.1:8501`
- Backend health: `http://127.0.0.1:8000/health`

## 1) Setup (manual)

```bash
./up.sh
```

`up.sh` will:
- create `.venv`
- install dependencies from `requirements.txt`
- build indexes if missing
- start FastAPI and Streamlit concurrently

## 2) Data ingestion (manual)

Place `.txt` and `.md` files into:
```
data/raw
```

### Sample dataset (300+ docs)
If you need a quick 300+ document corpus, you can generate a lightweight synthetic set:
```bash
python - <<'PY'
from pathlib import Path
base = Path("data/raw")
base.mkdir(parents=True, exist_ok=True)
for i in range(1, 301):
    (base / f"doc_{i:03}.txt").write_text(
        f"Sample document {i}\\nThis is synthetic content for evaluation and indexing.\\n",
        encoding="utf-8",
    )
print("Wrote 300 docs to data/raw/")
PY
```
This is only for testing; you can replace with any real dataset you want.

Then run:
```bash
python backend/app/ingest.py
```

Output:
```
data/processed/docs.jsonl
```

## 3) Indexing (manual)

BM25:
```bash
python backend/app/search/bm25.py
```

Vector:
```bash
python backend/app/search/vector.py
```

Outputs:
```
data/index/bm25/bm25.pkl
data/index/vector/index.faiss
data/index/vector/docs.pkl
data/index/vector/metadata.json
```

## 4) FastAPI backend (manual)

Run:
```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Endpoints:
- `GET /health`
- `POST /search`
- `GET /metrics`

`/search` logs to:
```
data/metrics/logs.db
```

## 5) Streamlit dashboard (manual)

Run:
```bash
streamlit run frontend/dashboard.py
```

Pages:
- Search: hybrid results + score breakdown
- KPI: p95 latency + top queries from SQLite logs
- Evaluation: nDCG trend from experiments.csv

## 6) Evaluation

Provide a `qrels.json` file at:
```
data/qrels.json
```

Format (list of queries):
```json
[
  {"query": "example", "relevant": ["doc_id_1", "doc_id_2"]}
]
```

Run:
```bash
python backend/app/eval.py
```

Appends metrics to:
```
data/metrics/experiments.csv
```

## Notes

- All components are CPU-only.
- Vector embeddings use `all-MiniLM-L6-v2`.
- Hybrid score uses min-max normalization:
  `alpha * norm_bm25 + (1 - alpha) * norm_vector`
