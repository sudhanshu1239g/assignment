# Codex Log (Summary)

## 2026-03-13
- Ingestion: implemented `backend/app/ingest.py` to normalize `.txt`/`.md` into JSONL.
- BM25 indexing: added `backend/app/search/bm25.py` with build/query and pickle persistence.
- Vector indexing: added `backend/app/search/vector.py` with FAISS CPU and sentence-transformers.
- Hybrid scoring: added `backend/app/search/hybrid.py` with min-max normalization.
- API: added FastAPI routes in `backend/app/main.py` and basic Prometheus-style metrics.
- Dashboard: added Streamlit app in `frontend/dashboard.py` (Search/KPI/Eval pages).
- Eval: added `backend/app/eval.py` for nDCG@10 and Recall@10.
- Ops: added `up.sh` automation script and `requirements.txt`.
- Fixes: addressed macOS FAISS stability, import paths, and startup reliability.
