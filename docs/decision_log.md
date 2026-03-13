# Decision Log

## 2026-03-13
- Chose BM25 (rank-bm25) + Sentence Transformers (all-MiniLM-L6-v2) with FAISS CPU for hybrid retrieval because it is lightweight and CPU-friendly.
- Used min-max normalization for hybrid scoring to keep scores comparable across BM25 and vector search.
- Streamlit selected for dashboard speed and simplicity over React.
- SQLite used for local, zero-dependency logging and KPI queries.
