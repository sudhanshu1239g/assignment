"""Streamlit dashboard for search, KPIs, and evaluation trends."""

from __future__ import annotations

import sys
import sqlite3
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.search.hybrid import HybridRetriever

LOG_DB_PATH = Path("data/metrics/logs.db")
EXPERIMENTS_CSV = Path("data/metrics/experiments.csv")


st.set_page_config(page_title="Hybrid Search Dashboard", layout="wide")


@st.cache_resource
def _get_retriever(alpha: float) -> HybridRetriever:
    return HybridRetriever(alpha=alpha)


def _detect_log_table(conn: sqlite3.Connection) -> Optional[Tuple[str, List[str]]]:
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    for (name,) in tables:
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({name})").fetchall()]
        lower_cols = {c.lower() for c in cols}
        if "query" in lower_cols and (
            "latency_ms" in lower_cols
            or "latency_seconds" in lower_cols
            or "latency" in lower_cols
        ):
            return name, cols
    return None


def _load_logs(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()

    conn = sqlite3.connect(str(db_path))
    try:
        table_info = _detect_log_table(conn)
        if not table_info:
            return pd.DataFrame()
        table, cols = table_info
        cols_lower = {c.lower(): c for c in cols}

        query_col = cols_lower["query"]
        latency_col = (
            cols_lower.get("latency_ms")
            or cols_lower.get("latency_seconds")
            or cols_lower.get("latency")
        )
        time_col = cols_lower.get("created_at") or cols_lower.get("timestamp")

        select_cols = [query_col, latency_col]
        if time_col:
            select_cols.append(time_col)

        df = pd.read_sql_query(
            f"SELECT {', '.join(select_cols)} FROM {table}", conn
        )
        df.rename(
            columns={
                query_col: "query",
                latency_col: "latency",
                time_col or "": "timestamp",
            },
            inplace=True,
        )

        # Normalize latency to seconds when possible.
        if latency_col and latency_col.lower() == "latency_ms":
            df["latency_seconds"] = df["latency"] / 1000.0
        else:
            df["latency_seconds"] = df["latency"].astype(float)

        return df
    finally:
        conn.close()


def _render_search_page() -> None:
    st.header("Search")
    query = st.text_input("Query")
    col1, col2 = st.columns(2)
    with col1:
        top_k = st.number_input("Top K", min_value=1, max_value=100, value=5, step=1)
    with col2:
        alpha = st.slider("Alpha", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

    if st.button("Search") and query.strip():
        retriever = _get_retriever(alpha)
        results = retriever.query(query, top_k=int(top_k))

        st.subheader("Results")
        if not results:
            st.info("No results returned.")
            return

        for r in results:
            title = r.get("title") or r.get("doc_id") or "Untitled"
            score = r.get("score", 0.0)
            breakdown = r.get("score_breakdown", {})
            text = r.get("text", "")
            st.markdown(f"**{title}**  ")
            st.caption(f"Score: {score:.4f}")
            if breakdown:
                st.caption(
                    f"BM25: {breakdown.get('bm25', 0.0):.4f} | "
                    f"Vector: {breakdown.get('vector', 0.0):.4f} | "
                    f"Alpha: {breakdown.get('alpha', 0.0):.2f}"
                )
            if text:
                st.write(text[:800] + ("..." if len(text) > 800 else ""))
            st.divider()


def _render_kpi_page() -> None:
    st.header("KPI")
    st.caption(f"SQLite logs: {LOG_DB_PATH}")

    df = _load_logs(LOG_DB_PATH)
    if df.empty:
        st.info("No log data found. Ensure the SQLite log DB exists with query and latency columns.")
        return

    p95 = df["latency_seconds"].quantile(0.95)
    st.metric("p95 latency (s)", f"{p95:.4f}")

    top_queries = (
        df["query"].value_counts().reset_index().rename(columns={"index": "query", "query": "count"})
    )
    st.subheader("Top Queries")
    st.dataframe(top_queries.head(10), use_container_width=True)


def _render_eval_page() -> None:
    st.header("Evaluation")
    st.caption(f"Experiments: {EXPERIMENTS_CSV}")

    if not EXPERIMENTS_CSV.exists():
        st.info("No experiments.csv found. Run backend/app/eval.py to generate metrics.")
        return

    df = pd.read_csv(EXPERIMENTS_CSV)
    if df.empty or "ndcg@10" not in df.columns:
        st.info("experiments.csv is missing nDCG data.")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp")

    st.subheader("nDCG@10 Trend")
    st.line_chart(df.set_index("timestamp")["ndcg@10"], height=300)


page = st.sidebar.selectbox("Page", ["Search", "KPI", "Evaluation"], index=0)

if page == "Search":
    _render_search_page()
elif page == "KPI":
    _render_kpi_page()
else:
    _render_eval_page()
