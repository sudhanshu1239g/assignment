"""Streamlit dashboard for search and KPIs."""

from __future__ import annotations

import sys
import os
import sqlite3
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
import requests
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.search.hybrid import HybridRetriever

LOG_DB_PATH = Path("data/metrics/logs.db")
API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")


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

        result_col = cols_lower.get("result_count")
        select_cols = [query_col, latency_col]
        if time_col:
            select_cols.append(time_col)
        if result_col:
            select_cols.append(result_col)

        df = pd.read_sql_query(
            f"SELECT {', '.join(select_cols)} FROM {table}", conn
        )
        df.rename(
            columns={
                query_col: "query",
                latency_col: "latency",
                time_col or "": "timestamp",
                result_col or "": "result_count",
            },
            inplace=True,
        )

        # Normalize latency to seconds when possible.
        if latency_col and latency_col.lower() == "latency_ms":
            df["latency_seconds"] = df["latency"] / 1000.0
        else:
            df["latency_seconds"] = df["latency"].astype(float)

        if "result_count" not in df.columns:
            df["result_count"] = pd.NA
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
        results = None
        # Prefer API so searches get logged for KPI page.
        try:
            resp = requests.post(
                f"{API_BASE_URL}/search",
                json={"query": query, "top_k": int(top_k), "alpha": float(alpha)},
                timeout=10,
            )
            if resp.ok:
                payload = resp.json()
                results = payload.get("results", [])
            else:
                st.warning("API search failed; falling back to local retriever.")
        except Exception:
            st.warning("API not reachable; falling back to local retriever.")

        if results is None:
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

    p50 = df["latency_seconds"].quantile(0.50)
    p95 = df["latency_seconds"].quantile(0.95)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("p50 latency (s)", f"{p50:.4f}")
    with col2:
        st.metric("p95 latency (s)", f"{p95:.4f}")

    # Request volume over time (per minute).
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        if not df.empty:
            volume = df.set_index("timestamp").resample("1min").size()
            st.subheader("Request Volume (per minute)")
            st.line_chart(volume, height=220)

    top_queries = df["query"].value_counts().reset_index(name="count").rename(columns={"index": "query"})
    st.subheader("Top Queries")
    st.dataframe(top_queries.head(10), use_container_width=True)

    # Zero-result queries.
    if "result_count" in df.columns and df["result_count"].notna().any():
        zero_df = df[df["result_count"] == 0]
        if not zero_df.empty:
            zero_queries = zero_df["query"].value_counts().reset_index(name="count").rename(
                columns={"index": "query"}
            )
            st.subheader("Zero-Result Queries")
            st.dataframe(zero_queries.head(10), use_container_width=True)


page = st.sidebar.selectbox("Page", ["Search", "KPI"], index=0)

if page == "Search":
    _render_search_page()
else:
    _render_kpi_page()
