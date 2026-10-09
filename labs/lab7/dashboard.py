#!/usr/bin/env python3
"""Streamlit observability dashboard for local AIP traces."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.config import settings  # noqa: E402

st.set_page_config(page_title="Aurora Assistant — Ops", layout="wide")
st.title("Aurora Policy Assistant — operations")

runs = sorted(settings.trace_dir.glob("*.jsonl"), reverse=True)
if not runs:
    st.info(f"No traces yet in {settings.trace_dir}. Run some queries first.")
    st.stop()

run_ids = [path.stem for path in runs]
chosen = st.sidebar.multiselect("Runs", run_ids, default=[run_ids[0]])
rows = [
    json.loads(line)
    for path in runs
    if path.stem in chosen
    for line in path.open(encoding="utf-8")
    if line.strip()
]
if not rows:
    st.info("Select at least one run containing trace records.")
    st.stop()

df = pd.DataFrame(rows)
if "ts" in df:
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
if "duration_ms" not in df:
    df["duration_ms"] = 0.0
df["duration_ms"] = pd.to_numeric(df["duration_ms"], errors="coerce").fillna(0.0)

request_names = {"http.ask", "http.ask.stream"}
requests = df[df["name"].isin(request_names)].copy()
total_cost = (
    pd.to_numeric(requests.get("cost_usd", pd.Series(dtype=float)), errors="coerce")
    .fillna(0.0)
    .sum()
)
request_count = len(requests)
cache_rate = (
    requests.get("cached", pd.Series(False, index=requests.index))
    .fillna(False)
    .astype(bool)
    .mean()
    if request_count
    else 0.0
)
error_count = (
    int((requests.get("status", pd.Series(index=requests.index, dtype=object)) == "error").sum())
    if request_count
    else 0
)

kpi_columns = st.columns(4)
kpi_columns[0].metric("Requests", request_count)
kpi_columns[1].metric("Cost", f"${total_cost:.4f}")
kpi_columns[2].metric("Cost per request", f"${total_cost / request_count:.5f}" if request_count else "—")
kpi_columns[3].metric("Response-cache hit rate", f"{cache_rate:.0%}")
st.metric("Errors", error_count)

st.subheader("Latency by span")
duration_spans = df[df["duration_ms"] > 0]
if duration_spans.empty:
    st.info("No duration spans are present in the selected runs.")
else:
    stage_table = (
        duration_spans.groupby("name")["duration_ms"]
        .agg(
            n="count",
            p50_ms="median",
            p95_ms=lambda values: values.quantile(0.95),
            total_ms="sum",
        )
        .sort_values("total_ms", ascending=False)
    )
    st.dataframe(stage_table.round(2))

    if "ts" in duration_spans:
        stage_time = duration_spans[["ts", "name", "duration_ms"]].copy()
        stage_time["window"] = stage_time["ts"].dt.floor("5min")
        stage_time = (
            stage_time.groupby(["window", "name"])["duration_ms"]
            .quantile(0.95)
            .unstack("name")
            .sort_index()
        )
        if not stage_time.empty:
            st.caption("Five-minute p95 latency by span name")
            st.line_chart(stage_time)

st.subheader("Cumulative request cost")
if request_count and "ts" in requests:
    request_cost = requests[["ts"]].copy()
    request_cost["cost_usd"] = pd.to_numeric(
        requests.get("cost_usd", pd.Series(0.0, index=requests.index)),
        errors="coerce",
    ).fillna(0.0)
    cumulative = request_cost.sort_values("ts")
    cumulative["cumulative_usd"] = cumulative["cost_usd"].cumsum()
    st.line_chart(cumulative.set_index("ts")["cumulative_usd"])
else:
    st.info("No request-level cost spans are present in the selected runs.")

st.subheader("Errors by type")
error_events = df[df["name"] == "http.ask.error"]
if not error_events.empty:
    error_summary = error_events.groupby("error_type").size().rename("count").to_frame()
    error_summary["rate"] = error_summary["count"] / max(request_count, 1)
    st.dataframe(error_summary)
else:
    st.success("No request errors in the selected runs.")

st.subheader("Refusal-rate alert")
if request_count >= 40 and "refused" in requests:
    ordered_requests = requests.sort_values("ts")
    previous = ordered_requests.iloc[-40:-20]
    recent = ordered_requests.iloc[-20:]
    baseline_rate = previous["refused"].fillna(False).astype(bool).mean()
    current_rate = recent["refused"].fillna(False).astype(bool).mean()
    doubled = current_rate > 0 and (
        baseline_rate == 0 or current_rate >= 2 * baseline_rate
    )
    alert_columns = st.columns(2)
    alert_columns[0].metric("Previous 20 requests", f"{baseline_rate:.0%}")
    alert_columns[1].metric("Most recent 20 requests", f"{current_rate:.0%}")
    if doubled:
        st.error(
            "Refusal rate doubled. Pause rollout, verify corpus/index freshness, "
            "then rerun the golden-set gate before restoring traffic."
        )
    else:
        st.success("Refusal rate has not doubled across the two latest 20-request windows.")
else:
    st.info(
        "The refusal alert needs at least 40 request spans: it compares the latest "
        "20 requests with the preceding 20."
    )
