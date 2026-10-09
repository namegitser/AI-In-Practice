#!/usr/bin/env python3
"""Lab 7 — Streamlit front end.

    streamlit run labs/lab7/ui.py

Requires the service to be running:
    uvicorn labs.lab7.service:app --port 8000

The one non-negotiable UI requirement: **citations must be expandable to show
the source text.** Grounding the user cannot check is decoration.
"""
from __future__ import annotations

import json

import requests
import streamlit as st

API = st.sidebar.text_input("Service URL", "http://localhost:8000")

st.title("Aurora Policy Assistant")
st.caption("Answers come only from Aurora's policy documents. "
           "Every claim is cited. When the documents do not cover a question, "
           "the assistant says so instead of guessing.")

q = st.text_input("Ask a question",
                  placeholder="How long do I have to file a reimbursement claim?")
mode = st.selectbox("Answer mode", ["rag", "tools"])
stream_answer = st.checkbox(
    "Stream after validation",
    help="The service buffers and validates the complete answer before sending SSE events.",
)

def _stream_ask(question: str, answer_mode: str) -> dict:
    response = requests.post(
        f"{API}/ask/stream",
        json={"question": question, "mode": answer_mode},
        timeout=(10, 120),
        stream=True,
    )
    response.raise_for_status()
    text_slot = st.empty()
    chunks: list[str] = []
    event_name = ""
    complete: dict | None = None
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("event:"):
            event_name = line.partition(":")[2].strip()
        elif line.startswith("data:"):
            payload = json.loads(line.partition(":")[2].strip())
            if event_name == "token":
                chunks.append(payload["text"])
                text_slot.markdown("".join(chunks))
            elif event_name == "complete":
                complete = payload
    if complete is None:
        raise ValueError("The stream ended without a complete response event")
    return complete


if st.button("Ask", type="primary") and q:
    try:
        with st.spinner(
            "validating the complete answer before streaming"
            if stream_answer
            else "thinking"
        ):
            if stream_answer:
                data = _stream_ask(q, mode)
            else:
                r = requests.post(
                    f"{API}/ask",
                    json={"question": q, "mode": mode},
                    timeout=120,
                )
                r.raise_for_status()
                data = r.json()
    except requests.HTTPError as exc:
        st.error(f"{exc.response.status_code}: {exc.response.text[:300]}")
        st.stop()
    except requests.RequestException as exc:
        st.error(f"service unreachable: {exc}")
        st.stop()
    except (ValueError, KeyError) as exc:
        st.error(f"invalid service response: {exc}")
        st.stop()

    if data.get("refused"):
        st.warning(data["answer"])
    elif not stream_answer:
        st.markdown(data["answer"])

    citations = data.get("citations", [])
    for citation in citations:
        label = f"[{citation['index']}] {citation['doc_id']}"
        with st.expander(label):
            st.text(citation["excerpt"])

    cols = st.columns(4)
    cols[0].metric("latency", f"{data.get('latency_ms', 0):.0f} ms")
    cols[1].metric("cost", f"${data.get('cost_usd', 0):.5f}")
    cols[2].metric("cached", "yes" if data.get("cached") else "no")
    cols[3].metric("sources", len(data.get("sources", citations)))
    st.caption(f"trace: `{data.get('trace_id', '')}`")
