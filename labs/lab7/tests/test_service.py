from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from labs.lab4 import rag
from labs.lab7 import gate, service

from aip.cost import BudgetExceeded


class FakeHit:
    doc_id = "claims-timelines"
    text = "Submit the claim within 30 days of discharge."


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path):
    components = service.ApplicationPipeline(
        retriever=SimpleNamespace(chunks=[FakeHit()]),
        allowed_tools=frozenset(),
        cache_fingerprint="test-corpus",
    )
    monkeypatch.setattr(service, "pipeline", lambda: components)
    monkeypatch.setattr(service.settings, "trace_dir", tmp_path)
    monkeypatch.setattr(service.settings, "cache_dir", tmp_path)
    monkeypatch.setattr(service.cache, "_DB_PATH", tmp_path / "calls.sqlite3")
    monkeypatch.setattr(service.tracing, "RUN_ID", "test")
    monkeypatch.setattr(service.tracing, "_TRACE_FILE", tmp_path / "test.jsonl")
    with service._RESPONSE_LOCK:
        service._EXACT_RESPONSES.clear()
        service._SEMANTIC_RESPONSES.clear()
    with TestClient(service.app) as test_client:
        yield test_client


def answer_for(question: str, retriever, **kwargs):
    return SimpleNamespace(
        text="Submit the claim within 30 days of discharge [1].",
        refused=False,
        citations_valid=True,
        hits=[FakeHit()],
    )


def test_malformed_request_returns_422(client):
    response = client.post("/ask", json={"question": "no"})
    assert response.status_code == 422


def test_ask_returns_sources_trace_and_exact_cache(client, monkeypatch):
    calls = []

    def fake_answer(question, retriever, **kwargs):
        calls.append(question)
        return answer_for(question, retriever, **kwargs)

    monkeypatch.setattr(rag, "answer_question", fake_answer)
    first = client.post("/ask", json={"question": "How long to file a claim?"})
    assert first.status_code == 200
    assert first.json()["cost_usd"] == 0
    assert first.json()["trace_id"]
    assert first.json()["sources"][0]["doc_id"] == "claims-timelines"
    with service._RESPONSE_LOCK:
        service._EXACT_RESPONSES.clear()

    second = client.post("/ask", json={"question": "How long to file a claim?"})
    assert second.json()["cached"] is True
    assert second.json()["cost_usd"] == 0
    assert second.json()["trace_id"] != first.json()["trace_id"]
    assert len(calls) == 1


def test_budget_exhaustion_returns_429(client, monkeypatch):
    def over_budget(*args, **kwargs):
        raise BudgetExceeded("request budget exceeded")

    monkeypatch.setattr(rag, "answer_question", over_budget)
    response = client.post("/ask", json={"question": "How long to file a claim?"})
    assert response.status_code == 429


def test_provider_timeout_returns_503_and_retry_after(client, monkeypatch):
    def provider_timeout(*args, **kwargs):
        raise TimeoutError("provider timed out")

    monkeypatch.setattr(rag, "answer_question", provider_timeout)
    response = client.post("/ask", json={"question": "How long to file a claim?"})
    assert response.status_code == 503
    assert response.headers["retry-after"] == "5"
    assert "Traceback" not in response.text
    assert client.get("/metrics").json()["error_rate_by_type"] == {"TimeoutError": 1.0}


def test_stream_waits_for_validation_then_emits_answer(client, monkeypatch):
    monkeypatch.setattr(rag, "answer_question", answer_for)
    response = client.post(
        "/ask/stream",
        json={"question": "How long to file a claim?"},
    )

    assert response.status_code == 200
    assert "event: token" in response.text
    assert "event: citations" in response.text
    assert "event: complete" in response.text


def test_semantic_cache_reuses_only_similar_rag_answers(client, monkeypatch):
    monkeypatch.setattr(service, "SEMANTIC_CACHE_ENABLED", True)
    vectors = {
        "How long is the claim filing deadline?": np.array([1.0, 0.0]),
        "What is the deadline to file a claim?": np.array([0.9, np.sqrt(0.19)]),
    }
    monkeypatch.setattr(service, "embed", lambda question, **kwargs: vectors[question])
    calls = []

    def fake_answer(question, retriever, **kwargs):
        calls.append(question)
        return answer_for(question, retriever, **kwargs)

    monkeypatch.setattr(rag, "answer_question", fake_answer)
    first = client.post("/ask", json={"question": "How long is the claim filing deadline?"})
    second = client.post("/ask", json={"question": "What is the deadline to file a claim?"})

    assert first.status_code == second.status_code == 200
    assert second.json()["cached"] is True
    assert len(calls) == 1


def test_gate_exits_nonzero_when_a_metric_breaches(monkeypatch, capsys):
    monkeypatch.setattr(gate, "measure", lambda *, live=False: {"correctness": 0.0})
    monkeypatch.setattr(gate.yaml, "safe_load", lambda _: {"correctness": {"min": 0.75}})
    monkeypatch.setattr(sys, "argv", ["gate.py", "--config", "labs/lab7/thresholds.yml"])

    assert gate.main() == 1
    assert "GATE FAILED" in capsys.readouterr().out


def test_gate_live_flag_explicitly_bypasses_offline_chat_cache(
    monkeypatch, capsys
):
    def measure_live(*, live=False):
        assert live is True
        assert os.environ["AIP_OFFLINE"] == "0"
        assert os.environ["AIP_LLM_CACHE_BYPASS"] == "1"
        return {"correctness": 0.8}

    monkeypatch.setattr(gate, "measure", measure_live)
    monkeypatch.setattr(gate.yaml, "safe_load", lambda _: {"correctness": {"min": 0.75}})
    monkeypatch.setattr(
        sys,
        "argv",
        ["gate.py", "--live", "--config", "labs/lab7/thresholds.yml"],
    )

    assert gate.main() == 0
    assert "LIVE MODE" in capsys.readouterr().out
