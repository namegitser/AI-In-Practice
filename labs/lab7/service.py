#!/usr/bin/env python3
"""FastAPI service for the Aurora policy assistant."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import statistics
import sys
import threading
import time
from collections import OrderedDict, defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import run_in_threadpool

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip import cache, tracing  # noqa: E402
from aip.config import settings  # noqa: E402
from aip.cost import Budget, BudgetExceeded, global_budget  # noqa: E402
from aip.embed import embed  # noqa: E402
from aip.guards import ToolGuard, redact_pii  # noqa: E402

RESPONSE_CACHE_SIZE = 512
RESPONSE_CACHE_VERSION = "lab7-answer-v1"
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("AIP_SEMANTIC_CACHE_THRESHOLD", "0.85"))
SEMANTIC_CACHE_ENABLED = os.getenv("AIP_SEMANTIC_CACHE", "0").strip().lower() in {
    "1", "true", "yes", "on",
}
if not 0.0 < SEMANTIC_CACHE_THRESHOLD <= 1.0:
    raise ValueError("AIP_SEMANTIC_CACHE_THRESHOLD must be greater than 0 and at most 1")

app = FastAPI(title="Aurora Policy Assistant", version="1.0")
_PIPELINE: ApplicationPipeline | None = None
_PIPELINE_LOCK = threading.Lock()
_RESPONSE_LOCK = threading.RLock()
_EXACT_RESPONSES: OrderedDict[str, dict[str, Any]] = OrderedDict()
_SEMANTIC_RESPONSES: OrderedDict[str, SemanticEntry] = OrderedDict()
_STARTED = time.time()


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    mode: str = Field(default="rag", pattern="^(rag|tools)$")


class Citation(BaseModel):
    index: int
    doc_id: str
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    citations: list[Citation]
    sources: list[Citation]
    latency_ms: float
    cost_usd: float
    cached: bool
    trace_id: str


@dataclass
class ApplicationPipeline:
    retriever: Any
    allowed_tools: frozenset[str]
    cache_fingerprint: str
    max_tool_calls: int = 6

    def new_tool_guard(self) -> ToolGuard:
        return ToolGuard(
            max_calls=self.max_tool_calls,
            allow=set(self.allowed_tools),
            requires_confirmation={"issue_refund"},
        )


@dataclass
class SemanticEntry:
    vector: np.ndarray
    response: dict[str, Any]
    mode: str
    top_k: int


def pipeline() -> ApplicationPipeline:
    """Build the Labs 3-5 retriever once and retain Lab 6 guard policy."""
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    with _PIPELINE_LOCK:
        if _PIPELINE is None:
            settings.cache_dir.mkdir(parents=True, exist_ok=True)
            from labs.lab4.evaluate import build_retriever
            from labs.lab6.agent import set_layers

            retriever = build_retriever()
            set_layers({1})
            fingerprint = hashlib.sha256()
            for chunk in retriever.chunks:
                fingerprint.update(chunk.doc_id.encode("utf-8"))
                fingerprint.update(b"\0")
                fingerprint.update(chunk.text.encode("utf-8"))
                fingerprint.update(b"\0")
            _PIPELINE = ApplicationPipeline(
                retriever=retriever,
                allowed_tools=frozenset({
                    "search_policy",
                    "get_policy_details",
                    "compute_premium",
                    "issue_refund",
                }),
                cache_fingerprint=fingerprint.hexdigest(),
            )
    return _PIPELINE


@asynccontextmanager
async def _lifespan(_: FastAPI):
    pipeline()
    yield


app.router.lifespan_context = _lifespan


def _normalise_question(question: str) -> str:
    return " ".join(question.casefold().split())


def _cache_key(req: AskRequest) -> str:
    question_hash = hashlib.sha256(
        _normalise_question(req.question).encode("utf-8")
    ).hexdigest()
    return cache.make_key("ask.response", {
        "question_sha256": question_hash,
        "response_version": RESPONSE_CACHE_VERSION,
        "pipeline_fingerprint": (
            _PIPELINE.cache_fingerprint if _PIPELINE is not None else "unbuilt"
        ),
        "main_model": settings.models["MAIN"],
        "embedding_model": settings.models["EMBED"],
        "mode": req.mode,
        "top_k": req.top_k,
    })


def _cached_response(req: AskRequest, vector: np.ndarray | None) -> tuple[dict | None, str | None]:
    key = _cache_key(req)
    with _RESPONSE_LOCK:
        exact = _EXACT_RESPONSES.get(key)
        if exact is not None:
            _EXACT_RESPONSES.move_to_end(key)
            return exact.copy(), "exact"

        exact = cache.get(key)
        if exact is not None:
            _EXACT_RESPONSES[key] = exact
            _EXACT_RESPONSES.move_to_end(key)
            while len(_EXACT_RESPONSES) > RESPONSE_CACHE_SIZE:
                _EXACT_RESPONSES.popitem(last=False)
            return exact.copy(), "exact"

        if vector is None or req.mode != "rag" or not SEMANTIC_CACHE_ENABLED:
            return None, None

        best_key: str | None = None
        best_similarity = -1.0
        for candidate_key, entry in _SEMANTIC_RESPONSES.items():
            if entry.mode != req.mode or entry.top_k != req.top_k:
                continue
            similarity = float(np.dot(vector, entry.vector))
            if similarity > best_similarity:
                best_key, best_similarity = candidate_key, similarity

        if best_key is None or best_similarity < SEMANTIC_CACHE_THRESHOLD:
            return None, None

        _SEMANTIC_RESPONSES.move_to_end(best_key)
        return _SEMANTIC_RESPONSES[best_key].response.copy(), "semantic"


def _store_response(
    req: AskRequest,
    response: dict[str, Any],
    vector: np.ndarray | None,
) -> None:
    key = _cache_key(req)
    cache.put(
        key,
        "ask.response",
        {
            "question_sha256": hashlib.sha256(
                _normalise_question(req.question).encode("utf-8")
            ).hexdigest(),
            "mode": req.mode,
            "top_k": req.top_k,
        },
        response,
    )
    with _RESPONSE_LOCK:
        _EXACT_RESPONSES[key] = response.copy()
        _EXACT_RESPONSES.move_to_end(key)
        while len(_EXACT_RESPONSES) > RESPONSE_CACHE_SIZE:
            _EXACT_RESPONSES.popitem(last=False)

        if (
            SEMANTIC_CACHE_ENABLED
            and req.mode == "rag"
            and vector is not None
            and not response["refused"]
        ):
            _SEMANTIC_RESPONSES[key] = SemanticEntry(
                vector=vector.copy(),
                response=response.copy(),
                mode=req.mode,
                top_k=req.top_k,
            )
            _SEMANTIC_RESPONSES.move_to_end(key)
            while len(_SEMANTIC_RESPONSES) > RESPONSE_CACHE_SIZE:
                _SEMANTIC_RESPONSES.popitem(last=False)


def _provider_unavailable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    status = status or getattr(response, "status_code", None)
    if status in {408, 425, 429, 500, 502, 503, 504}:
        return True

    name = type(exc).__name__.lower()
    return (
        isinstance(exc, (TimeoutError, ConnectionError, OSError))
        or any(marker in name for marker in (
            "timeout",
            "ratelimit",
            "apiconnection",
            "serviceunavailable",
            "internalserver",
            "overloaded",
        ))
    )


def _raise_http_error(exc: Exception, trace_id: str) -> NoReturn:
    if isinstance(exc, HTTPException):
        raise exc
    if isinstance(exc, BudgetExceeded):
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    if isinstance(exc, cache.CacheMiss):
        raise HTTPException(
            status_code=503,
            detail="Offline replay has no cached response for this request",
            headers={"Retry-After": "5", "X-Trace-ID": trace_id},
        ) from exc
    if _provider_unavailable(exc):
        retry_after = getattr(exc, "retry_after", None)
        if not isinstance(retry_after, (int, float)) or retry_after <= 0:
            retry_after = 5
        raise HTTPException(
            status_code=503,
            detail="Model provider unavailable; retry the request later",
            headers={"Retry-After": str(math.ceil(retry_after)), "X-Trace-ID": trace_id},
        ) from exc
    raise HTTPException(
        status_code=500,
        detail="Internal service error",
        headers={"X-Trace-ID": trace_id},
    ) from exc


def _citation_sources(answer: Any) -> tuple[list[Citation], list[Citation]]:
    source_list = [
        Citation(index=index, doc_id=hit.doc_id, excerpt=hit.text.strip())
        for index, hit in enumerate(answer.hits, start=1)
    ]
    cited_indexes = {
        int(value) for value in re.findall(r"\[(\d+)\]", answer.text)
    }
    citations = [source for source in source_list if source.index in cited_indexes]
    return citations, source_list


def _run_request(req: AskRequest, span_name: str = "http.ask") -> AskResponse:
    started = time.perf_counter()
    trace_id = ""
    meter = Budget(limit_usd=settings.budget_usd, label="lab7-request")

    try:
        with tracing.trace(
            span_name,
            question_sha256=hashlib.sha256(req.question.encode("utf-8")).hexdigest(),
            mode=req.mode,
            top_k=req.top_k,
        ) as span:
            trace_id = span["span_id"]
            span["cached"] = False
            span["refused"] = False

            clean_question, pii_counts = redact_pii(req.question)
            span["pii_redacted"] = pii_counts
            request = req.model_copy(update={"question": clean_question})

            settings.cache_dir.mkdir(parents=True, exist_ok=True)
            components = pipeline()
            query_vector: np.ndarray | None = None
            if SEMANTIC_CACHE_ENABLED and req.mode == "rag":
                query_vector = embed(clean_question, input_type="query")

            cached, cache_kind = _cached_response(request, query_vector)
            if cached is not None:
                duration_ms = (time.perf_counter() - started) * 1000
                response = AskResponse(
                    **{
                        **cached,
                        "latency_ms": round(duration_ms, 2),
                        "cost_usd": 0.0,
                        "cached": True,
                        "trace_id": trace_id,
                    }
                )
                span.update(
                    cached=True,
                    cache_kind=cache_kind,
                    cost_usd=0.0,
                    refused=response.refused,
                )
                return response

            try:
                with meter:
                    if req.mode == "rag":
                        from labs.lab4.rag import answer_question, validate_answer

                        with tracing.trace("lab7.rag.answer", top_k=req.top_k):
                            answer = answer_question(
                                clean_question,
                                components.retriever,
                                k=12,
                                final_k=req.top_k,
                            )
                        response_text = answer.text
                        refused = answer.refused or response_text.strip() == (
                            "I don't have enough information in the provided sources to answer that."
                        )
                        with tracing.trace("lab7.rag.validate") as validation_span:
                            validation = validate_answer(response_text, len(answer.hits))
                            if not validation["valid"]:
                                raise RuntimeError(
                                    "RAG pipeline returned an answer that failed citation validation: "
                                    + str(validation["reason"])
                                )
                            validation_span["refused"] = refused
                        citations, sources = _citation_sources(answer)
                    else:
                        from labs.lab6.agent import run_agent

                        result = run_agent(
                            clean_question,
                            guard=components.new_tool_guard(),
                        )
                        stop_reason = result["stopped_because"]
                        if stop_reason.startswith("[lab6-") and "spent $" in stop_reason:
                            raise BudgetExceeded(stop_reason)
                        if stop_reason.startswith("tool_call_budget_exhausted:"):
                            raise BudgetExceeded(
                                "Lab 6 tool-call safety budget was exhausted: " + stop_reason
                            )
                        if stop_reason.startswith("wall_clock_limit:"):
                            raise TimeoutError("Lab 6 agent exceeded its wall-clock limit")
                        response_text = result["answer"]
                        if stop_reason != "completed" or not response_text:
                            raise RuntimeError(
                                f"Lab 6 agent did not complete the request: {stop_reason}"
                            )
                        refused = False
                        citations, sources = [], []
            finally:
                span["cost_usd"] = round(meter.spent_usd, 8)

            duration_ms = (time.perf_counter() - started) * 1000
            response = AskResponse(
                answer=response_text,
                refused=refused,
                citations=citations,
                sources=sources,
                latency_ms=round(duration_ms, 2),
                cost_usd=round(meter.spent_usd, 8),
                cached=False,
                trace_id=trace_id,
            )
            span.update(
                cached=False,
                refused=refused,
                source_count=len(sources),
                cost_usd=response.cost_usd,
            )

            if req.mode == "rag":
                if query_vector is None and SEMANTIC_CACHE_ENABLED:
                    query_vector = embed(clean_question, input_type="query")
                _store_response(request, response.model_dump(), query_vector)
            return response
    except Exception as exc:
        if trace_id:
            tracing.event(
                "http.ask.error",
                trace_id=trace_id,
                error_type=type(exc).__name__,
            )
        _raise_http_error(exc, trace_id)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    return _run_request(req)


@app.post("/ask/stream")
async def ask_stream(req: AskRequest) -> EventSourceResponse:
    """Buffer and validate the complete answer before emitting SSE events."""
    response = await run_in_threadpool(_run_request, req, "http.ask.stream")

    async def events():
        text = response.answer
        for start in range(0, len(text), 80):
            yield {
                "event": "token",
                "data": json.dumps({"text": text[start : start + 80]}),
            }
            await asyncio.sleep(0)
        yield {
            "event": "citations",
            "data": json.dumps([item.model_dump() for item in response.citations]),
        }
        yield {
            "event": "complete",
            "data": json.dumps(response.model_dump()),
        }

    return EventSourceResponse(events())


@app.get("/health")
def health() -> dict[str, Any]:
    components = pipeline()
    chunks = getattr(components.retriever, "chunks", [])
    cache_stats = cache.stats()
    return {
        "status": "ok",
        "uptime_s": round(time.time() - _STARTED, 1),
        "index_size": len(chunks),
        "model": settings.models["MAIN"],
        "cache": cache_stats,
        "response_cache": {
            "exact_entries": cache_stats.get("ask.response", len(_EXACT_RESPONSES)),
            "semantic_entries": len(_SEMANTIC_RESPONSES),
            "semantic_enabled": SEMANTIC_CACHE_ENABLED,
            "semantic_threshold": SEMANTIC_CACHE_THRESHOLD,
        },
    }


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    if percentile == 50:
        return round(statistics.median(values), 2)
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(percentile / 100 * len(ordered)) - 1))
    return round(ordered[index], 2)


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    traces = tracing.read_traces()
    today = date.today()
    requests = [
        row for row in traces
        if row.get("name") in {"http.ask", "http.ask.stream"}
        and date.fromtimestamp(float(row.get("ts", 0))) == today
    ]
    errors_by_type: defaultdict[str, int] = defaultdict(int)
    for row in traces:
        if (
            row.get("name") == "http.ask.error"
            and date.fromtimestamp(float(row.get("ts", 0))) == today
        ):
            errors_by_type[str(row.get("error_type", "unknown"))] += 1

    budget = global_budget()
    latencies = [float(row["duration_ms"]) for row in requests if row.get("duration_ms") is not None]
    cached_latencies = [
        float(row["duration_ms"])
        for row in requests
        if row.get("cached") and row.get("duration_ms") is not None
    ]
    uncached_latencies = [
        float(row["duration_ms"])
        for row in requests
        if not row.get("cached") and row.get("duration_ms") is not None
    ]
    tool_counts: defaultdict[str, int] = defaultdict(int)
    for row in traces:
        if row.get("name") == "tool.call":
            tool_counts[str(row.get("tool", "unknown"))] += 1

    cache_hits = sum(bool(row.get("cached")) for row in requests)
    refusals = sum(bool(row.get("refused")) for row in requests)
    cost_today = sum(float(row.get("cost_usd", 0.0)) for row in requests)
    return {
        **budget.as_dict(),
        "cost_today_usd": round(cost_today, 8),
        "cost_per_query_usd": round(cost_today / len(requests), 8) if requests else 0.0,
        "request_count_today": len(requests),
        "cache_hit_rate": round(cache_hits / len(requests), 4) if requests else 0.0,
        "refusal_rate": round(refusals / len(requests), 4) if requests else 0.0,
        "error_rate_by_type": {
            error_type: round(count / len(requests), 4) if requests else 0.0
            for error_type, count in sorted(errors_by_type.items())
        },
        "tool_call_counts": dict(sorted(tool_counts.items())),
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p95_ms": _percentile(latencies, 95),
        "latency_p99_ms": _percentile(latencies, 99),
        "latency_cached_p50_ms": _percentile(cached_latencies, 50),
        "latency_cached_p95_ms": _percentile(cached_latencies, 95),
        "latency_uncached_p50_ms": _percentile(uncached_latencies, 50),
        "latency_uncached_p95_ms": _percentile(uncached_latencies, 95),
    }
