#!/usr/bin/env python3
"""Lab 7 — the regression gate. Exits non-zero when a threshold is breached.

    python labs/lab7/gate.py --config labs/lab7/thresholds.yml
    use --live for live call default is cache 
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def measure(*, live: bool = False) -> dict[str, float]:
    """Evaluate the Lab 4 RAG pipeline against the golden set and score it.

    Offline replay is the default. Live model calls require the explicit
    --live option in main().
    """
    from aip.config import resolve_model, settings
    from aip.cost import Budget, is_priced, price_of
    from aip.evals import retrieval_metrics
    from aip.retrieval import format_context
    from labs.lab3.search import load_questions
    from labs.lab4.evaluate import (
        build_retriever,
        judge_correctness,
        judge_faithfulness,
    )
    from labs.lab4.rag import REFUSAL, answer_question

    if live and settings.offline:
        raise RuntimeError("Live mode cannot run while AIP_OFFLINE=1")
    if not live and not settings.offline:
        raise RuntimeError("Use --live for provider calls; offline replay requires AIP_OFFLINE=1")

    questions = load_questions(include_unanswerable=True)
    if not questions:
        raise RuntimeError("The golden set is empty")

    retriever = build_retriever()
    model = resolve_model("MAIN")
    if not is_priced(model):
        raise RuntimeError(
            f"Model {model!r} has no published price; refusing to report a false zero cost"
        )

    correctness: list[float] = []
    faithfulness: list[float] = []
    citation_validity: list[float] = []
    refusal_recall_hits = refusal_recall_total = 0
    refusal_precision_hits = refusal_total = 0
    retrieval_hits: list[float] = []
    estimated_costs: list[float] = []
    latencies: list[float] = []

    for question in questions:
        started = time.perf_counter()
        answer_meter = Budget(limit_usd=settings.budget_usd, label=f"gate-{question['id']}")
        with answer_meter:
            answer = answer_question(question["question"], retriever, final_k=5)
        latencies.append((time.perf_counter() - started) * 1000)

        is_unanswerable = (
            not question["relevant_docs"] or question["kind"] == "unanswerable"
        )
        is_refusal = answer.refused or answer.text.strip() == REFUSAL
        if is_unanswerable:
            refusal_recall_total += 1
            refusal_recall_hits += int(is_refusal)
        if is_refusal:
            refusal_total += 1
            refusal_precision_hits += int(is_unanswerable)

        citation_validity.append(float(answer.citations_valid))
        faithfulness.append(float(judge_faithfulness(
            answer.text,
            format_context(answer.hits),
        )))
        if not is_unanswerable:
            correctness.append(
                judge_correctness(
                    question["question"],
                    answer.text,
                    question["gold_answer"],
                ) / 2
            )

        if question["relevant_docs"]:
            ranked_docs = list(dict.fromkeys(hit.doc_id for hit in answer.hits))
            retrieval_hits.append(
                retrieval_metrics(ranked_docs, question["relevant_docs"], ks=(5,))[
                    "hit_rate@5"
                ]
            )

        # The offline model cache retains provider token counts. Re-price those
        # tokens so CI can still catch a prompt/model cost regression.
        estimated_costs.append(
            price_of(model, answer_meter.prompt_tokens, answer_meter.completion_tokens)
        )

    if not correctness or not retrieval_hits or refusal_recall_total == 0:
        raise RuntimeError("The golden set is missing required answerable, retrieval, or refusal cases")

    return {
        "correctness": sum(correctness) / len(correctness),
        "faithfulness": sum(faithfulness) / len(faithfulness),
        "citation_validity": sum(citation_validity) / len(citation_validity),
        "refusal_recall": refusal_recall_hits / refusal_recall_total,
        "refusal_precision": (
            refusal_precision_hits / refusal_total if refusal_total else 1.0
        ),
        "hit_rate_at_5": sum(retrieval_hits) / len(retrieval_hits),
        "cost_per_query_usd": sum(estimated_costs) / len(estimated_costs),
        "p95_latency_ms": sorted(latencies)[
            min(len(latencies) - 1, math.ceil(0.95 * len(latencies)) - 1)
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="labs/lab7/thresholds.yml")
    ap.add_argument("--json", help="optional path to save the measured metrics")
    ap.add_argument(
        "--live",
        action="store_true",
        help="call live model providers instead of replaying cached chat responses",
    )
    args = ap.parse_args()

    if args.live:
        os.environ["AIP_OFFLINE"] = "0"
        os.environ["AIP_LLM_CACHE_BYPASS"] = "1"
        print(
            "LIVE MODE: chat/generation/judge calls use configured providers; "
            "cached embeddings may be reused. "
            f"Process budget: ${os.getenv('AIP_BUDGET_USD', '2.0')}."
        )
    else:
        os.environ["AIP_OFFLINE"] = "1"
        os.environ.pop("AIP_LLM_CACHE_BYPASS", None)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    thresholds = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(thresholds, dict) or not thresholds:
        raise ValueError(f"Threshold config is empty or invalid: {config_path}")
    metrics = measure(live=args.live)
    if args.json:
        report_path = Path(args.json)
        if not report_path.is_absolute():
            report_path = ROOT / report_path
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    failures = []
    width = max(len(k) for k in thresholds)
    print(f"{'metric':<{width}}  {'value':>10}  {'gate':>14}  status")
    print("-" * (width + 40))
    for name, rule in thresholds.items():
        value = metrics.get(name)
        if value is None:
            failures.append(f"{name}: not measured")
            print(f"{name:<{width}}  {'—':>10}  {'':>14}  MISSING")
            continue
        ok, gate = True, ""
        if "min" in rule:
            gate, ok = f">= {rule['min']}", value >= rule["min"]
        if "max" in rule and ok:
            gate, ok = f"<= {rule['max']}", value <= rule["max"]
        if not ok:
            failures.append(f"{name}: {value} violates {gate}")
        print(f"{name:<{width}}  {value:>10.4f}  {gate:>14}  {'ok' if ok else 'FAIL'}")

    if failures:
        print("\nGATE FAILED:")
        for f in failures:
            print("  " + f)
        return 1
    print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
