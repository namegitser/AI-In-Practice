#!/usr/bin/env python3
"""Lab 3 — retrieval sweeps.

The scaffolding (corpus loading, metric computation, table printing) is
written for you. The sweeps are yours.

    python labs/lab3/search.py --baseline
    python labs/lab3/search.py --sweep chunking
    python labs/lab3/search.py --sweep retrieval
    python labs/lab3/search.py --sweep rerank
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip import cost  # noqa: E402
from aip.chunking import STRATEGIES, Chunk  # noqa: E402
from aip.evals import retrieval_metrics  # noqa: E402
from aip.retrieval import (  # noqa: E402, F401
    Bm25Retriever,
    ChromaRetriever,
    CrossEncoderReranker,
    DenseRetriever,
    HybridRetriever,
    LLMReranker,
    Retriever,
)

CORPUS_DIR = ROOT / "data/corpus"
GOLDEN = ROOT / "data/eval/rag_golden.jsonl"


# ---------------------------------------------------------------------------
# scaffolding (provided)
# ---------------------------------------------------------------------------
def load_corpus() -> dict[str, str]:
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(CORPUS_DIR.glob("*.md"))}


def load_questions(include_unanswerable: bool = False) -> list[dict]:
    rows = [json.loads(l) for l in GOLDEN.open(encoding="utf-8")]
    if include_unanswerable:
        return rows
    # THREE questions (Q36, Q38, Q39) have no relevant document, so recall and
    # nDCG are undefined for them -- you cannot rank correctly against an empty
    # relevant set. Dropping them leaves n = 42.
    #
    # Do not confuse that with the FIVE questions of kind 'unanswerable'
    # (Q36-Q40): two of those do keep relevant documents, because part of what
    # they ask is supported. All five are measured properly in Lab 4, as
    # refusal precision and recall.
    #
    # Excluding the three is correct -- but say so in your report rather than
    # letting an unexplained n = 42 pass for a stated 45.
    return [r for r in rows if r["relevant_docs"]]


def build_chunks(corpus: dict[str, str], strategy: str = "sliding",
                 size: int = 800, **kw) -> list[Chunk]:
    fn = STRATEGIES[strategy]
    out: list[Chunk] = []
    for doc_id, text in corpus.items():
        try:
            out.extend(fn(text, doc_id, size=size, **kw))
        except TypeError:                       # chunker without that kwarg
            out.extend(fn(text, doc_id, size=size))
    return out


def evaluate(retriever: Retriever, questions: list[dict], k: int = 10,
             reranker=None, final_k: int = 5) -> dict:
    """Run every question, return aggregate metrics + per-kind breakdown."""
    agg: dict[str, list[float]] = defaultdict(list)
    by_kind: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    latencies: list[float] = []
    per_q: dict[str, float] = {}
    per_q_mrr: dict[str, float] = {}

    for q in questions:
        t0 = time.perf_counter()
        hits = retriever.search(q["question"], k=k)
        if reranker is not None:
            hits = reranker.rerank(q["question"], hits, k=final_k)
        latencies.append((time.perf_counter() - t0) * 1000)

        # A document counts as retrieved at rank r if any of its chunks does.
        seen, ranked = set(), []
        for h in hits:
            if h.doc_id not in seen:
                seen.add(h.doc_id)
                ranked.append(h.doc_id)

        m = retrieval_metrics(ranked, q["relevant_docs"], ks=(1, 3, 5, 10))
        per_q[q["id"]] = m["hit_rate@5"]
        per_q_mrr[q["id"]] = m["mrr"]
        for key, val in m.items():
            agg[key].append(val)
            by_kind[q["kind"]][key].append(val)

    out = {k2: statistics.fmean(v) for k2, v in agg.items()}
    out["latency_p50_ms"] = statistics.median(latencies)
    out["latency_p95_ms"] = sorted(latencies)[int(0.95 * (len(latencies) - 1))]
    out["_by_kind"] = {kind: {k2: statistics.fmean(v) for k2, v in d.items()}
                       for kind, d in by_kind.items()}
    out["_per_question"] = per_q            # hit_rate@5 -- saturated, see kind_table
    out["_per_question_mrr"] = per_q_mrr    # use this one for Part B
    out["_kind_n"] = {kind: len(d["mrr"]) for kind, d in by_kind.items()}
    return out


def table(rows: dict[str, dict], cols: tuple[str, ...] =
          ("hit_rate@1", "hit_rate@5", "recall@5", "mrr", "ndcg@10",
           "latency_p95_ms")) -> str:
    name_w = max(len(n) for n in rows) + 2
    head = f"{'config':<{name_w}}" + "".join(f"{c:>15}" for c in cols)
    lines = [head, "-" * len(head)]
    for name, m in rows.items():
        lines.append(f"{name:<{name_w}}" + "".join(f"{m.get(c, 0):>15.4f}" for c in cols))
    return "\n".join(lines)


def kind_table(metrics: dict, col: str = "hit_rate@5") -> str:
    """Break a result down by question kind.

    NOTE the default column. `hit_rate@5` is saturated on this corpus -- every
    retriever scores 0.93-0.98 -- so this table will look flat and tell you
    nothing. Pass col='mrr' or col='ndcg@10' for Part B. The default is left
    saturated on purpose.
    """
    bk, counts = metrics["_by_kind"], metrics.get("_kind_n", {})
    w = max(len(k) for k in bk) + 2
    lines = [f"{'kind':<{w}}{col:>12}{'n':>6}", "-" * (w + 18)]
    for kind, m in sorted(bk.items()):
        lines.append(f"{kind:<{w}}{m.get(col, 0):>12.4f}{counts.get(kind, 0):>6}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# sweeps (yours)
# ---------------------------------------------------------------------------
def sweep_baseline() -> None:
    corpus, questions = load_corpus(), load_questions()
    chunks = build_chunks(corpus, "sliding", 800, overlap=150)
    print(f"corpus: {len(corpus)} docs -> {len(chunks)} chunks "
          f"(mean {statistics.fmean(len(c) for c in chunks):.0f} chars)")
    r = DenseRetriever(chunks)
    m = evaluate(r, questions)
    print(table({"baseline sliding-800 dense": m}))
    print()
    print(kind_table(m))
    print("\nWrite these numbers down before you change anything.")


def sweep_chunking() -> None:
    """TODO A1-A3.

    A1: all four strategies at size=800.
    A2: the winner at sizes 400 / 800 / 1600. Plot or tabulate the curve.
    A3: markdown WITH and WITHOUT the '[heading > path]' prefix.
        (Strip it with a list comprehension over the chunks -- do not modify
         aip/chunking.py; other labs depend on it.)

    Report chunk count and index build time alongside quality. A configuration
    that is 1 point better and takes 4x as long to build is a real trade-off.
    """
    corpus, questions = load_corpus(), load_questions()

    def run(strategy: str, size: int) -> tuple[list[Chunk], dict]:
        kwargs = {"overlap": 150} if strategy == "sliding" else {}
        chunks = build_chunks(corpus, strategy, size, **kwargs)
        started = time.perf_counter()
        retriever = DenseRetriever(chunks, show_progress=False)
        build_ms = (time.perf_counter() - started) * 1000
        metrics = evaluate(retriever, questions)
        metrics["chunk_count"] = len(chunks)
        metrics["index_build_ms"] = build_ms
        return chunks, metrics

    print("A1: strategies at size=800")
    strategy_results: dict[str, dict] = {}
    strategy_chunks: dict[str, list[Chunk]] = {}
    for strategy in STRATEGIES:
        chunks, metrics = run(strategy, 800)
        strategy_chunks[strategy] = chunks
        strategy_results[strategy] = metrics
    print(table(strategy_results,
                ("ndcg@10", "recall@5", "hit_rate@1", "mrr")))
    for strategy, metrics in strategy_results.items():
        print(f"{strategy}: chunks={metrics['chunk_count']} "
              f"index_build_ms={metrics['index_build_ms']:.1f}")

    winner = max(strategy_results, key=lambda name: strategy_results[name]["ndcg@10"])
    print(f"\nA2: {winner} at sizes 400 / 800 / 1600")
    size_results: dict[str, dict] = {}
    for size in (400, 800, 1600):
        _, metrics = run(winner, size)
        size_results[f"{winner}-{size}"] = metrics
    print(table(size_results,
                ("ndcg@10", "recall@5", "hit_rate@1", "mrr")))
    for name, metrics in size_results.items():
        print(f"{name}: chunks={metrics['chunk_count']} "
              f"index_build_ms={metrics['index_build_ms']:.1f}")

    print("\nA3: markdown heading-path prefix")
    markdown_chunks_with_prefix, with_prefix = run("markdown", 800)
    without_prefix_chunks = [
        Chunk(chunk.text.partition("\n")[2], chunk.doc_id, chunk.chunk_id, chunk.meta)
        if chunk.text.startswith("[") and "\n" in chunk.text else chunk
        for chunk in markdown_chunks_with_prefix
    ]
    started = time.perf_counter()
    without_prefix_retriever = DenseRetriever(without_prefix_chunks,
                                               show_progress=False)
    without_prefix_build_ms = (time.perf_counter() - started) * 1000
    without_prefix = evaluate(without_prefix_retriever, questions)
    without_prefix["chunk_count"] = len(without_prefix_chunks)
    without_prefix["index_build_ms"] = without_prefix_build_ms
    print(table({"with-prefix": with_prefix, "without-prefix": without_prefix},
                ("ndcg@10", "recall@5", "hit_rate@1", "mrr")))
    print(f"with-prefix: chunks={with_prefix['chunk_count']} "
          f"index_build_ms={with_prefix['index_build_ms']:.1f}")
    print(f"without-prefix: chunks={without_prefix['chunk_count']} "
          f"index_build_ms={without_prefix['index_build_ms']:.1f}")

    print("\nA4: one chunk-boundary failure")
    winner_chunks, winner_metrics = run(winner, 800)
    winner_retriever = DenseRetriever(winner_chunks, show_progress=False)
    for question in questions:
        if winner_metrics["_per_question_mrr"][question["id"]] == 0:
            expected = [chunk for chunk in winner_chunks
                        if chunk.doc_id in question["relevant_docs"]]
            retrieved = winner_retriever.search(question["question"], k=5)
            print(f"question={question['id']}: {question['question']}")
            print("gold chunks:")
            for chunk in expected[:2]:
                print(f"  {chunk.doc_id}: {chunk.text[:240]!r}")
            print("retrieved chunks:")
            for hit in retrieved:
                print(f"  {hit.doc_id}: {hit.text[:240]!r}")
            break


def sweep_retrieval() -> None:
    """TODO B1-B4.

    B1: dense / bm25 / hybrid on your best chunking.
    B2: print kind_table(m, col='mrr') for each, and pull out Q44 and Q41
        individually from metrics['_per_question_mrr'].

        USE MRR, NOT hit_rate@5. Every retriever here scores 0.93-0.98 on
        hit_rate@5, so it is saturated and shows you nothing -- which is why
        kind_table() and metrics['_per_question'] both default to it. That
        default is the trap, and noticing it is part of the lab.

    B3: RRF k in {10, 30, 60, 100} -- HybridRetriever(..., rrf_k=k).
    B4: unequal fusion weights -- HybridRetriever(..., weights=[2.0, 1.0]).
    """
    corpus, questions = load_corpus(), load_questions()
    chunks = build_chunks(corpus, "markdown", 800)
    dense = DenseRetriever(chunks, show_progress=False)
    bm25 = Bm25Retriever(chunks)

    retrievers = {
        "dense": dense,
        "bm25": bm25,
        "hybrid": HybridRetriever([dense, bm25]),
    }
    results = {name: evaluate(retriever, questions)
               for name, retriever in retrievers.items()}

    print("B1/B2: retrieval comparison on markdown-800")
    print(table(results, ("mrr", "ndcg@10", "hit_rate@1", "recall@5")))
    for name, metrics in results.items():
        print(f"\n{name}")
        print(kind_table(metrics, col="mrr"))
        print(f"Q44_mrr={metrics['_per_question_mrr'].get('Q44', 0.0):.4f} "
              f"Q41_mrr={metrics['_per_question_mrr'].get('Q41', 0.0):.4f}")

    print("\nB3: RRF sensitivity")
    rrf_results = {}
    for rrf_k in (10, 30, 60, 100):
        fused = HybridRetriever([dense, bm25], rrf_k=rrf_k)
        rrf_results[f"rrf-{rrf_k}"] = evaluate(fused, questions)
    print(table(rrf_results, ("mrr", "ndcg@10", "hit_rate@1", "recall@5")))

    print("\nB4: unequal fusion weights")
    weighted_results = {}
    for weights in ([2.0, 1.0], [1.0, 2.0]):
        label = f"weights-{weights[0]:.0f}:{weights[1]:.0f}"
        fused = HybridRetriever([dense, bm25], weights=weights)
        weighted_results[label] = evaluate(fused, questions)
    print(table(weighted_results, ("mrr", "ndcg@10", "hit_rate@1", "recall@5")))


def sweep_rerank() -> None:
    """TODO C1-C4.

    Retrieve k=30, rerank to 5: evaluate(r, questions, k=30, reranker=rr,
    final_k=5).

    C1: CrossEncoderReranker. First run downloads ~90 MB.
    C2: LLMReranker -- report cost as well as latency.
    C3: the decision table, and TWO different deployment answers
        (interactive search box vs overnight batch). They should differ.
    C4: find a query reranking made worse, using
        metrics['_per_question_mrr'] before and after.
    """
    corpus, questions = load_corpus(), load_questions()
    chunks = build_chunks(corpus, "markdown", 800)
    retriever = DenseRetriever(chunks, show_progress=False)

    def run(reranker=None) -> dict:
        return evaluate(retriever, questions, k=30, reranker=reranker, final_k=5)

    baseline = evaluate(retriever, questions, k=5)
    print("C1/C2: retrieve 30, rerank 5")
    print(table({"dense@30": baseline},
                ("ndcg@5", "hit_rate@1", "recall@5", "latency_p95_ms")))

    cross = CrossEncoderReranker()
    cross_metrics = run(cross)
    print(table({"cross-encoder": cross_metrics},
                ("ndcg@5", "hit_rate@1", "recall@5", "latency_p95_ms")))
    print(f"cross-encoder deltas: ndcg@5={cross_metrics['ndcg@5'] - baseline['ndcg@5']:+.4f} "
          f"hit_rate@1={cross_metrics['hit_rate@1'] - baseline['hit_rate@1']:+.4f} "
          f"recall@5={cross_metrics['recall@5'] - baseline['recall@5']:+.4f} "
          f"added_p95_ms={cross_metrics['latency_p95_ms'] - baseline['latency_p95_ms']:+.1f}")

    before_spend = cost.global_budget().spent_usd
    before_calls = cost.global_budget().calls
    llm_metrics = run(LLMReranker())
    budget = cost.global_budget()
    print(table({"llm-reranker": llm_metrics},
                ("ndcg@5", "hit_rate@1", "recall@5", "latency_p95_ms")))
    print(f"llm-reranker: calls={budget.calls - before_calls} "
          f"cost_delta=${budget.spent_usd - before_spend:.4f}")

    llm_cost_per_1000 = ((budget.spent_usd - before_spend) / max(
        budget.calls - before_calls, 1)) * 1000
    print("\nC3: deployment decision")
    print(f"{'config':<18}{'ndcg@5':>10}{'hit_rate@1':>14}"
          f"{'p95_ms':>12}{'$/1k queries':>16}")
    print("-" * 70)
    print(f"{'dense top-5':<18}{baseline['ndcg@5']:>10.4f}"
          f"{baseline['hit_rate@1']:>14.4f}{baseline['latency_p95_ms']:>12.1f}"
          f"{'0.0000':>16}")
    print(f"{'cross-encoder':<18}{cross_metrics['ndcg@5']:>10.4f}"
          f"{cross_metrics['hit_rate@1']:>14.4f}{cross_metrics['latency_p95_ms']:>12.1f}"
          f"{'0.0000':>16}")
    print(f"{'llm-reranker':<18}{llm_metrics['ndcg@5']:>10.4f}"
          f"{llm_metrics['hit_rate@1']:>14.4f}{llm_metrics['latency_p95_ms']:>12.1f}"
          f"{llm_cost_per_1000:>16.4f}")
    print("interactive search box: use dense or the cross-encoder; avoid the "
          "serial LLM reranker because its p95 and per-query cost are too high")
    print("overnight batch: use the LLM reranker when its measured quality gain "
          "justifies the cost and throughput budget")

    worse = [
        qid for qid in baseline["_per_question_mrr"]
        if cross_metrics["_per_question_mrr"][qid] < baseline["_per_question_mrr"][qid]
    ]
    print("\nC4: reranking regressions")
    if worse:
        qid = worse[0]
        print(f"question={qid}: dense_mrr={baseline['_per_question_mrr'][qid]:.4f} "
              f"cross_encoder_mrr={cross_metrics['_per_question_mrr'][qid]:.4f}")
    else:
        print("no cross-encoder regression found in this run")


def sweep_index() -> None:
    """TODO D1-D3.

    D1/D2: ChromaRetriever vs DenseRetriever -- recall gap and latency.
    D3: pass status metadata into the chunks and filter at query time.

        Set chunk.meta['status'] = 'archived' if 'ARCHIVED' in doc_id else 'current'
        then ChromaRetriever.search(..., where={"status": "current"}).

        Report hit_rate@1 on Q29/Q30/Q31 before and after (hit_rate@1, not
        @5 -- @5 is saturated here and will hide the whole effect).
    """
    corpus, questions = load_corpus(), load_questions()
    chunks = build_chunks(corpus, "markdown", 800)

    def timed_dense(items: list[Chunk]) -> tuple[DenseRetriever, float]:
        started = time.perf_counter()
        result = DenseRetriever(items, show_progress=False)
        return result, (time.perf_counter() - started) * 1000

    dense, dense_build_ms = timed_dense(chunks)
    chroma_path = str(ROOT / ".chroma_lab3")
    started = time.perf_counter()
    chroma = ChromaRetriever(chunks, path=chroma_path,
                             collection="markdown800", reset=True)
    chroma_build_ms = (time.perf_counter() - started) * 1000
    dense_metrics = evaluate(dense, questions)
    chroma_metrics = evaluate(chroma, questions)
    print("D1: DenseRetriever vs ChromaRetriever")
    print(table({"dense": dense_metrics, "chroma": chroma_metrics},
                ("recall@5", "ndcg@10", "latency_p95_ms")))
    print(f"build_ms: dense={dense_build_ms:.1f} chroma={chroma_build_ms:.1f}")

    print("\nD2: index scaling")
    base = chunks
    for target in (160, 4000, 40000):
        items = [
            Chunk(chunk.text, chunk.doc_id, f"{chunk.chunk_id}::x{index}", dict(chunk.meta))
            for index, chunk in enumerate((base * ((target + len(base) - 1) // len(base)))[:target])
        ]
        scaled_dense, dense_ms = timed_dense(items)
        started = time.perf_counter()
        scaled_chroma = ChromaRetriever(items, path=chroma_path,
                                         collection=f"scale_{target}", reset=True)
        chroma_ms = (time.perf_counter() - started) * 1000
        print(f"chunks={target}: dense_build_ms={dense_ms:.1f} "
              f"chroma_build_ms={chroma_ms:.1f} "
              f"dense_search_ms={_search_latency(scaled_dense, questions[0]['question']):.1f} "
              f"chroma_search_ms={_search_latency(scaled_chroma, questions[0]['question']):.1f}")

    print("\nD3: metadata filtering")
    for chunk in chunks:
        chunk.meta["status"] = "archived" if "ARCHIVED" in chunk.doc_id else "current"
    filtered = ChromaRetriever(chunks, path=chroma_path,
                               collection="status_filtered", reset=True)
    target_questions = [q for q in load_questions(include_unanswerable=True)
                        if q["id"] in {"Q29", "Q30", "Q31"}]
    before = {}
    after = {}
    for question in target_questions:
        before_hits = filtered.search(question["question"], k=1)
        after_hits = filtered.search(question["question"], k=1, where={"status": "current"})
        before[question["id"]] = float(bool(before_hits and before_hits[0].doc_id in question["relevant_docs"]))
        after[question["id"]] = float(bool(after_hits and after_hits[0].doc_id in question["relevant_docs"]))
    print(f"Q29-Q31 hit_rate@1 before={before} after={after}")


def _search_latency(retriever: Retriever, query: str) -> float:
    started = time.perf_counter()
    retriever.search(query, k=10)
    return (time.perf_counter() - started) * 1000


SWEEPS = {
    "chunking": sweep_chunking,
    "retrieval": sweep_retrieval,
    "rerank": sweep_rerank,
    "index": sweep_index,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--sweep", choices=list(SWEEPS))
    args = ap.parse_args()
    if args.baseline or not args.sweep:
        sweep_baseline()
    if args.sweep:
        SWEEPS[args.sweep]()


if __name__ == "__main__":
    main()
