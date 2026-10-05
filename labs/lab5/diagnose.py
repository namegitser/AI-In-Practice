#!/usr/bin/env python3
"""Lab 5 — the failure classifier.

    python labs/lab5/diagnose.py --input reports/lab4.json
    python labs/lab5/diagnose.py --input reports/lab4.json --pareto

Implements the T4 §5 diagnostic tree. Everything that can be decided by code
is decided by code; mode 2 needs your eyes and the script says so.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from labs.lab3.search import load_corpus, load_questions  # noqa: E402

MODES = {
    1: "missing_content",
    2: "chunk_boundary",
    3: "embedding_mismatch",
    4: "ranking",
    5: "reranker",
    6: "generation",
    7: "presentation",
}


def answer_in_corpus(gold_answer: str, corpus: dict[str, str],
                     relevant_docs: list[str]) -> bool:
    """Mode 1 test.

    A literal substring test is too brittle for policy answers, which often
    contain numbers, dates, or reworded phrasing. This version weights the
    important signal words and numeric facts rather than requiring the whole
    sentence to be copied verbatim.
    """
    if not gold_answer or not relevant_docs:
        return False

    text = "\n".join(corpus.get(d, "") for d in relevant_docs if d in corpus)
    if not text:
        return False

    def norm(s: str) -> str:
        s = s.lower()
        s = re.sub(r"[^a-z0-9%]+", " ", s)
        return " ".join(s.split())

    text_norm = norm(text)
    ans_norm = norm(gold_answer)
    if ans_norm and ans_norm in text_norm:
        return True

    # Keep numbers and important words; drop common filler words.
    stop_words = {
        "the", "and", "for", "with", "that", "this", "from", "into",
        "what", "when", "where", "which", "your", "their", "there",
        "have", "does", "will", "must", "not", "been", "more", "less",
        "used", "plan", "policy", "coverage", "covered"
    }

    gold_tokens = [
        t for t in ans_norm.split()
        if t and t not in stop_words and (len(t) > 2 or t.isdigit() or "%" in t)
    ]
    if not gold_tokens:
        return True

    text_tokens = set(norm(text).split())
    matches = sum(1 for t in gold_tokens if t in text_tokens)
    if matches == 0:
        # Numeric values are often the critical fact even if phrasing changes.
        nums = re.findall(r"\d[\d,./%]*", gold_answer)
        if nums:
            return any(n in text for n in nums)
        return False

    return matches / len(gold_tokens) >= 0.35


def classify(row: dict, q: dict, corpus: dict[str, str], *,
             gold_context_fixes_it: bool | None = None,
             in_top_30: bool | None = None,
             dropped_by_reranker: bool | None = None) -> tuple[int, str]:
    """Walk the T4 §5 diagnostic tree. Returns (mode, evidence).

    Ordering matters: the branches are mutually exclusive, and this sequence is
    the diagnostic procedure.
    """
    # Mode 7 first: right answer, wrong citation. Check this before anything
    # else, because a mode-7 failure is not a retrieval failure at all.
    if row.get("correctness", 0) >= 2 and not row.get("citations_valid", True):
        return 7, f"correct answer, invalid citations {row.get('invalid_citations')}"

    # Mode 1: is the answer even in the corpus?
    if not answer_in_corpus(q["gold_answer"], corpus, q["relevant_docs"]):
        return 1, "gold answer content not found in the relevant documents"

    # Mode 6: does gold context fix it? This is the one people invert.
    # Gold context fixing the answer means retrieval was at fault; if it does not,
    # the model was still wrong even with the correct material and this is a
    # generation failure.
    if gold_context_fixes_it is False:
        return 6, "gold context did not fix the answer: generation failure"

    # Mode 4/5: the right doc appears in the candidate pool but still fails in
    # the final ranked context. If the reranker dropped it, it is mode 5.
    if in_top_30 is True:
        if dropped_by_reranker is True:
            return 5, "gold doc was in top 30 but was dropped by the reranker"
        return 4, "gold doc was in top 30 but not in the final k"

    # Mode 3 vs 2: if the doc is not in the top 30, verify whether the gold
    # chunk is retrievable by its own text. If yes, the query embedding is the
    # issue (mode 3). Otherwise it is a chunk-boundary / findability problem.
    if in_top_30 is False:
        gold_text = "\n".join(
            corpus.get(doc_id, "") for doc_id in q.get("relevant_docs", [])
            if doc_id in corpus
        )
        if gold_text:
            text_norm = " ".join(re.findall(r"[A-Za-z0-9%]+", gold_text.lower()))
            retriever_seen = set(
                re.findall(r"[A-Za-z0-9%]+", " ".join(row.get("retrieved", [])).lower())
            )
            gold_tokens = set(re.findall(r"[A-Za-z0-9%]+", gold_text.lower()))
            if gold_tokens and gold_tokens & retriever_seen:
                return 3, "gold chunk text itself can be retrieved; query-side mismatch"
        return 2, "needs_human_check: open the chunks around the gold answer"

    # If the caller did not provide the retrieval-stage flags, keep the original
    # conservative fall-back: this is a judgement call rather than a false auto-
    # classification. This prevents a spurious mode-6 assignment when the data is
    # incomplete.
    if gold_context_fixes_it is None and in_top_30 is None:
        return 2, "needs_human_check: open the chunks around the gold answer"

    # Fallback for partially supplied metadata.
    retrieved = set(row.get("retrieved", []))
    relevant = set(q.get("relevant_docs", []))
    if relevant and retrieved and relevant & retrieved:
        return 4, "gold doc appears in the retrieved list but was not selected for final context"
    return 2, "needs_human_check: open the chunks around the gold answer"


def pareto(tally: Counter) -> str:
    total = sum(tally.values()) or 1
    lines, cum = ["failure mode          n    share   cumulative"], 0
    for mode, n in tally.most_common():
        cum += n
        bar = "█" * round(30 * n / total)
        lines.append(f"{MODES[mode]:<20} {n:>3}   {n/total:>5.1%}   "
                     f"{cum/total:>5.1%}  {bar}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="reports/lab4.json")
    ap.add_argument("--pareto", action="store_true")
    ap.add_argument("--save", default="reports/lab5_diagnosis.json")
    args = ap.parse_args()

    rows = json.loads((ROOT / args.input).read_text(encoding="utf-8"))
    questions = {q["id"]: q for q in load_questions(include_unanswerable=True)}
    corpus = load_corpus()

    failures = [r for r in rows
                if r.get("correctness", 2) < 2 or not r.get("citations_valid", True)]
    print(f"{len(failures)} failures out of {len(rows)}\n")

    out, tally = [], Counter()
    for r in failures:
        q = questions[r["id"]]
        mode, evidence = classify(r, q, corpus)
        tally[mode] += 1
        out.append({"id": r["id"], "kind": q["kind"], "mode": mode,
                    "mode_name": MODES[mode], "evidence": evidence,
                    "question": q["question"], "answer": r["answer"][:300]})
        print(f"  {r['id']:<5} {MODES[mode]:<20} {evidence}")

    print("\n" + pareto(tally))
    print("\nCases marked needs_human_check are Part A2. Open them.")

    p = ROOT / args.save
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
