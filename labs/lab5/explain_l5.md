# Lab 5: Diagnose the failure modes

## What are we doing in this lab?

This lab is not about guessing random fixes. It is about finding the exact stage where a retrieval-augmented system failed on each wrong answer.

The core idea is:

- A bad answer is usually caused by one specific failure mode.
- The system can fail in the retrieval stage, reranker stage, generation stage, or presentation stage.
- We must diagnose each failure, count how often each mode happens, and then fix the biggest relevant cluster.

The project is built around the T4 §5 diagnostic tree. In plain terms, we ask a sequence of questions:

1. Is the correct answer even present in the relevant corpus documents?
2. If yes, is the answer trapped across a chunk boundary and therefore not fully visible?
3. Was the gold document retrieved at all, or did it get lost in ranking?
4. Did the reranker remove it even though it was available?
5. If the correct gold context is provided, does the model recover the answer? If yes, this is a retrieval problem; if not, it is a generation problem.
6. If the answer is right but the citation is wrong, that is a presentation problem.

This is the heart of Lab 5: classify failures, rank the clusters by impact, and only then choose a fix.

The command we run is:

```bash
python diagnose.py --input reports/lab4.json
```

This script walks through the Lab 4 failures and labels each one according to the diagnostic tree. It tells us:

- how many questions failed,
- which mode each failed question belongs to,
- which cases are ambiguous and need a human check,
- and the final Pareto summary showing which failure modes dominate.

---

## Interpreting the output

The command output is:

```text
21 failures out of 45

  Q03   chunk_boundary       needs_human_check: open the chunks around the gold answer
  Q04   chunk_boundary       needs_human_check: open the chunks around the gold answer
  Q05   chunk_boundary       needs_human_check: open the chunks around the gold answer
  Q10   chunk_boundary       needs_human_check: open the chunks around the gold answer
  Q11   chunk_boundary       needs_human_check: open the chunks around the gold answer
  Q19   chunk_boundary       needs_human_check: open the chunks around the gold answer
  Q20   chunk_boundary       needs_human_check: open the chunks around the gold answer
  Q21   chunk_boundary       needs_human_check: open the chunks around the gold answer
  Q23   chunk_boundary       needs_human_check: open the chunks around the gold answer
  Q24   chunk_boundary       needs_human_check: open the chunks around the gold answer
  Q26   chunk_boundary       needs_human_check: open the chunks around the gold answer
  Q29   chunk_boundary       needs_human_check: open the chunks around the gold answer
  Q32   chunk_boundary       needs_human_check: open the chunks around the gold answer
  Q35   chunk_boundary       needs_human_check: open the chunks around the gold answer
  Q36   missing_content      gold answer content not found in the relevant documents
  Q37   chunk_boundary       needs_human_check: open the chunks around the gold answer
  Q38   missing_content      gold answer content not found in the relevant documents
  Q39   missing_content      gold answer content not found in the relevant documents
  Q40   chunk_boundary       needs_human_check: open the chunks around the gold answer
  Q42   chunk_boundary       needs_human_check: open the chunks around the gold answer
  Q44   chunk_boundary       needs_human_check: open the chunks around the gold answer

failure mode          n    share   cumulative
chunk_boundary        18   85.7%   85.7%  ██████████████████████████
missing_content        3   14.3%   100.0%  ████

Cases marked needs_human_check are Part A2. Open them.
```

### 1. 21 failures out of 45

This means that out of the 45 questions in the Lab 4 report, 21 did not pass. The script only evaluates the failed questions, because the goal is to understand why the system failed, not simply count all questions.

This makes the diagnosis actionable: we are studying the errors, not the successful answers.

---

### 2. The failure labels

The classifier identifies each bad question as one of the main failure modes:

- `chunk_boundary`
- `missing_content`

In this run, the only two modes that appear are these.

#### `chunk_boundary`

This means the gold answer is likely present in the relevant documents, but the retrieval system may be splitting the answer across chunk boundaries. In other words, the answer is logically there, but not fully visible in one chunk or in a retrievable unit.

The script marks these as:

```text
needs_human_check: open the chunks around the gold answer
```

This is important: the script is intentionally conservative. It cannot decide automatically whether the failure is caused by a chunk boundary, because a human must inspect the nearby chunks and judge whether the gold answer is fragmented or cut in the wrong place.

So the questions from Q03, Q04, Q05, Q10, Q11, Q19, Q20, Q21, Q23, Q24, Q26, Q29, Q32, Q35, Q37, Q40, Q42, and Q44 are all flagged for a manual review of the surrounding chunks.

This is exactly what Part A2 of the lab asks for: open the chunks around the gold answer and decide whether the content is split across chunk boundaries.

#### `missing_content`

This means that the relevant documents did not contain the answer content at all. In the output, the failing questions are:

- Q36
- Q38
- Q39

The script says:

```text
gold answer content not found in the relevant documents
```

This is a strong signal that the corpus or the document set is missing the necessary information. It is not a retrieval-ranking problem or a generation problem in the usual sense; the answer is simply absent from the evidence pool.

---

### 3. Pareto summary

The second half of the output is the distribution:

```text
failure mode          n    share   cumulative
chunk_boundary        18   85.7%   85.7%
missing_content        3   14.3%   100.0%
```

This is a Pareto view: the failure modes are ranked by frequency.

- `chunk_boundary`: 18 questions, 85.7%
- `missing_content`: 3 questions, 14.3%

This tells us the dominant issue is chunking, not ranking or generation. In other words:

- most of the broken answers are not due to the model ignoring the right evidence,
- they are due to the evidence being split or fragmented across chunk boundaries,
- and a smaller number are due to the corpus simply lacking the necessary content.

This is the main diagnostic takeaway from the run.

---

## Why this matters in Lab 5

The lab is designed to prevent a common mistake: fixing the system before understanding the root cause.

If we looked only at the final bad answers, we might try a random change such as:

- changing model settings,
- widening the retrieval window,
- modifying the reranker,
- or adjusting prompts.

But the diagnosis says the biggest cluster is `chunk_boundary`, which suggests the system is not seeing complete answer-bearing chunks. That points to evidence fragmentation, not necessarily a bad language model.

The script is therefore making the lab’s main point explicit:

- find the dominant failure mode,
- inspect the evidence,
- then choose the targeted fix,
- and prove the effect with before/after results.

---

## The practical interpretation of this specific run

The real conclusion from this output is:

- Most failures are concentrated in the chunk-boundary category.
- The script has already reported that these need human checks.
- The remaining failures are missing-content issues, which point to insufficient evidence in the corpus.

So the next action in the lab would be:

1. Open the chunk windows around the 18 flagged `chunk_boundary` questions.
2. Verify whether the gold answer is split across adjacent chunks.
3. Decide whether a larger chunk size, greater overlap, or markdown-aware chunking would help.
4. For the 3 `missing_content` cases, record that the corpus itself lacks those facts and treat them as a data-coverage issue rather than a retrieval tuning issue.

This is exactly how Lab 5 is meant to work: diagnose first, then fix the largest evidence-backed cluster.

---

## Final summary

The output tells us:

- 21 of 45 questions failed.
- 18 of those failures are `chunk_boundary` issues.
- 3 are `missing_content` issues.
- The dominant root cause is chunking / evidence fragmentation.
- The `chunk_boundary` cases are intentionally marked for manual inspection as part of Part A2.

That is the big message of this command: the system is not failing evenly across every possible problem; it is failing mostly because the retrieved context is split or incomplete.
