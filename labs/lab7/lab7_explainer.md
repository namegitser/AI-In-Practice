# Lab 7 Explainer: From Lab Foundations to a Measured Service

## Why Lab 7 exists

The earlier labs built and tested the pieces of an AI application: structured
extraction, model and prompt selection, semantic retrieval, grounded answers,
failure diagnosis, and guarded tool use. Lab 7 brings those pieces into a
service and asks a different question: can we operate it, measure it, and
detect a regression before shipping it?

This lab wraps the prior RAG pipeline in a FastAPI service, provides a small
user interface and observability dashboard, and evaluates the behavior against
a fixed 45-question golden set. The thresholds live in
[`thresholds.yml`](thresholds.yml); the evaluation runner is
[`gate.py`](gate.py).

## How the previous labs lead into Lab 7

| Lab | What it established | How it informs Lab 7 |
|---|---|---|
| **Lab 1 — Reliable extraction** | Parse messy user messages into structured, validated records; measure schema validity, accuracy, cost, and latency. | Establishes the expectation that model output needs validation and that quality, cost, and speed are all part of a result. |
| **Lab 2 — Prompt and model experiments** | Compare prompt strategies, model tiers, and routing using an experiment harness rather than intuition. | Provides the experimental mindset behind a repeatable evaluation and explicit cost/quality trade-offs. |
| **Lab 3 — Semantic search** | Build a retriever over a corpus and labeled queries; compare retrieval configurations and ranking quality. | Supplies the retrieval foundation used by the RAG answer pipeline and the retrieval hit-rate metric. |
| **Lab 4 — Grounded RAG** | Generate answers from retrieved passages, cite sources, validate citations, and refuse when the corpus does not support an answer. | Supplies the answer-generation pipeline and the golden-set quality and refusal checks measured in this lab. |
| **Lab 5 — RAG diagnosis and repair** | Classify answer failures by stage, predict a targeted fix, then check every metric for regressions. | Reminds us to investigate the cause of a failure rather than merely loosen a threshold. The Lab 7 report identifies concrete remaining errors, including false refusals and partially correct answers. |
| **Lab 6 — Tools and red-teaming** | Add a bounded tool loop, argument validation, safety guards, and an attack suite. | Supplies the tool-mode safety policies reused by the service; reminds us that operational controls and adversarial behavior matter alongside answer quality. |
| **Lab 7 — Ship and observe** | Connect the components to an API, collect traces and metrics, and enforce a regression gate. | Turns the lab pipeline into something that can be exercised through an API and monitored over time. |

Lab 7 is not a replacement for the earlier work: it is the integration and
operational layer around it.

## The two evaluation paths

Both paths evaluate the **same full golden set of 45 questions** using the
current retrieval and RAG pipeline. They differ in where model responses come
from and what the result can tell us.

### 1. Cached offline replay

The default path sets `AIP_OFFLINE=1`. When the pipeline requests a model
response, the shared AIP client looks for the exact request in
`.aip_cache/calls.sqlite3` and replays the saved response. If the requested
response is not cached, the client raises a cache-miss error rather than
contacting a provider. Embedding responses may also be served by the cache.

Run it from the repository root:

```powershell
python labs\lab7\gate.py --config labs\lab7\thresholds.yml --json reports\lab7_gate.json
```

Or from `labs\lab7`:

```powershell
python .\gate.py --config .\thresholds.yml --json ..\..\reports\lab7_gate.json
```

**What this path is good for:** repeatable local checks and CI without a
provider key, network dependency, or new model-call spend. Given the same
inputs and cache, it gives the same model outputs.

**What it cannot establish:** current live-provider response time, current
provider behavior, or the cost of uncached production calls. Offline latency
mostly measures local work and cache access; it is not a provider SLO result.
The cache must exist and match the requests. It is ignored by Git by default,
so a clean CI checkout cannot replay it unless the cache is reviewed and
deliberately made available to CI.

### 2. Live-provider evaluation

The live path sets offline mode off and bypasses the shared **chat response**
cache for model generation and LLM judging. It makes real provider calls using
the selected `AIP_PROFILE` and configured credentials. Cached embeddings may
still be reused, so this run is live for chat/generation and judging, but is
not necessarily a completely uncached run for every component.

Run it from the repository root:

```powershell
$env:AIP_BUDGET_USD = "2"
python labs\lab7\gate.py --live --config labs\lab7\thresholds.yml --json reports\lab7_live_gate.json
```

Or from `labs\lab7`:

```powershell
$env:AIP_BUDGET_USD = "2"
python .\gate.py --live --config .\thresholds.yml --json ..\..\reports\lab7_live_gate.json
```

**What this path is good for:** observing current provider behavior, answer
quality, and latency when calls actually go to the model service.

**Trade-offs:** it needs a configured provider key and network access; it can
take several minutes, its outputs may vary, provider limits may cause retries,
and it may incur cost. The budget is a safety cap, not a promise that the
evaluation costs that much. The gate prints the configured cap; measured
token-price estimates are reported separately below.

### At a glance

| Property | Cached offline replay | Live-provider evaluation |
|---|---|---|
| Chat/generation/judge calls | Replay exact saved responses | Make provider calls; response-cache bypass is enabled |
| Embeddings | Cache may supply them | Cache may still supply them |
| Network and provider key | Not needed for cached requests | Required |
| Repeatability | High for an unchanged cache and code | Lower; model output and provider conditions can vary |
| New model-call cost | None for replayed responses | Possible; bounded by the configured process budget |
| Latency meaning | Primarily local/cache-path timing | Includes live answer-generation provider latency |
| Best use | Regression check and deterministic CI | Provider quality/latency observation and pre-release validation |

Keep the two reports separate. A fast offline replay does not prove the live
latency target, and a live result can change between runs even when the code
does not.

## Results against `thresholds.yml`

The tables below compare the saved offline run
([`lab7_gate.json`](../../reports/lab7_gate.json)) and live run
([`lab7_live_gate.json`](../../reports/lab7_live_gate.json)) to the same
thresholds.

| Metric | Meaning | Threshold | Offline | Live |
|---|---|---:|---:|---:|
| `correctness` | Mean correctness judge score, normalized to 0–1, on the 40 answerable questions. | ≥ 0.75 | 0.7750 — Pass | 0.7500 — Pass, exactly at floor |
| `faithfulness` | Fraction of all 45 answers judged supported by the retrieved context. | ≥ 0.90 | 0.9556 — Pass | 0.9778 — Pass |
| `citation_validity` | Fraction of answers whose citations pass the Lab 4 citation check. | ≥ 0.98 | 1.0000 — Pass | 1.0000 — Pass |
| `refusal_recall` | Fraction of the 5 unanswerable questions correctly refused. | ≥ 0.80 | 1.0000 (5/5) — Pass | 1.0000 (5/5) — Pass |
| `refusal_precision` | Of all refusals, fraction that were on unanswerable questions. | ≥ 0.70 | 0.7143 (5/7) — Pass, narrow margin | 0.7143 (5/7) — Pass, narrow margin |
| `hit_rate_at_5` | Fraction of the 42 questions with labeled relevant documents for which at least one relevant document appeared in the top 5. | ≥ 0.85 | 0.9762 (41/42) — Pass | 0.9762 (41/42) — Pass |
| `cost_per_query_usd` | Estimated MAIN-model answer-generation cost per question, based on model price and recorded prompt/completion tokens. | ≤ $0.010 | $0.00182185 — Pass | $0.00193997 — Pass |
| `p95_latency_ms` | 95th percentile of elapsed time spent generating each of the 45 answers in the gate. | ≤ 6,000 ms | 10.19 ms — Pass offline only | 13,646.75 ms — **Fail** |

### What each metric says—and does not say

#### Correctness

This is an LLM-judged score, with each answer’s 0–2 judge score divided by two
and averaged over the **40 answerable** questions. Both runs satisfy the floor;
the live result is exactly at the threshold, so it has no headroom against
run-to-run variation. Sixteen of the 40 answers were not fully correct in the
offline baseline; a score below 1.0 can include partial answers as well as
incorrect ones. The metric is a small-sample estimate, not proof of production
accuracy.

#### Faithfulness

This records whether the answer’s factual claims were supported by the
retrieved context. Both runs exceed 0.90. The live judge scored 44/45 as
faithful; the offline judge scored 43/45. Faithfulness is distinct from
correctness: a response can be supported by the passage but incomplete, or
well cited yet fail to answer the question.

#### Citation validity

All 45 answers passed in both runs. This checks citation structure and
references; it does **not** prove that a cited source entails every claim, nor
that the answer is complete. That is why citation validity must be read beside
faithfulness and correctness.

#### Refusal recall and precision

Recall is 5/5 in both runs: all five questions labeled unanswerable were
refused. Precision is 5/7: there were seven total refusals, of which two were
false refusals on answerable questions (Q23 and Q44). The precision threshold
was set to 0.70 to keep this measured baseline green; it is only
0.0143 above the floor. One more false refusal, with the same five correct
refusals, changes precision to 5/8 = 0.625 and fails. With only five
unanswerable items, both refusal metrics are coarse and sensitive to one
example.

#### Retrieval hit rate@5

At least one labeled relevant document appeared in the top five for 41 of the
42 questions with a relevant-document set. This passes the 0.85 floor
comfortably. It does not measure ranking quality within the top five or prove
the generator used the right passage. One retrieval miss still matters: if the
answer is not in the context, generation cannot reliably cite it.

#### Cost per query

The measured answer-generation estimates are below the configured $0.010
maximum in both runs. The live estimate is about $0.00194 per question. This
metric **does not include all evaluation cost**: LLM-judge calls are excluded,
and cached embedding calls do not provide complete token counts. During the
live gate, 123 chat-provider calls were traced (47 answer-generation and 76
judge calls), with an estimated configured-price total of $0.458278 for those
chat calls. This estimate is not a provider invoice; verify actual billing in
the provider console.

#### p95 latency — the failed live threshold

The offline result, 10.19 ms, is fast because model chat responses are replayed
from cache. It passes the numeric threshold but is **not evidence** that a live
request meets the 6,000 ms SLO.

The live answer p95 was 13,646.75 ms, approximately **2.27×** the 6,000 ms
limit. The gate measures the elapsed time around each `answer_question` call;
it does not include the subsequent LLM-judge calls in that per-answer latency.
The live run therefore exposes a real latency problem in the measured answer
path, but this single sequential golden-set run does not isolate whether the
cause was model generation time, retries, provider variability, prompt size,
or another stage. Do not treat the offline and live latency numbers as
comparable service-load benchmarks.

## Summary

- The golden-set gate runs the full 45-question suite in both paths.
- All quality metrics passed in both the cached offline and live runs. Live
  correctness is at its minimum threshold, and refusal precision is only
  slightly above its floor.
- The live answer-generation cost metric passed. The full evaluation made
  additional judge calls, so the per-query metric is not the total evaluation
  cost.
- **The live p95 latency threshold failed** at 13.65 seconds against a 6-second
  maximum. The offline latency pass does not override this result.
- The evaluation is useful evidence on this 45-question dataset, but it is not
  a guarantee of behavior on new questions, real traffic, or a larger corpus.

## Best practices for operating these two paths

1. Use offline replay for routine regression checks; use live mode intentionally
   when you need current provider evidence.
2. Keep offline and live JSON reports separate and label each with the run date,
   model/profile, cache state, and budget. Do not replace the offline baseline
   with live output.
3. Keep a finite `AIP_BUDGET_USD` for live runs. Check estimates and actual
   provider billing; remember that judge calls add evaluation cost.
4. Preserve the golden set and report raw denominators with ratios. A metric
   such as refusal precision on seven refusals can move substantially because
   of one example.
5. Treat a threshold as a release check, not a target to relax after a failure.
   Investigate the underlying stage and add a regression example before
   changing a floor or ceiling.
6. Keep semantic answer caching disabled until near-duplicate answer
   substitutions have been evaluated on labeled examples. Similar questions
   can require different policy distinctions.
7. Do not expose the prototype as a production decision system without human
   review, authentication, traffic controls, and a continuously maintained
   authoritative corpus.

## Next steps to address the failed live latency metric

1. **Find the long-tail questions.** Record per-question elapsed time and
   correlate each slow answer with its `llm.call` spans, retry events, model,
   token counts, and cache status. Inspect metadata only; avoid logging user
   secrets or full sensitive prompts.
2. **Separate latency by stage.** Measure query embedding, retrieval, answer
   generation, validation, and any provider retries independently. This gate’s
   p95 is the answer call, not the entire service request and not the judge
   time.
3. **Reduce avoidable generation work.** Review prompt/context length and
   `final_k`; remove redundant context while checking correctness and
   faithfulness. Test an appropriate faster model or routing policy on the same
   golden set. Do not trade away grounding or citation checks without measuring
   the regression.
4. **Measure repeated runs.** Run the live suite more than once at a controlled
   time and distinguish provider variance from a consistent slow tail. Record
   median, p95, retries, and cost for every run.
5. **Set an explicit latency policy.** Keep offline CI deterministic, and run a
   separate scheduled or pre-release live performance check if provider SLO
   evidence is required. Do not use a cached p95 as a substitute for live
   latency.
6. **Test streaming separately.** The current stream endpoint buffers and
   validates the complete answer before sending SSE chunks. Measure both
   time-to-first-byte/token and total validated-answer time; improving perceived
   responsiveness must not send unvalidated claims.
7. **Improve refusal precision.** Diagnose Q23 and Q44 through retrieval and
   generation traces, add them to focused regression tests, then reassess the
   0.70 threshold only after new evidence.
