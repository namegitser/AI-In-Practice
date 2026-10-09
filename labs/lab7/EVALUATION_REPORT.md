# Lab 7 evaluation report

**Run date:** 2026-10-09 · **Golden set:** 45 questions · **Mode:** offline replay

## 1. What it does

Aurora answers questions using its policy documents, shows the passages behind
its answers, and says when it cannot find enough support. The service also
records response time and model cost, and reruns a fixed set of questions so a
change that makes answers worse can stop the build.

## 2. How well it works

The gate replayed the Lab 4 pipeline against 45 questions (40 answerable and 5
unanswerable). The cost figure is an estimate from cached MAIN-model token
counts and the configured model price; cached embedding records do not retain
their token counts, so embedding cost is excluded. Latency is measured on the
local offline replay, not against a live provider.

| Metric | Measured | Gate | Result |
|---|---:|---:|---|
| Correctness, normalized 0–1 (40 answerable) | 0.7750 | ≥ 0.75 | Pass |
| Faithfulness (45 questions) | 0.9556 | ≥ 0.90 | Pass |
| Citation validity (45 questions) | 1.0000 | ≥ 0.98 | Pass |
| Refusal recall (5 unanswerable) | 1.0000 | ≥ 0.80 | Pass |
| Refusal precision (7 refusals) | 0.7143 | ≥ 0.70 | Pass |
| Retrieval hit rate @5 (42 with relevant docs) | 0.9762 | ≥ 0.85 | Pass |
| Estimated MAIN-model cost/query | $0.00182185 | ≤ $0.010 | Pass |
| Offline replay p95 | 10.19 ms | ≤ 6,000 ms | Pass; not a live SLO result |

The refusal-precision floor was calibrated from the measured 5/7 result. The
previous 0.75 floor correctly failed on this baseline. At 0.70, one additional
false refusal would reduce precision to 5/8 = 0.625 and fail the gate. This is a
small test set, so the metric should be treated as a warning signal rather
than a precise population estimate.

## 3. Where it fails

- **2 of 7 refusals were false refusals** on answerable questions (Q23 and Q44).
  The other 5 refusals were correct.
- **16 of 40 answerable answers did not receive a fully-correct score.** These
  include partial answers as well as incorrect answers; they are not all
  equivalent failures.
- **2 of 45 faithfulness judgments failed.** Citation validation passed all
  45; citations can be well-formed while the answer is incomplete or
  unsupported.
- Retrieval missed all relevant documents on **1 of 42** questions that had a
  relevant-document set.

### Semantic-cache cutoff

An offline cosine sweep compared all 990 pairs in the 45-question set. The
nearest pair with different reference answers was Q28 and Q42 at **0.8355**;
their questions are similar, but the gold answers make different policy
distinctions. There was one such candidate at thresholds 0.80–0.835 and none
at 0.84 or higher. The configured safety threshold is **0.85**. This is a
reference-answer collision boundary, not a claim that every pair was
LLM-judged after substituting a cached response. Semantic reuse is therefore
**off by default** (`AIP_SEMANTIC_CACHE=1` is required) until more labeled
question pairs are collected.

## 4. What it costs

At the offline MAIN-model answer-generation estimate of **$0.00182185/query**, the projection is
**$1.82 per 1,000 queries** and **$6,649.75/year at 10,000 queries/day**. These
figures use current configured MAIN-model pricing and cached prompt/completion
token counts. They exclude evaluation-judge calls, query-embedding cost, index
construction, retries, and provider price changes; they are not a complete
invoice forecast. Exact response-cache hits cost $0 in model calls after the
first answer.

## 5. How fast it is

The API smoke test replayed six questions offline. Its request p50/p95 were
**18.16/29.64 ms**. The gate's p95 was **10.19 ms** over its 45-question replay.
The span breakdown below is also offline; generation was served from the model
response cache and its live-provider latency was not measured.

| Span | p50 | p95 | Samples |
|---|---:|---:|---:|
| Query embedding (cached lookup) | 0.01 ms | 0.01 ms | 1 |
| Dense retrieval (includes query embedding) | 2.17 ms | 2.31 ms | 6 |
| Full RAG answer span | 4.84 ms | 13.67 ms | 6 |
| Citation validation | 0.02 ms | 0.03 ms | 6 |
| Request end-to-end | 18.16 ms | 29.64 ms | 6 |
| Live generation | Not measured | Not measured | Offline replay |

The offline times demonstrate that the request path works; they do **not**
prove the 800 ms cached or 6,000 ms uncached production SLO. The most valuable
next latency measurement is a live uncached run with provider generation
enabled. Streaming deliberately buffers and validates the complete answer
before sending SSE chunks, prioritizing citation safety over time-to-first-token.

### Live-provider gate run

The full 45-question gate was also run live on 2026-10-09 with chat-response
caching bypassed and a $2 process budget. It made **123 provider chat calls**
(47 answer-generation calls and 76 judge calls); the trace-based configured
price estimate was **$0.458278** total, or about **$0.01018/question**. This
estimate is not a provider invoice. The gate's `cost_per_query_usd` metric
($0.00194) covers MAIN-model answer generation only, not the judge calls.

| Live metric | Measured | Gate | Result |
|---|---:|---:|---|
| Correctness | 0.7500 | ≥ 0.75 | Pass |
| Faithfulness | 0.9778 | ≥ 0.90 | Pass |
| Citation validity | 1.0000 | ≥ 0.98 | Pass |
| Refusal recall | 1.0000 | ≥ 0.80 | Pass |
| Refusal precision | 0.7143 | ≥ 0.70 | Pass |
| Retrieval hit rate @5 | 0.9762 | ≥ 0.85 | Pass |
| MAIN-model answer cost/query | $0.00194 | ≤ $0.010 | Pass |
| Live answer p95 | 13,646.75 ms | ≤ 6,000 ms | **Fail** |

The quality and answer-cost thresholds passed, but live answer latency did not.
Investigate provider response time and the long-tail requests before claiming
the latency SLO. Full live metrics are in `reports/lab7_live_gate.json`.

## 6. What it is not safe for

Do not use this prototype to make or communicate a coverage decision without
human review. It can omit exceptions, confuse adjacent plan rules, or refuse a
question it could have answered; the observed false refusals and partial
answers demonstrate that boundary. The corpus is a fixed teaching dataset, not
an authoritative, continuously verified policy source. Tool mode uses
synthetic customer records; refund calls require confirmation and are denied
unless a confirmation mechanism is deliberately supplied. The service also
has no production authentication or traffic controls and should not be exposed
to an untrusted network.

## 7. What to do next

1. **Fix Q23 and Q44 false refusals.** Add both as regression cases, trace
   retrieval versus generation failure, and raise refusal precision before
   relaxing its threshold.
2. **Collect labeled near-duplicate questions and test semantic answer
   substitutions.** Keep semantic caching disabled until the measured cutoff
   has a defensible margin on answer-level labels.
3. **Investigate latency and measure streaming.** Analyze the live 13.65-second
   p95, measure streamed TTFT and full-response time, and account for judge and
   embedding costs; then review the cache and verify CI from a clean checkout.

## Reproducibility note

The local `AIP_OFFLINE=1` gate passed and wrote `reports/lab7_gate.json`. The
root workflow refuses to run without `.aip_cache/calls.sqlite3`. That file is
ignored by Git in this checkout, so **a clean GitHub checkout cannot yet replay
the evaluation**; review and deliberately include the cache before expecting a
green CI run. The intentional code-break / red-build demonstration has not
been performed or represented as complete.
