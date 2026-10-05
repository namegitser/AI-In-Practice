# Retrieval Metric Definitions

## `hit_rate@k` — “Did I find it?”

Measures whether at least one correct document appears among the top `k` results. It is a pass/fail metric.

For example, `hit_rate@5` is 1 if a correct document appears anywhere in the first five results; otherwise, it is 0.

## `recall@k` — “How complete are the results?”

Measures how many of the relevant documents appear among the top `k` results.

If a question has four valid answer documents and the top five results contain two of them, recall is 50%.

## `mrr` — “How quickly did I find the first answer?”

Mean Reciprocal Rank measures the position of the first relevant document:

- Rank 1: `1.0`
- Rank 2: `0.5`
- Rank 3: `0.33`

A high MRR means users usually find a correct answer near the top of the results.

## `ndcg@10` — “Are the results in the right order?”

Normalized Discounted Cumulative Gain evaluates the ranking quality of the top ten results. Documents receive more credit when they appear near the top, while relevant documents at lower ranks receive less credit.

## `latency_p95_ms` — “How fast is the system?”

Measures the time taken to process search queries at the 95th percentile. A p95 latency of 904 ms means that 95% of queries completed in 904 ms or less.

## `latency_p95_ms` — “How fast is the system under slow conditions?”

Measures query latency at the 95th percentile. A p95 latency of 904 ms means that 95% of queries completed in 904 ms or less, isolating worst-case tail latency.

## `index_build_ms` — “How long does setup take?”

Measures the wall-clock time (in milliseconds) required to chunk the document corpus, generate embedding vectors via API calls, and build the search index.

## `chunk_count` — “How granular is the search index?”

Measures the total number of text segments produced from the document corpus. It directly tracks index size, storage footprint, and chunk granularity.