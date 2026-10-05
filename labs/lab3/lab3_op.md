I ran these commands -
    python labs/lab3/search.py --baseline
    python labs/lab3/search.py --sweep chunking
    python labs/lab3/search.py --sweep retrieval
    python labs/lab3/search.py --sweep rerank

Op for 1- 
python search.py --baseline          
corpus: 30 docs -> 91 chunks (mean 707 chars)
config                           hit_rate@1     hit_rate@5       recall@5            mrr        ndcg@10 latency_p95_ms
----------------------------------------------------------------------------------------------------------------------
baseline sliding-800 dense           0.7857         0.9286         0.8452         0.8451         0.8053       904.7961

kind             hit_rate@5     n
---------------------------------
aggregation          1.0000     4
multi_hop            1.0000    10
paraphrase           1.0000     5
single_hop           0.8889    18
trap_archived        1.0000     3
unanswerable         0.5000     2

python search.py --sweep chunking
A1: strategies at size=800
config             ndcg@10       recall@5     hit_rate@1            mrr
-----------------------------------------------------------------------
fixed               0.7952         0.8373         0.7381         0.8387
sliding             0.8053         0.8452         0.7857         0.8451
recursive           0.8251         0.8750         0.7619         0.8611
markdown            0.8458         0.8988         0.7619         0.8720
fixed: chunks=83 index_build_ms=159.3
sliding: chunks=91 index_build_ms=117.7
recursive: chunks=98 index_build_ms=106.2
markdown: chunks=164 index_build_ms=10920.9

A2: markdown at sizes 400 / 800 / 1600
config                 ndcg@10       recall@5     hit_rate@1            mrr
---------------------------------------------------------------------------
markdown-400            0.8527         0.9028         0.7857         0.8800
markdown-800            0.8458         0.8988         0.7619         0.8720
markdown-1600           0.8075         0.8750         0.7143         0.8262
markdown-400: chunks=235 index_build_ms=7987.1
markdown-800: chunks=164 index_build_ms=232.6
markdown-1600: chunks=150 index_build_ms=1415.8

A3: markdown heading-path prefix
config                  ndcg@10       recall@5     hit_rate@1            mrr
----------------------------------------------------------------------------
with-prefix              0.8458         0.8988         0.7619         0.8720
without-prefix           0.7915         0.9048         0.6190         0.7837
with-prefix: chunks=164 index_build_ms=225.8
without-prefix: chunks=164 index_build_ms=10028.8

A4: one chunk-boundary failure

python search.py --sweep retrievaliks\AI-in-Practice-Lab\aip-lab1\labs\lab3> 
B1/B2: retrieval comparison on markdown-800
config              mrr        ndcg@10     hit_rate@1       recall@5
--------------------------------------------------------------------
dense            0.8720         0.8458         0.7619         0.8988
bm25             0.6933         0.7009         0.5238         0.7897
hybrid           0.8387         0.8298         0.7143         0.8571

dense
kind                    mrr     n
---------------------------------
aggregation          0.7500     4
multi_hop            1.0000    10
paraphrase           0.9000     5
single_hop           0.8889    18
trap_archived        0.8333     3
unanswerable         0.3125     2
Q44_mrr=0.5000 Q41_mrr=1.0000

bm25
kind                    mrr     n
---------------------------------
aggregation          0.5208     4
multi_hop            0.7200    10
paraphrase           0.5000     5
single_hop           0.8519    18
trap_archived        0.3889     3
unanswerable         0.4167     2
Q44_mrr=1.0000 Q41_mrr=0.0000

hybrid
kind                    mrr     n
---------------------------------
aggregation          0.5833     4
multi_hop            0.8250    10
paraphrase           0.7286     5
single_hop           0.9722    18
trap_archived        0.8333     3
unanswerable         0.5000     2
Q44_mrr=1.0000 Q41_mrr=0.5000

B3: RRF sensitivity
config               mrr        ndcg@10     hit_rate@1       recall@5
---------------------------------------------------------------------
rrf-10            0.8373         0.8382         0.7143         0.8929
rrf-30            0.8387         0.8330         0.7143         0.8571
rrf-60            0.8387         0.8298         0.7143         0.8571
rrf-100           0.8506         0.8381         0.7381         0.8512

B4: unequal fusion weights
config                   mrr        ndcg@10     hit_rate@1       recall@5
-------------------------------------------------------------------------
weights-2:1           0.8248         0.8282         0.6905         0.8770
weights-1:2           0.8508         0.8298         0.7381         0.8829



---

config                 ndcg@5     hit_rate@1       recall@5 latency_p95_ms
--------------------------------------------------------------------------
llm-reranker           0.8460         0.8095         0.8968     37615.0050
llm-reranker: calls=1260 cost_delta=$0.0269

C3: deployment decision
config                ndcg@5    hit_rate@1      p95_ms    $/1k queries
----------------------------------------------------------------------
dense top-5           0.8303        0.7619        12.7          0.0000
cross-encoder         0.8000        0.7381     10530.8          0.0000
llm-reranker          0.8460        0.8095     37615.0          0.0214
interactive search box: use dense or the cross-encoder; avoid the serial LLM reranker because its p95 and per-query cost are too high
overnight batch: use the LLM reranker when its measured quality gain justifies the cost and throughput budget

C4: reranking regressions
question=Q01: dense_mrr=0.5000 cross_encoder_mrr=0.3333