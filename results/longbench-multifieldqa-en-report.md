# LongBench MultiFieldQA-en Evaluation

Dataset: [official THUDM/LongBench MultiFieldQA-en](https://huggingface.co/datasets/THUDM/LongBench), locally stored at `data/longbench/multifieldqa_en.jsonl`.

## Data and protocol

- Questions evaluated: **150** over **112** distinct article contexts.
- Context length: mean **4539 words** per question context; passage corpus totals **1,665** chunks across distinct contexts.
- Question categories: all 150 rows are English, but the released record schema has no per-question category label, so no category-wise scores are claimed; questions are drawn from articles in diverse fields.
- Passage chunking: project sentence chunker, project Chunker, sentence-based, 400 words with 40-word overlap unless overridden.
- Retrieval methods: BM25, exact dense cosine, reciprocal rank fusion, HNSW→PPR, and HNSW→PPR→cross-encoder.
- Reranking: cross-encoder/ms-marco-MiniLM-L-6-v2; up to 40 candidate passages scored.
- Context packing budget: 4096 tokens using `openai/gpt-oss-20b`.
- Observed retrieval sizes: mean HNSW/lexical entry union **13.5** passages; cross-encoder scored **13.6** per question on average (maximum 23); mean packed tokens were **3427** for BM25 and **3658** for the project ranking.
- Graph coverage: Temporary T-only passage graph with semantic HNSW edges per article; no extracted entity/relation/summary nodes or Leiden communities.
- Split: the official test set is evaluated as published. No hyperparameter tuning was performed on it.
- Correct answer: LongBench normalized token F1 against the best supplied answer reference. Exact match is also reported as a secondary metric.
- Retrieval evidence limitation: this split supplies answers but no gold evidence spans. Recall/MRR/nDCG are therefore not official gold-evidence metrics here.
- Answer-anchor subset: **47/150** questions have at least one normalized reference answer string appearing contiguously in the source; retrieval figures below are explicitly a weak diagnostic on that subset.

## Retrieval results (answer-anchor diagnostic subset)

| Method | n | Hit@1 | Hit@5 | Hit@10 | MRR | nDCG@10 | Packed answer-anchor recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 47 | 0.830 | 0.957 | 1.000 | 0.882 | 0.868 | 1.000 |
| dense_exact | 47 | 0.574 | 0.851 | 0.957 | 0.702 | 0.707 | 0.894 |
| hybrid_rrf | 47 | 0.723 | 0.936 | 1.000 | 0.807 | 0.819 | 0.979 |
| hnsw_ppr | 47 | 0.234 | 0.638 | 0.809 | 0.437 | 0.474 | 0.681 |
| hnsw_ppr_crossencoder | 47 | 0.830 | 0.957 | 1.000 | 0.888 | 0.889 | 0.979 |

The answer-anchor subset is selected by whether answer text is recoverable verbatim after normalization. It is biased toward extractive answers, can miss paraphrased evidence, and can count an answer mention that is not the actual reasoning evidence. Do not present these figures as dataset gold Recall@k/MRR/nDCG.

## Generation results

Generation is scored separately on the frozen packed contexts. BM25 is the controlled retrieval baseline; the project condition uses HNSW→PPR→cross-encoder. Both use the same answer-only prompt and provider fallback configuration.

| Retrieved context | Completed | LongBench QA F1 | Normalized EM |
|---|---:|---:|---:|
| bm25 | 150/150 | 41.70 | 10.67 |
| hnsw_ppr_crossencoder | 150/150 | 37.93 | 7.33 |

Paired F1 difference (graph minus BM25) over questions answered by both: **-3.76 percentage points** [-7.22, -0.44] across 150 pairs; among pairs answered by the same provider model: **-0.01 percentage points** [-4.91, 5.17] across 44 pairs. Brackets show deterministic 95% percentile bootstrap intervals.

Provider/model assignment was subject to live rate limits and the configured key/model fallback sequence. This is not a fixed-model controlled generation comparison; use the same-model paired subset as a secondary diagnostic and interpret both aggregate scores cautiously.

| Provider model | Retrieval context | n | LongBench QA F1 |
|---|---|---:|---:|
| openai/gpt-oss-120b | bm25 | 56 | 40.94 |
| openai/gpt-oss-120b | hnsw_ppr_crossencoder | 49 | 39.70 |
| openai/gpt-oss-20b | bm25 | 50 | 38.42 |
| openai/gpt-oss-20b | hnsw_ppr_crossencoder | 54 | 30.03 |
| qwen/qwen3.8-27b | bm25 | 44 | 46.38 |
| qwen/qwen3.8-27b | hnsw_ppr_crossencoder | 47 | 45.16 |

Failed provider calls are excluded from the F1 denominator and counted in the completed column. Per-question predictions and model/key-slot metadata are in the JSON result; secret values are not saved.

## Interpretation and limits

- The test set is small (150 questions) and contains only single-document questions. Results are diagnostic, not a broad significance claim.
- The official set has no train/dev/test partition for this task; avoid tuning on it and then reporting the same score as held-out generalization.
- The temporary benchmark graph contains only source passage nodes and semantic HNSW links. It does not test the project's LLM-extracted entities/relations, Leiden communities, or summary hierarchy.
- The fixed 4,096-token context limit makes this a retrieval/compression comparison; it is not LongBench's full-context model score.
- LongBench defines MultiFieldQA-en as single-document QA and uses normalized QA F1. See the [official task statistics](https://github.com/THUDM/LongBench/blob/main/LongBench/task.md) and [official evaluator](https://github.com/THUDM/LongBench/blob/main/LongBench/eval.py).

## Artifacts

- `data/longbench/multifieldqa_en.jsonl` — downloaded official test data.
- `scripts/longbench/run_multifieldqa_eval.py` — repeatable retrieval and generation runner.
- `results/longbench-multifieldqa-en-retrieval.json` — frozen rankings, packed prompts, metrics, and run metadata.
- `results/longbench-multifieldqa-en-generation.json` — per-answer output, scores, provider metadata, and resumable checkpoints.

Re-run with `python scripts/longbench/run_multifieldqa_eval.py --stage all`. Use `--stage retrieval` to rerun retrieval only or `--stage generation` to reuse the frozen retrieval artifact.
