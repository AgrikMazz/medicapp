# Full-Book GraphRAG Run and Evaluation

**Run date:** 2026-10-08  
**Source:** [`data/medical-CareForInsane/book.txt`](../data/medical-CareForInsane/book.txt)  
**Detailed per-question result:** [`results/book-text-results.json`](book-text-results.json)

The original 24-question results below are a historical pilot. The later controlled retrieval run at the end of this report uses a corrected, passage-disjoint development/held-out question set and should be used for the current retrieval comparison.

**Updated retrieval tuning and held-out report:** [`book-text-retrieval-tuning-report.md`](book-text-retrieval-tuning-report.md).

## What was run

The entire text file was indexed, not a page sample. The source is 181,866 bytes, 179,927 Unicode characters, and 30,690 words. Chunk offsets start at character 0 and end at 179,927; the union of persisted passage ranges covers **all non-whitespace source characters** (0 uncovered). Five passage strings have harmless sentence-boundary whitespace normalization; no non-whitespace text falls outside the passage ranges.

The build produced **95 T passages**, **95 S summaries**, **1,261 N entities**, **510 R relations**, and **18 H/O community pairs**: **1,997 graph nodes and 10,325 edges**. The HNSW index embeds 208 T/S/H nodes. All 95 chunks have entity lists and summaries; all relation jobs completed (93 passages contain at least one relation, 2 have an empty relation list). All 18 communities have a title and high-level summary. A nodes are not implemented in this build.

## Evaluation design

- **Questions:** 24 LLM-generated questions with exact supporting quotes and source-character offsets: 18 single-section and 6 cross-section. They are saved in [`data/book-graph/eval_questions.generated.jsonl`](../data/book-graph/eval_questions.generated.jsonl).
- **Split:** no train/test or held-out split; this is a one-book pilot. The questions were generated from the same book and were not human-reviewed. One cross-section item incorrectly asks how brain function ?relates? to ebook-license terms, a relationship the source does not establish. Treat cross-section results as especially noisy.
- **Correct retrieval:** a passage is relevant when its source-character range overlaps an annotated evidence quote. Recall@k checks whether the first *k* ranked T passage ranges cover each full quote; MRR uses the first passage overlapping any quote; nDCG@10 uses the number of annotated spans overlapped as passage relevance. Scores are macro-averaged by question.
- **Pipeline:** HNSW top 10 + lexical candidate retrieval + shallow PPR, followed by graph reranking/pruning. There were no exact N/O label matches for these 24 questions.
- **Controlled baseline:** the custom BM25-style ranker scores only the same 95 T passage nodes, using the same questions and source-span labels.
- **Separate stages:** retrieval-only scores are in [`results/book-text-retrieval-final.json`](book-text-retrieval-final.json); answers were then generated in a second pass. The two passes have identical retrieved T node IDs and identical serialized context strings for all 24 questions.
- **Generation evaluator:** `NodeRAGEvaluator` computes normalized lexical Exact Match, token precision/recall/F1, ROUGE-L, METEOR, and BLEU. This is not a human or LLM-judge faithfulness evaluation.

## Retrieval results

| System/stage | Evidence Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| BM25-only over T passages | 0.646 | 0.854 | 0.917 | 0.795 | 0.806 |
| Graph retrieval after PPR (raw candidate subgraph) | 0.396 | 0.729 | 0.896 | 0.644 | 0.662 |
| Final graph context after pruning | 0.292 | 0.583 | 0.792 | 0.467 | 0.517 |

The graph pipeline?s raw candidate recall@10 is **0.896**, versus **0.917** for BM25. BM25 also ranks evidence better overall (MRR 0.795 vs. 0.644; nDCG@10 0.806 vs. 0.662). Graph pruning lowers recall@10 to **0.792**. On this pilot, HNSW+PPR+pruning does **not** beat the simple passage baseline overall.

The 12,000-character context cap introduces another loss: exact annotated evidence quotes have **0.583 serialized-context recall**, and every annotated quote is present for only **0.500** of questions. By category, single-section questions have raw graph Recall@10 1.000, final-context span recall 0.889, and serialized-span recall 0.667; cross-section questions score 0.583, 0.500, and 0.333, respectively. None of the six cross-section questions had *all* annotated spans in the serialized prompt.

## Generation results

| Metric | Score |
|---|---:|
| Answers generated | 24/24 |
| Mean answer token F1 | 0.274 |
| Exact Match | 0.000 |
| ROUGE-L | 0.256 |
| METEOR | 0.328 |
| BLEU-1 | 0.182 |
| BLEU-4 | 0.082 |

One question needed a second generation attempt after rate limiting; all 24 ultimately succeeded. Because the references are synthetic and evidence serialization is incomplete, these answer scores are diagnostic only. They mix generation quality with retrieval/context omissions and are not a reliable standalone estimate of answer correctness. The final prompt now asks text-only answers to cite source chunk IDs instead of inventing page numbers; 13 of 24 responses used the requested source-ID citation form.

## What the run says

1. **The full book is stored.** All non-whitespace source characters are represented by persisted T-passage source ranges, and all 95 passages completed entity and summary enrichment.
2. **BM25 is the stronger baseline here.** It wins overall Recall@10, MRR, and nDCG@10. The graph pipeline retrieves more inter-context signals than a first-hit metric alone suggests, but its candidate recall remains lower and pruning loses relevant spans.
3. **Cross-section retrieval is the failure point.** The synthetic cross-section slice has only 6 items, and the graph?s final context covers half of the annotated spans on average; the 12k prompt serialization never includes both spans for any of those six items.
4. **Generation cannot compensate for evidence that did not reach its prompt.** Answer F1 is low, but this evaluation cannot separate model reasoning from missing serialized evidence.

## Most useful next steps

- Replace or human-review the 24 synthetic QA records. Build a held-out set across the entire book with answer aliases, question type, and evidence spans; verify each cross-section question really requires both passages and has a source-supported answer.
- Keep BM25 as the current baseline. Tune HNSW candidate count, PPR expansion, reranking, and pruning against Recall@k/MRR/nDCG; do not claim a graph improvement until it wins against this same-corpus baseline.
- Improve query-aware context packing. The graph may contain evidence that its final 12k-character prompt drops; select and pack the highest-ranked evidence passages before graph summaries/edges, then score exact span recall on the actual serialized prompt.
- Add a semantic answer/faithfulness judge with human spot checks and evaluate citation correctness separately. The current lexical answer metrics do not measure factual support.

## Reproduction and artifacts

The full enrichment run was `python scripts/legacy/run_textbook.py` in the project virtual environment. To repeat evaluation without paying to rebuild the graph:

```powershell
.\.venv\Scripts\python.exe scripts\legacy\run_textbook.py --skip-build --retrieval-only --context-char-limit 12000 --result results/book-text-retrieval-final.json
.\.venv\Scripts\python.exe scripts\legacy\run_textbook.py --skip-build --context-char-limit 12000 --result results/book-text-generation-final.json
```

Final combined results are in [`results/book-text-results.json`](book-text-results.json). The saved source, graph, chunks, HNSW index, question annotations, and community map are in `data/book-graph/`. The run used both configured key slots and the three configured models; no credential values are stored in the report. Across the initial full pipeline and the final answer-evaluation pass, 400 successful API requests used 516,617 prompt+completion tokens.
