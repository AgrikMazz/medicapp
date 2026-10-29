# Full-Book Retrieval Tuning Report

**Run date:** 2026-10-08  
**Source:** [`data/medical-CareForInsane/book.txt`](../data/medical-CareForInsane/book.txt)  
**Graph/index:** saved artifacts under `data/book-graph/`; graph rebuild disabled  
**Evaluation artifacts:** [question set](../data/book-graph/eval_questions.reviewed.jsonl), [selected config](book-retrieval-selected-config.json), [development sweep](book-retrieval-tuning.json), [held-out results](book-retrieval-final.json)

This report records the controlled retrieval update. The earlier 24-question pilot in [`book-text-evaluation-report.md`](book-text-evaluation-report.md) remains historical and used unreviewed synthetic questions.

## Objective

Find whether a query-time HNSW/PPR/cross-encoder retrieval path can beat a controlled BM25 baseline on this whole-book corpus, while measuring the context that would actually be sent to generation. The question is supplied by the user at request time. Retrieval ranks passages for that live query, then context packing takes whole source passages in rank order under a token budget; no future query is assumed.

## Corpus and ownership

The run reused the fully built book graph; it did not rebuild or rewrite it. The source has 179,927 Unicode characters and 30,690 words. The stored graph has 95 source T passages, 95 S summaries, 1,261 N entities, 510 R relations, and 18 H/O community pairs (1,997 nodes and 10,325 edges). Its HNSW index contains 208 T/S/H vectors. The earlier source-offset audit found the 95 T passage ranges cover all non-whitespace source characters.

The new work is query-time passage retrieval, controlled baseline comparison, passage-preserving token-budget context packing, the development-only tuning runner, the held-out retrieval runner, application integration for HNSW/PPR/cross-encoder queries, and this report. The graph’s original extraction and enrichment artifacts were left intact.

## Evaluation data and protocol

- **60 questions:** 37 development and 23 held-out. The development split contains 21 legacy questions retained only for tuning plus 16 additional questions; the held-out split contains 23 assistant-authored questions.
- **Held-out categories:** 18 single-fact, 4 multi-evidence reasoning, and 1 multi-evidence lookup. The multi-evidence sample is too small for a strong category-level claim.
- **Correct retrieval:** every evidence quote must exactly match its annotated `[char_start, char_end)` source interval. Recall@k is the fraction of annotated evidence spans fully covered by the union of the top-k T-passage source intervals. MRR counts a passage as relevant when it overlaps any evidence span. nDCG@10 gives a passage a gain equal to the number of annotated spans it overlaps.
- **Context evaluation:** passages are serialized whole, in ranking order, using the GPT-OSS tokenizer and a 4,096-token evidence-context budget. Packed-context span recall checks the exact source offsets against the passages actually selected for the prompt.
- **Split control:** held-out evidence passage IDs are disjoint from development evidence passage IDs; the question-preparation script asserts this. All answer references and labels were reviewed against source text by the assistant, not independently annotated by a human or subject expert.
- **Baselines:** Okapi-style BM25 (`k1=1.5`, `b=0.75`), exact dense cosine (`all-MiniLM-L12-v2`), BM25+dense reciprocal-rank fusion (`c=60`), exact-dense graph PPR, saved HNSW plus lexical entry points and PPR, and cross-encoder reranking.
- **Tuning:** 19 configurations were declared before the development sweep: HNSW candidate count `{10,20,40}` crossed with PPR expansion `{10,20,40}`; exact-dense graph expansion `{10,20,40}` with damping `{0.35,0.5}`; hybrid cross-encoder candidate counts `{20,40,60}`; and one BM25+dense+cross-encoder RRF variant. Graph and HNSW cross-encoder reranking used a fixed candidate cap of 40; the candidate-count sweep was applied to the hybrid reranker. HNSW `ef_search` remained fixed at 200. Selection used development nDCG@10, with MRR, evidence-span recall, and complete evidence as tie-breakers. The held-out split was not scored until the configuration was frozen.
- **Uncertainty:** paired question-level bootstrap intervals use 5,000 resamples and a fixed seed. Questions that share evidence passages are correlated, so these intervals are exploratory rather than a formal population-level significance claim.

The HNSW index retrieves among 208 saved T/S/H vectors; HNSW results and lexical entry points seed shallow PPR, but only original T passages are ranked as answer evidence. The selected setting is HNSW top-20 graph nodes, PPR top-20 T passages, `alpha=0.5`, then cross-encoder reranking over up to 40 T candidates.

## Held-out retrieval results

| Method | Evidence Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 | Packed-context span recall | Questions with all evidence packed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.652 | 0.935 | 0.978 | 0.794 | 0.771 | 0.978 | 0.957 |
| Exact dense cosine | 0.783 | 0.935 | 0.935 | 0.938 | 0.746 | 0.935 | 0.913 |
| BM25 + dense RRF | 0.870 | 0.978 | 1.000 | 0.922 | 0.839 | 0.978 | 0.957 |
| Exact-dense graph PPR + RRF | 0.848 | 0.978 | 1.000 | 0.935 | 0.837 | 0.978 | 0.957 |
| HNSW + lexical entry points + PPR | 0.435 | 0.913 | 1.000 | 0.679 | 0.712 | 1.000 | 1.000 |
| HNSW + PPR + cross-encoder (selected) | 0.848 | 1.000 | 1.000 | **0.957** | **0.923** | 1.000 | 1.000 |

On the frozen held-out set, the selected pipeline beats BM25 by **0.163 MRR** and **0.152 nDCG@10**. Evidence-span Recall@10 improves by **0.022**, from 0.978 to 1.000; both systems are already near ceiling at that cutoff. Every annotated evidence span is present in the selected top-10 and in its packed context.

| Selected-minus-BM25 metric | Difference | Paired 95% bootstrap interval |
| --- | ---: | ---: |
| MRR | +0.163 | [0.057, 0.281] |
| nDCG@10 | +0.152 | [0.060, 0.261] |
| Evidence-span Recall@10 | +0.022 | [0.000, 0.065] |
| All evidence packed | +0.043 | [0.000, 0.130] |

The recall and all-evidence intervals touch zero. The strongest observed changes are in ranking quality (MRR and nDCG), not candidate recall. The development winner also led BM25, but its paired development intervals included zero; the held-out result is the relevant comparison.

| Held-out category | Questions | BM25 nDCG@10 | Selected nDCG@10 | BM25 MRR | Selected MRR | Selected Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Single fact | 18 | 0.793 | 0.916 | 0.806 | 0.944 | 1.000 |
| Multi-evidence reasoning | 4 | 0.676 | 0.949 | 0.688 | 1.000 | 1.000 |
| Multi-evidence lookup | 1 | 0.754 | 0.944 | 1.000 | 1.000 | 1.000 |

The four reasoning questions are encouraging but too few to support a general multi-hop claim; the one lookup question is only a smoke-sized category result.

### What the ablations say

HNSW+PPR without reranking retrieves all annotated spans by rank 10 but ranks them less effectively than BM25 (nDCG@10 0.712 vs 0.771; MRR 0.679 vs 0.794). Adding the cross-encoder raises nDCG@10 to 0.923. This supports the combined candidate-generation and reranking path on this sample; it does not show that PPR alone causes the gain.

The full-corpus cross-encoder diagnostic scores all 95 passages and reaches nDCG@10 0.914. It is an upper-bound-style diagnostic, not the scalable online design. The selected online path scores at most 40 candidates after HNSW/PPR.

## Failure case

For `cur-test-020`, the question asks for both the anatomy and warning signs of choking. The selected system ranks the warning-sign passage above the anatomy passage, so its nDCG@10 is 0.797 versus BM25’s 0.945. Both still retrieve all annotated evidence by rank 10 and pack it into context. The issue is ordering across evidence facets, not missing book coverage.

This is why the result should not be presented as “the graph always improves retrieval.” Candidate expansion improves coverage, and the cross-encoder improves average ordering, but it can still place one required facet too low within the top ranks.

## Code changes and runtime flow

1. Load the existing graph and HNSW index; do not rebuild either.
2. For a live user question, get entry points from the saved HNSW index and lexical matching.
3. Run shallow PPR over the retrieval graph and return ranked original T passages.
4. Cross-encode the first 40 T candidates and rerank them.
5. Pack intact passages under a 4,096-token budget measured with the configured generator tokenizer.
6. Generate from that frozen packed context; evaluate retrieval and generation separately.

The app’s default QA path now supports `hnsw_ppr_crossencoder`, with environment overrides for the HNSW index, candidate count, PPR expansion, damping, reranker candidate count, and retrieval method. `PassageRetriever.cross_rerank` caches per-query passage scores and scores only the candidate IDs it receives in an online query.

## Generation status

The generation evaluation is a separate pass over the selected system’s exact saved packed contexts; it makes no retrieval calls and is recorded in [`book-retrieval-generation-final.json`](book-retrieval-generation-final.json).

| Generation measure | Result |
| --- | ---: |
| Answers generated | 23 / 23 |
| Token F1 vs. one reference answer | 0.260 |
| Exact match | 0.000 |
| ROUGE-L | 0.225 |
| Answers with a source citation | 1.000 |
| Cited source IDs present in packed context | 1.000 |
| Citation recall against annotated evidence IDs | 0.957 |
| Citation precision against annotated evidence IDs | 0.724 |

These are diagnostics, not a correctness score: each question has one concise reference, while generated answers often add relevant-looking detail, lowering lexical overlap. For example, `cur-test-003` asks for one motive but the answer lists five. Citation overlap with annotated evidence is also not true claim-level faithfulness; valid citations can support claims outside the single annotated gold span. No human answer review or semantic faithfulness judge was run.

The 23 successful calls used three configured models (8, 8, and 7 answers) after 22 rate-limit fallback events. This means the generation result is not a controlled single-model comparison. The fallback path completed all answers without exposing credential values; the retrieval comparison is unaffected by this generation-model variation.

## Remaining limitations and next work

- This is one book and 23 assistant-authored held-out questions; it is not evidence of cross-document generalization.
- The held-out questions are not independently human-annotated, and the multi-evidence slice has only five questions.
- Question-level bootstrap intervals do not fully account for repeated evidence passages.
- The cross-encoder was not trained on this book; the cross-encoder and candidate pool are a reranking baseline, not a learned Align-GRAG aligner.
- The graph has no trained Align-GRAG anchor objective, graph/text contrastive training, or graph-conditioned generator path.
- The 4,096-token limit covers serialized evidence, not the system prompt and question overhead. The current budget is conservative but not an exact total prompt-token constraint.
- No clinical/specialist judgment was used; this is a retrieval engineering experiment, not validation of the 1891 manual’s medical content.

Next, collect a larger human-reviewed question set across the book with multiple independently supported facets per question, then test on a second long text PDF. Keep all contexts fixed when comparing generation, and separately score answer completeness, exact citation support, and claim-level faithfulness. Tune HNSW `ef_search` only when the corpus is large enough for ANN approximation to matter; on this 95-passage corpus, exact dense search is a useful control and the HNSW index includes nearly as many summaries as source passages.

## Reproduction

```powershell
.\.venv\Scripts\python.exe scripts\book\prepare_book_eval_questions.py
.\.venv\Scripts\python.exe scripts\book\run_book_retrieval_benchmark.py --mode tune --questions data/book-graph/eval_questions.reviewed.jsonl --output results/book-retrieval-tuning.json --selected-config results/book-retrieval-selected-config.json --context-token-budget 4096 --bootstrap-iterations 1000
.\.venv\Scripts\python.exe scripts\book\run_book_retrieval_benchmark.py --mode final --questions data/book-graph/eval_questions.reviewed.jsonl --output results/book-retrieval-final.json --selected-config results/book-retrieval-selected-config.json --context-token-budget 4096 --bootstrap-iterations 5000
.\.venv\Scripts\python.exe scripts\book\run_book_generation_eval.py
```
