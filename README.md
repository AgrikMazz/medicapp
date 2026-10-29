# Graph RAG for long PDFs

This project builds a page-aware index for one large PDF. Original passages remain in the graph as source evidence; summaries and entity/relation nodes help retrieval find information across sections. Answers are prompted to cite the source pages.

See [the project report](PROJECT_REPORT.md) for the implementation, evaluation, and limitations, and [the architecture diagram](docs/architecture.svg) for the end-to-end flow. The [detailed architecture and latest-run manifest](PROJECT_REPORT_DETAILED.md) explains the full pipeline and its exact inputs, outputs, and metrics.
Runnable scripts are grouped under [`scripts/`](scripts/README.md).

The app's live query path is a practical graph-retrieval prototype: it uses deterministic embedding and PageRank reranking and does not load a trained Align-GRAG aligner. An offline Align-GRAG-style aligner has now been trained and evaluated on the saved medical-book graph; it underperformed BM25 and the current cross-encoder pipeline on the held-out questions. In that run all development rationale/anchor labels fell back to source quotes because the configured API calls were rate-limited, so treat it as a negative result for this prototype and supervision setup, not a faithful reproduction or test of the full paper. See [the Align-GRAG evaluation report](results/align-grag-evaluation-report.md) and its [metrics artifact](results/align-grag-eval.json).

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file with the Groq-compatible chat endpoint and key:

```text
GROQ_API_KEY=...
GROQ_API_URL=https://api.groq.com/openai/v1
ANSWER_CONTEXT_CHAR_LIMIT=24000
```

Neo4j is optional. The current application stores the working graph and HNSW index locally.

## Build and query

```powershell
streamlit run app.py
```

Use the app stages in order:

1. Upload and chunk the PDF.
2. Extract entities, summaries, and relationships.
3. Build the graph and detect communities.
4. Generate community summaries/titles and build semantic indexes.
5. Ask questions. Answers should cite page numbers shown in the retrieved evidence.

The PDF reader preserves page numbers and text offsets for extracted text. It does not OCR scanned pages or fully represent figures and table layout; evaluate parsing quality on the pages that matter to your document.
For each live query, the app ranks original source passages, then packs whole passages in rank order under a 4,096-token evidence budget measured with the configured generator tokenizer. It does not truncate a passage mid-source. Set `ANSWER_CONTEXT_TOKEN_BUDGET` to change this budget and `RAG_CONTEXT_TOKENIZER` to choose the tokenizer. `ANSWER_CONTEXT_CHAR_LIMIT` remains the cap for fallback graph serialization.

## Evaluate against your PDF

For a real, bounded run using the included `medbook.pdf` slice and annotated questions:

```powershell
.\.venv\Scripts\python.exe scripts\medbook\run_medbook_demo.py
```

It indexes PDF pages 3, 4, 10-13, and 18, writes isolated graph/index artifacts to `data/medbook-demo/`, and saves answer and retrieval metrics to `results/medbook-demo-results.json`.
To rerun only the five questions against the saved index, use `python scripts\medbook\run_medbook_demo.py --evaluate-only`.
To recompute retrieval and answer metrics from saved predictions without another API call, use `python scripts\medbook\run_medbook_demo.py --rescore-only`.

The LLM-enriched full-book build can be attempted with `python scripts\medbook\run_medbook_demo.py --full-book`. In the observed run, annotation stalled after 41 of 140 chunks, so it did not produce a completed enriched graph or full-book evaluation result.

The completed passage-only full-book index contains 140 passages covering pages 1-81 (268,429 extracted characters). It is stored under data/medbook-full-text/ and has no LLM-generated entity, relation, or community layer. Build or reuse it with:

~~~powershell
.\.venv\Scripts\python.exe scripts\medbook\run_medbook_fulltext.py
.\.venv\Scripts\python.exe scripts\medbook\run_medbook_fulltext.py --evaluate-only --retrieval-only
.\.venv\Scripts\python.exe scripts\medbook\run_medbook_fulltext.py --generation-only
~~~

For a controlled retrieval comparison with exact-span labels, run:

~~~powershell
.\.venv\Scripts\python.exe scripts\medbook\run_medbook_retrieval_benchmark.py
~~~

This compares BM25, exact dense cosine, BM25+dense RRF, and the current hybrid/PPR/pruning retriever on the same 21 questions, 25 source spans, 140-passage corpus, and 24,000-character context budget. It makes no model API calls. Full rankings, per-question scores, paired bootstrap intervals, and exact final contexts are saved to results/medbook-retrieval-benchmark.json and results/medbook-retrieval-benchmark-contexts.jsonl. See the [project report](PROJECT_REPORT.md) for results and limitations.

### Observed run (2026-10-05)

The saved run indexed 7 of the PDF's 81 pages (3, 4, 10-13, and 18), producing 13 source chunks. The question set has four answerable questions and one intentionally unanswerable question.

| Metric | Result |
| --- | ---: |
| Gold evidence-page recall in retrieved passages | 1.00 |
| Gold evidence-page recall in final answer context | 1.00 |
| Citation-page precision / recall | 1.00 / 1.00 |
| Answer token F1 | 0.400 |
| ROUGE-L / METEOR | 0.353 / 0.356 |
| BLEU-4 / exact match | 0.101 / 0.00 |
| Unanswerable question abstained | yes (1 of 1) |

These are smoke-test results from five hand-authored questions on one selected PDF slice, not a representative quality estimate. Answer string scores average four answerable questions; page and citation scores use the annotated evidence pages. Review answer correctness and evidence spans with a subject-matter reviewer before drawing quality or safety conclusions. The full predictions and run metadata are in `results/medbook-demo-results.json`.

The evaluator first measures evidence-page coverage in the retrieved candidate graph and final context, then generates each answer from that same final retrieved context and scores it. This tests the real retrieval-to-generation path. An oracle-evidence pass can be added as an optional diagnostic, but is not required for this evaluation.

Create a held-out JSONL question file. The whole PDF stays indexed; split questions into development and test sets rather than splitting the PDF into unrelated corpora.

```json
{"id":"q1","question":"Which factors affect the treatment choice?","answers":["..."],"answerable":true,"evidence":[{"page":12,"quote":"..."},{"page":48,"quote":"..."}]}
{"id":"q2","question":"What does the book say about topic X?","answers":[],"answerable":false,"evidence":[]}
```

The `evidence` entries can also use `page_start` and `page_end` for multi-page spans. The medbook controlled retrieval pilot now uses exact source character spans in `data/medbook-retrieval-benchmark.jsonl`; those annotations still need independent domain review. Page-level evidence recall remains a coarse metric for other question files.

After building the graph and index, run evaluation from Python or the notebook:

```python
results = evaluator.evaluate_pdf_questions(
    questions_path="data/pdf_questions_test.jsonl",
    hnsw_index_path="data/hnsw_index.bin",
    graph_path="graph.pkl",
)
evaluator.save_results(results, "results/pdf_eval.json")
```

This reports answer string metrics when reference answers are supplied, evidence-page recall before and after pruning, citation-page precision/recall, and abstention on explicitly unanswerable questions. Treat string metrics as diagnostics for short answers; review long answers with a human rubric for correctness, completeness, and support.

## Index artifacts

The app writes chunks and progress files under `data/`, the graph to `graph.pkl`, and the HNSW vectors to `data/hnsw_index.bin`. The `.nodes.json` sidecar stores the exact vector-label-to-node mapping and a fingerprint; rebuild both index files together if the graph changes.

## Full-book retrieval tuning run

The saved `data/book-graph/` graph and HNSW index for `data/medical-CareForInsane/book.txt` are evaluated without rebuilding. To reproduce the controlled comparison:

```powershell
.\.venv\Scripts\python.exe scripts\book\prepare_book_eval_questions.py
.\.venv\Scripts\python.exe scripts\book\run_book_retrieval_benchmark.py --mode tune --questions data/book-graph/eval_questions.reviewed.jsonl --output results/book-retrieval-tuning.json --selected-config results/book-retrieval-selected-config.json --context-token-budget 4096 --bootstrap-iterations 1000
.\.venv\Scripts\python.exe scripts\book\run_book_retrieval_benchmark.py --mode final --questions data/book-graph/eval_questions.reviewed.jsonl --output results/book-retrieval-final.json --selected-config results/book-retrieval-selected-config.json --context-token-budget 4096 --bootstrap-iterations 5000
.\.venv\Scripts\python.exe scripts\book\run_book_generation_eval.py
.\.venv\Scripts\python.exe scripts\book\run_book_align_grag_eval.py
```

On the 23-question held-out split, HNSW entry points → PPR → cross-encoder reranking scored **MRR 0.957** and **nDCG@10 0.923**, compared with BM25 at **0.794** and **0.771**. It retrieved all annotated evidence spans by rank 10 and packed them under the 4,096-token evidence budget. This is an exploratory, one-book result; the questions were assistant-authored and source-audited, not independently human-annotated. See the [retrieval tuning report](results/book-text-retrieval-tuning-report.md) for baselines, per-category results, confidence intervals, and limitations.

The app’s default QA path can run the same HNSW/PPR/cross-encoder structure. Set `RETRIEVAL_METHOD`, `HNSW_INDEX_PATH`, `RETRIEVAL_HNSW_CANDIDATES`, `GRAPH_RETRIEVAL_EXPAND_K`, `GRAPH_RETRIEVAL_PPR_ALPHA`, and `RETRIEVAL_RERANK_CANDIDATES` in `.env` to override it. The user query drives retrieval and passage packing at request time.
