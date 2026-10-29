# Runnable scripts

The runners are grouped by corpus and purpose and derive data/config paths from the repository root. The example commands assume the repository root is the current directory; from elsewhere, invoke the script by its absolute path.

## Saved medical book graph (`data/book-graph/`)

- `book/prepare_book_eval_questions.py` prepares/reviews the source-span QA set.
- `book/run_book_retrieval_benchmark.py` tunes and runs controlled retrieval comparisons.
- `book/run_book_generation_eval.py` evaluates generation using frozen retrieved contexts.
- `book/run_book_align_grag_eval.py` trains and evaluates the offline Align-GRAG-style aligner on the saved graph.

Example:

```powershell
.\.venv\Scripts\python.exe scripts\book\run_book_align_grag_eval.py
```

## `medbook.pdf` experiments

- `medbook/run_medbook_demo.py` runs the bounded page-slice graph and QA demo.
- `medbook/run_medbook_fulltext.py` builds/uses the full passage-only book index and evaluates retrieval or generation.
- `medbook/run_medbook_retrieval_benchmark.py` compares retrieval methods on exact-span annotations without generation calls.

## LongBench MultiFieldQA-en

- `longbench/run_multifieldqa_eval.py` downloads the official split if needed, compares passage retrieval methods, then evaluates answer generation from frozen BM25 and HNSW/PPR/cross-encoder contexts.
- The report and detailed retrieval/generation artifacts are written under `results/longbench-multifieldqa-en-*`.

Example:

```powershell
.\.venv\Scripts\python.exe scripts\longbench\run_multifieldqa_eval.py --stage all
```

Use `--stage retrieval` to run retrieval only or `--stage generation` to generate from the saved retrieval result.

## Earlier generic book pipeline

- `legacy/run_textbook.py` builds and evaluates a graph from `data/medical-CareForInsane/book.txt`.

The legacy path remains available for reproducibility; the saved graph in `data/book-graph/` is the input to the latest Align-GRAG evaluation, so the Align-GRAG runner does not rebuild it.
