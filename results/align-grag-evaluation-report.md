# Align-GRAG Core Aligner Evaluation

**Run date:** 2026-10-08  
**Graph:** existing saved book graph, read-only; SHA-256 `f0b9ed825613400d8202b6b6a14031983ce2ca612d638c91bcd978aa826d8230`  
**Graph size:** 1,997 nodes, 10,325 edges, 95 source passages  
**Question set:** 37 development questions and 23 frozen held-out questions  
**Evaluation boundary:** graph/index are read-only; no graph rebuild; initial candidate subgraphs use the existing saved HNSW + shallow PPR path.

## What was evaluated

This experiment implements Align-GRAG's trainable refinement stage on the existing graph candidate subgraphs. A two-layer GraphSAGE encoder (implemented in plain PyTorch because `torch_geometric` is absent) scores nodes conditioned on the query. Training combines anchor-target KL loss and bidirectional in-batch graph/rationale contrastive loss. Inference selects the top 20 aligned nodes and preserves their one-hop neighbors; source T passages in that pruned graph are ranked by node score.

Development-only rationale/anchor examples use the question, its annotated answer, and the retrieved candidate graph. All 37/37 development examples used source-quote/answer-derived rationale and anchor fallbacks; none used an LLM-generated rationale or anchor. All configured key/model calls were rate-limited. Fallback rationales are constructed from the labeled answer and source quotes, so they are weaker supervision than the paper's question-answer-rationale-anchor examples. No held-out answer, rationale, or anchor was used to train or select settings. Hyperparameters were selected with 4-fold grouped cross-validation on development questions, grouping questions that share evidence passages. Selected settings: 5 epochs, 20 seed nodes; learning rate 0.0001, weight decay 0.0001.

The initial candidate retriever is held constant as HNSW + PPR to isolate learned alignment/pruning. Text features for graph nodes, questions, anchors, and rationales use `all-MiniLM-L12-v2`; graph states come from the trained GraphSAGE encoder. This is **not a full reproduction** of the paper's SBERT top-k node/edge + PCST retrieval. It implements the central KL and graph-text contrastive objectives, with a smaller GraphSAGE encoder for this graph and runtime.

## Held-out retrieval results

Gold relevance uses the same exact source character offsets and metric definitions as the previous controlled benchmark. Context uses the same passage-preserving 4,096-token budget.

R@k is the fraction of labeled evidence spans found in the top k passages. MRR measures the reciprocal rank of the first supporting passage. nDCG@10 rewards placing more supporting passages near the top. Packed span recall measures evidence spans present in the final packed context; "all evidence packed" is the fraction of questions whose complete labeled evidence fits in that context.

| Method | R@1 | R@5 | R@10 | MRR | nDCG@10 | Packed span recall | All evidence packed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25 | 0.652 | 0.935 | 0.978 | 0.794 | 0.771 | 0.978 | 0.957 |
| hnsw_ppr | 0.435 | 0.913 | 1.000 | 0.679 | 0.712 | 1.000 | 1.000 |
| align_grag_kl_only | 0.022 | 0.239 | 0.630 | 0.163 | 0.188 | 0.565 | 0.522 |
| align_grag_dual | 0.022 | 0.326 | 0.609 | 0.176 | 0.201 | 0.522 | 0.478 |
| hnsw_ppr_crossencoder_40 | 0.848 | 1.000 | 1.000 | 0.957 | 0.923 | 1.000 | 1.000 |

### Align-GRAG dual results by held-out question category

| Category | Questions | R@10 | MRR | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| multi_evidence_lookup | 1 | 1.000 | 1.000 | 0.559 |
| multi_evidence_reasoning | 4 | 0.625 | 0.226 | 0.239 |
| single_fact | 18 | 0.583 | 0.119 | 0.173 |

Align-GRAG dual alignment changes nDCG@10 by **-0.721** versus the current HNSW/PPR + cross-encoder system and **-0.569** versus BM25. The node-only ablation has nDCG@10 0.188; the dual model has 0.201. Paired per-question bootstrap 95% intervals (5,000 resamples) for Align dual minus cross-encoder are nDCG@10 [-0.800, -0.643], MRR [-0.866, -0.676], and R@10 [-0.587, -0.196]. Against BM25, the corresponding intervals are nDCG@10 [-0.677, -0.461], MRR [-0.746, -0.483], and R@10 [-0.565, -0.174].

For a concrete miss, on `cur-test-009` (recovering from dementia), Align dual retrieved no labeled evidence in its top 10 (nDCG@10 0.000), while BM25 and the cross-encoder each ranked the evidence first (nDCG@10 1.000). This illustrates the severe ranking failure behind the averages.

## Important scope limits

- The held-out set is small (23 questions), and the QA/evidence labels were assistant-authored and source-audited, not independently annotated by domain experts. Category counts are imbalanced, especially for multi-evidence questions.
- The existing graph and its candidate retrieval are evaluated; graph extraction quality is not changed.
- To produce comparable ranked passages, source `T` nodes retained after pruning are ordered by their aligner node logits. That is an evaluation adapter for this prototype; the paper's downstream model consumes the aligned graph representation directly.
- The current hosted Groq API accepts text, not a learned continuous graph token. Therefore the aligned graph token is not injected into the generator, and this is a retrieval/pruning comparison, not a complete Align-GRAG generation result.
- `results/align-grag-supervision.jsonl` contains development-only rationale/anchor supervision and provider/model metadata without credentials.

## Reproduction

```powershell
.\.venv\Scripts\python.exe scripts\book\run_book_align_grag_eval.py
```

Primary result artifact: `results/align-grag-eval.json`.
