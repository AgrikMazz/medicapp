# RAG Evaluation Metrics

This guide explains how to evaluate the two main parts of a Retrieval-Augmented Generation (RAG) system:

1. **Retrieval:** Did the system find the right information in the document?
2. **Generation:** Given the information it found, did the system produce a correct, complete, supported answer?

For long PDFs, measure the stages separately. A system can fail because it missed a needed passage, because it discarded that passage while packing the context, or because the language model used the evidence badly.

## 1. Retrieval metrics

Before calculating these metrics, create an evaluation set. For each question, record the source passage or exact text spans needed to answer it. For questions requiring several sections, label **all** necessary evidence, not just the first relevant passage.

Definitions used below:

- **Gold evidence:** the passages or spans human reviewers marked as needed for the question.
- **Retrieved results:** passages returned by the search system, in ranked order.
- **Top-k:** the first `k` retrieved passages.

### Recall@k — did retrieval find the evidence?

**Formula:**

```text
Recall@k = relevant gold passages in the top k / all gold passages
```

**Example:** A question needs four passages. The top 10 results contain three of them. Recall@10 = 3 / 4 = **0.75**.

**Meaning:** Higher recall means fewer required passages were missed. This is especially important for multi-section questions.

### Precision@k — how much of the retrieved material is useful?

**Formula:**

```text
Precision@k = relevant passages in the top k / k
```

**Example:** If 6 of the top 10 passages are relevant, Precision@10 = 6 / 10 = **0.60**.

**Meaning:** Higher precision means less irrelevant material is sent onward. Precision and recall often trade off: retrieving more passages can improve recall while adding noise.

### Hit@k — did at least one useful passage appear?

**Formula:**

```text
Hit@k = 1 if at least one gold passage appears in the top k; otherwise 0
```

**Meaning:** Hit@k is useful when a question can be answered from one passage. It is too lenient for questions that need evidence from several parts of a book: finding one of four required passages still counts as a hit.

### MRR — how high was the first useful result?

For each question, find the rank of the first relevant passage. Take its reciprocal, then average across questions.

**Formula:**

```text
MRR = average over questions of (1 / rank of first relevant result)
```

**Example:** If the first relevant result is at rank 2, that question's reciprocal rank is 1/2 = 0.5.

**Meaning:** Higher MRR means the first useful result tends to appear near the top. MRR does not tell you whether the other passages needed for a multi-hop answer were retrieved.

### nDCG@k — did the system rank the best passages near the top?

Give each result a relevance grade, for example 0 = irrelevant, 1 = somewhat useful, and 2 = directly needed. The metric rewards highly relevant passages more when they appear near the top.

**Formula:**

```text
DCG@k  = sum for rank i from 1 to k of ((2^relevance_grade_i - 1) / log2(i + 1))
nDCG@k = DCG@k / ideal DCG@k
```

The **ideal DCG** is the score for the best possible ordering of those same relevance grades. nDCG is between 0 and 1; higher is better.

**Meaning:** Use nDCG when relevance is graded and rank position matters. It can distinguish a directly supporting passage from one that only provides background.

### Final-context evidence recall — did useful evidence survive context packing?

Retrieval systems often fetch passages and then trim or reorder them to fit the language model's context limit. Evaluate the final text actually sent to the model, not just the earlier search results.

**Formula:**

```text
Final-context evidence recall = gold evidence spans present in final context / all gold evidence spans
```

**Example:** A question requires five annotated spans across the PDF. The search found all five, but the context packer dropped one. Final-context evidence recall = 4 / 5 = **0.80**.

**Meaning:** This catches failures caused by pruning, deduplication, or token limits. It is particularly important when answers combine evidence from distant pages.

### Context precision — are the useful results ranked early?

This is a rank-aware measure. At each rank, calculate precision so far, then average that value over the ranks containing relevant passages.

**Formula:**

```text
Context precision@k = average of Precision@i at each relevant rank i up to k
```

**Meaning:** A relevant passage at rank 1 contributes more than one buried at rank 10. This differs from simple Precision@k, which treats all top-k positions equally.

## 2. Generation metrics

For generation evaluation, save the exact retrieved context and give that same context to the answer generator. Keep it fixed across systems being compared. This isolates generation quality from retrieval quality.

### Exact Match (EM) — is the answer exactly the same?

Normalize harmless formatting differences (such as case and punctuation), then compare the answer with the reference.

**Formula:**

```text
EM = 1 if normalized answer exactly equals normalized reference; otherwise 0
```

**Meaning:** Easy to interpret, but strict: a correct paraphrase can score zero.

### Token F1 — how much answer text overlaps the reference?

Count overlapping tokens between the generated answer and reference answer.

**Formula:**

```text
Token precision = overlapping tokens / tokens in generated answer
Token recall    = overlapping tokens / tokens in reference answer
Token F1        = 2 * precision * recall / (precision + recall)
```

**Meaning:** More forgiving than exact match. It measures word overlap, not whether the answer is logically correct or supported by the PDF.

### Answer completeness — did it answer every part?

Break the reference into required facts or claims. Mark which ones the generated answer states correctly.

**Formula:**

```text
Completeness = required reference claims answered correctly / all required reference claims
```

**Meaning:** A useful metric for questions with multiple subparts or evidence spread across the book.

### Faithfulness — are the answer's claims supported by the retrieved context?

Break the answer into factual claims. Check each claim against the context that was provided to the model.

**Formula:**

```text
Faithfulness = answer claims supported by the retrieved context / all factual claims in the answer
```

**Meaning:** A high score means the answer stays grounded in its supplied evidence. It does not guarantee that the evidence itself was complete; use retrieval recall and answer completeness too.

### Citation precision and recall — are page or passage citations useful and complete?

If the system cites pages or passages, check citations claim by claim.

```text
Citation precision = citations that support their associated claims / all citations
Citation recall    = claims needing evidence that have a valid citation / all claims needing evidence
```

**Meaning:** Precision catches irrelevant or incorrect citations. Recall catches factual claims left uncited.

### Abstention accuracy — does it know when the PDF lacks enough evidence?

Label questions as answerable or unanswerable from the document. Count a correct answer on answerable questions and a correct abstention on unanswerable questions as correct decisions.

**Formula:**

```text
Abstention accuracy = correct answer/abstain decisions / all questions
```

**Meaning:** This checks whether the system avoids inventing answers when evidence is absent or incomplete.

## 3. What the NodeRAG paper measures

The NodeRAG paper mainly reports **end-to-end question-answering accuracy** and **the average number of retrieved tokens** on HotpotQA and MuSiQue. For example, it reports:

- **MuSiQue:** 46.29% accuracy and about 5.96K average retrieval tokens.
- **HotpotQA:** 89.50% accuracy and about 5.0K average retrieval tokens.

It also reports **win-plus-tie rates** in pairwise preference evaluations on RAG-QA Arena, along with token counts, and studies component ablations.

These results address two useful questions: how well did the complete system answer the benchmark questions, and how many retrieved tokens did it use? Token count is a context-cost measure; it is **not** a direct measure of evidence recall or retrieval precision. End-to-end accuracy combines retrieval, generation, prompts, and the evaluation procedure, so it cannot by itself show which stage caused an improvement.

The paper does not provide the full separate retrieval-and-generation breakdown listed above, such as gold evidence-span Recall@k, MRR/nDCG, final-context span recall, claim-level faithfulness, and citation precision/recall. Its reported benchmark results should be described as **NodeRAG paper results**, not results measured on this project's PDF.

Source: [NodeRAG: Structuring Graph-based RAG with Heterogeneous Nodes](https://openreview.net/pdf?id=68YmiLGFrz)

## 4. A practical evaluation for large PDFs

For each question, record:

1. The expected answer and its required facts.
2. Every source page and exact text span needed to answer it.
3. Whether the question is answerable from the PDF.
4. Whether it requires connecting multiple sections or pages.

Then run these evaluations separately:

### Retrieval run

Use the question to retrieve passages. Record Recall@k, Precision@k, MRR or nDCG, final-context evidence recall, context token count, and latency. Keep the retrieved passage IDs, source pages, and final packed context for audit.

### Generation run

Freeze the retrieved context and generate answers from it. Score exact match/token F1 where appropriate, completeness, faithfulness, citation precision/recall, and abstention. Review a sample manually because LLM-based judges can make mistakes; ideally use independent reviewers for a subset and report agreement.

### Essential comparisons

- Compare against a simple retrieval baseline such as BM25 using the same chunks, evidence labels, and token budget.
- Compare graph or hybrid retrieval against the baseline at the same context budget; otherwise, a method can appear better simply by sending more text.
- Add an **oracle-context** diagnostic: give the generator the human-labeled evidence spans. This estimates generation quality when retrieval succeeds. It is a diagnostic, not a deployable system result.
- Report results by question type: single-section lookup, cross-section synthesis, multi-hop reasoning, table/figure question, and unanswerable question.
- Report both averages and question-level results. With a small evaluation set, include uncertainty intervals and avoid treating small differences as conclusive.

The main diagnosis is straightforward:

- **Low retrieval recall:** the required evidence was not found.
- **Good search recall but low final-context evidence recall:** context packing or pruning dropped evidence.
- **Good final-context evidence recall but low answer completeness:** generation failed to use all the evidence.
- **Low faithfulness:** the answer includes claims unsupported by the provided passages.
- **Low citation recall/precision:** the answer's source references are missing or do not support the claims.

## References

- [NodeRAG paper](https://openreview.net/pdf?id=68YmiLGFrz)
- [RAGAS paper](https://arxiv.org/abs/2309.15217)
- [RAGAS context precision documentation](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/context_precision/)
- [RAGAS context recall documentation](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/context_recall/)
- [RAGAS faithfulness documentation](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/faithfulness/)

## 5. NodeRAG results against its baselines

NodeRAG compares against **NaiveRAG**, **HyDE**, **LightRAG**, and **GraphRAG**. The paper reports results on HotpotQA, MuSiQue, and MultiHop-RAG, plus pairwise human preference evaluations on RAG-QA Arena. The table below transcribes the main benchmark columns from the paper's Table 1.

| Method | HotpotQA accuracy | HotpotQA avg. retrieved tokens | MuSiQue accuracy | MuSiQue avg. retrieved tokens | MultiHop-RAG score | MultiHop-RAG avg. retrieved tokens |
|---|---:|---:|---:|---:|---:|---:|
| NaiveRAG | 87.50% | 9.8K | 39.43% | 9.6K | 0.56 | 8.9K |
| HyDE | 73.00% | 10.0K | 33.14% | 9.8K | 0.53 | 9.4K |
| LightRAG | 79.00% | 7.1K | 36.00% | 7.4K | 0.50 | 7.9K |
| GraphRAG | 89.00% | 6.6K | 41.71% | 6.6K | 0.53 | 7.4K |
| **NodeRAG** | **89.50%** | **5.0K** | **46.29%** | **5.96K** | **0.57** | **6.1K** |

### How to interpret the comparison

- **HotpotQA:** NodeRAG is 0.50 percentage points above GraphRAG in reported accuracy (89.50% vs. 89.00%) and uses 1.6K fewer retrieved tokens on average (5.0K vs. 6.6K).
- **MuSiQue:** NodeRAG is 4.58 percentage points above GraphRAG (46.29% vs. 41.71%) and uses about 0.64K fewer retrieved tokens (5.96K vs. 6.6K). It is 10.29 points above LightRAG (36.00%).
- **MultiHop-RAG:** The paper labels the answer metric `Sco` (score), rather than accuracy. NodeRAG reports 0.57 with 6.1K tokens; GraphRAG reports 0.53 with 7.4K and LightRAG 0.50 with 7.9K. Do not relabel `Sco` as accuracy without checking the paper's benchmark definition.
- **RAG-QA Arena:** This is a pairwise preference evaluation, not retrieval Recall@k. The paper reports win-plus-tie (W+T) ratios and token usage across domains. For example, in Lifestyle, NodeRAG's W+T is 0.949, versus 0.863 for GraphRAG and 0.817 for LightRAG; its average retrieved-token figure is 3.3K, versus 6.8K and 7.7K respectively.

### What these results do and do not establish

The table supports the paper's claim that NodeRAG performed better on its selected end-to-end benchmarks while using fewer retrieved tokens than these baselines. It is **not** a direct comparison of standalone retrieval metrics: the paper's headline QA accuracy/score combines retrieval and answer generation, and tokens measure context cost rather than whether all gold evidence was retrieved. The Arena W+T score measures preference between answers, not evidence recall.

These results are from the paper's experimental setup, not a rerun on this repository's `medbook.pdf`. When describing the project, attribute them as **published NodeRAG results**. For a local comparison, run NodeRAG-style retrieval and the baselines over the same PDF, question set, gold evidence spans, generator, and context-token budget; then report retrieval and generation metrics separately.

Source: [NodeRAG paper, Table 1](https://openreview.net/pdf?id=68YmiLGFrz).

## 6. NodeRAG retrieval and answer generation

### Retrieval: dual search, graph propagation, then filtering

NodeRAG does not retrieve only the nearest text chunks. It searches a heterogeneous graph whose node types represent original text (`T`), semantic units (`S`), entities (`N`), relations (`R`), attributes (`A`), community-level information (`H`), and overview/title information (`O`). The graph also contains semantic-neighbor edges built with HNSW.

At query time, its retrieval flow is:

1. **Parse the question:** an LLM extracts query entities; the question is also embedded as a vector.
2. **Dual search for entry points:** use exact matching to find entity/overview nodes (`N`/`O`), and vector similarity search (HNSW) to find semantically similar content-rich nodes (the paper describes embeddings for `T`, `S`, `A`, and `H`). This combines lexical/symbolic matching with semantic matching.
3. **Expand through the graph:** run shallow Personalized PageRank (PPR) from the entry points to surface connected “cross nodes,” helping find related evidence that is not itself an obvious semantic match to the question.
4. **Filter and select retrieval nodes:** combine the direct entry points and graph-expanded nodes, then select the content to send to the answer model.

In short:

```text
Question -> entity extraction + question embedding
         -> exact N/O lookup + HNSW search over embedded nodes
         -> entry points -> shallow PPR graph expansion
         -> filter/select evidence nodes -> retrieved context
```

The purpose is to find both **directly matching content** and **structurally connected evidence**. Exact matching can anchor a query to named entities; HNSW can find semantically related passages or summaries; PPR can connect those anchors to other relevant graph nodes.

### Generation: an LLM answers from the selected context

After retrieval, NodeRAG uses the retrieved information as context for a standard language-model answer-generation step. The generator receives the question and selected retrieved content and writes the answer. The paper does not describe a separate trained answer model that consumes a learned aligned-graph token; that trained graph/text alignment and graph-token path belongs to the separate Align-GRAG approach.

For its baseline comparisons, NodeRAG standardizes answer generation across methods: the paper says the methods use GPT-4o-mini, temperature 0, and a shared response prompt. This is meant to reduce the chance that different generation settings explain the observed differences. The exact retriever still differs by method; NodeRAG's contribution is primarily its graph representation and retrieval process.

### Simple mental model

- **Retrieval asks:** “Which graph nodes and connected source passages should be shown to the model?”
- **Generation asks:** “Given the question and those passages, what answer should the model produce?”
- **NodeRAG's distinguishing part:** the first question is handled through typed graph nodes, dual search, and shallow PPR. Its final answer is still produced by a general LLM from the retrieved context.

Source: [NodeRAG paper](https://openreview.net/pdf?id=68YmiLGFrz), Sections 2–3 and Appendix C.4; its baseline-generation settings are described in Section 3.2 / Appendix B.2.
