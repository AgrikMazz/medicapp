"""Controlled full-book retrieval benchmark for the medbook PDF.

Methods share one passage corpus, exact span annotations, and one context budget:
BM25, exact dense cosine, BM25+dense reciprocal-rank fusion, and the current
hybrid-entry/PPR/pruning graph path. No LLM calls are made.
"""

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import pickle
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

sys.path.insert(0, str(ROOT))
from modules.graph_builder import GraphBuilder  # noqa: E402

PASSAGE_PATH = ROOT / "data" / "medbook-full-text" / "chunked_output.jsonl"
GRAPH_PATH = ROOT / "data" / "medbook-full-text" / "graph.pkl"
INDEX_PATH = ROOT / "data" / "medbook-full-text" / "graph-index.bin"
QUESTION_PATH = ROOT / "data" / "medbook-retrieval-benchmark.jsonl"
RESULT_PATH = ROOT / "results" / "medbook-retrieval-benchmark.json"
CONTEXT_PATH = ROOT / "results" / "medbook-retrieval-benchmark-contexts.jsonl"
MODEL_NAME = "all-MiniLM-L12-v2"
RRF_K = 60
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "their", "this",
    "to", "was", "were", "what", "when", "which", "who", "why", "with",
}


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tokens(text):
    return [
        token for token in re.findall(r"[\w]+", str(text).casefold())
        if token not in STOPWORDS
    ]


def bm25_rankings(documents, queries, k1=1.5, b=0.75):
    doc_tokens = [tokens(doc["text"]) for doc in documents]
    n_docs = len(documents)
    average_length = sum(map(len, doc_tokens)) / max(n_docs, 1)
    df = Counter()
    for values in doc_tokens:
        df.update(set(values))
    idf = {
        term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
        for term, freq in df.items()
    }
    doc_freqs = [Counter(values) for values in doc_tokens]
    rankings = []
    for query in queries:
        query_terms = set(tokens(query))
        scores = []
        for doc_index, frequencies in enumerate(doc_freqs):
            length = len(doc_tokens[doc_index])
            score = 0.0
            for term in query_terms:
                tf = frequencies.get(term, 0)
                if tf == 0:
                    continue
                norm = tf + k1 * (1.0 - b + b * length / max(average_length, 1.0))
                score += idf[term] * tf * (k1 + 1.0) / norm
            if score > 0:
                scores.append((doc_index, score))
        scores.sort(key=lambda pair: (-pair[1], pair[0]))
        rankings.append(scores)
    return rankings


def dense_rankings(document_embeddings, query_embeddings):
    similarities = query_embeddings @ document_embeddings.T
    return [
        sorted(
            ((idx, float(score)) for idx, score in enumerate(row)),
            key=lambda pair: (-pair[1], pair[0]),
        )
        for row in similarities
    ]


def rrf_rankings(bm25_rows, dense_rows, n_docs, rrf_k=RRF_K):
    rankings = []
    for bm25, dense in zip(bm25_rows, dense_rows):
        fused = np.zeros(n_docs, dtype=np.float64)
        for row in (bm25, dense):
            for rank, (doc_index, _) in enumerate(row, start=1):
                fused[doc_index] += 1.0 / (rrf_k + rank)
        rankings.append(
            sorted(
                ((idx, float(score)) for idx, score in enumerate(fused) if score > 0),
                key=lambda pair: (-pair[1], pair[0]),
            )
        )
    return rankings


def passage_id_to_node_id(passage_id):
    return "T_" + str(passage_id)


def score_question_passages(model, question, graph, ranked_candidates):
    """Return the same semantic/PPR score used by GraphBuilder.prune_subgraph."""
    node_ids = list(ranked_candidates)
    if not node_ids:
        return []
    texts = [str(graph.nodes[node].get("text", "")) for node in node_ids]
    embeddings = model.encode(
        [question] + texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    similarities = embeddings[1:] @ embeddings[0]
    max_ppr = max(
        (float(graph.nodes[node].get("ppr_score", 0.0)) for node in node_ids),
        default=0.0,
    )
    rows = []
    order = {node: idx for idx, node in enumerate(node_ids)}
    for node, similarity in zip(node_ids, similarities):
        ppr = float(graph.nodes[node].get("ppr_score", 0.0))
        normalized_ppr = ppr / max_ppr if max_ppr > 0 else 0.0
        score = 0.75 * ((float(similarity) + 1.0) / 2.0) + 0.25 * normalized_ppr
        rows.append((node, float(score), order[node]))
    rows.sort(key=lambda row: (-row[1], row[2]))
    return [(node, score) for node, score, _ in rows]


def resolve_graph_rankings(builder, model, questions, node_to_doc, index_path, top_k=10):
    """Run the repository's actual dense+lexical -> PPR -> prune retrieval path."""
    candidate_rankings = []
    final_rankings = []
    for question in questions:
        with contextlib.redirect_stdout(io.StringIO()):
            entries, weights = builder.get_entry_points(
                question, hnsw_index_path=str(index_path), top_k=top_k
            )
            raw_graph, _ = builder.run_shallow_ppr(entries, weights)
            final_graph = builder.prune_subgraph(question, raw_graph)
        raw_passages = [
            node for node, data in raw_graph.nodes(data=True)
            if data.get("type") == "T" and node in node_to_doc
        ]
        ranked_raw = score_question_passages(
            model, question, raw_graph, raw_passages
        )
        final_passages = [
            node for node, data in final_graph.nodes(data=True)
            if data.get("type") == "T" and node in node_to_doc
        ]
        final_passages.sort(
            key=lambda node: (
                -float(final_graph.nodes[node].get("retrieval_score", 0.0)),
                node_to_doc[node]["order"],
            )
        )
        candidate_rankings.append([
            (node_to_doc[node]["id"], score) for node, score in ranked_raw
        ])
        final_rankings.append([
            (
                node_to_doc[node]["id"],
                float(final_graph.nodes[node].get("retrieval_score", 0.0)),
            )
            for node in final_passages
        ])
    return candidate_rankings, final_rankings


def minimum_passage_cover(span, documents_by_page):
    """Find a minimum greedy set of passages whose source offsets cover a span."""
    page = int(span["page"])
    start, end = int(span["char_start"]), int(span["char_end"])
    candidates = [
        doc for doc in documents_by_page.get(page, [])
        if doc["char_start"] < end and doc["char_end"] > start
    ]
    cursor = start
    selected = []
    while cursor < end:
        covering = [
            doc for doc in candidates
            if doc["char_start"] <= cursor < doc["char_end"]
        ]
        if not covering:
            raise ValueError(
                f"No source passage covers {span['id']} p{page} char {cursor}:{end}"
            )
        covering.sort(
            key=lambda doc: (-doc["char_end"], doc["char_end"] - doc["char_start"], doc["order"])
        )
        best = covering[0]
        selected.append(best["id"])
        cursor = min(end, best["char_end"])
    return list(dict.fromkeys(selected))


def add_gold_passages(questions, documents):
    by_page = {}
    for doc in documents:
        by_page.setdefault(doc["page"], []).append(doc)
    for question in questions:
        question["gold_passage_ids"] = []
        for span_index, span in enumerate(question.get("evidence_spans", [])):
            span["id"] = f"{question['id']}:span{span_index + 1}"
            covered_ids = minimum_passage_cover(span, by_page)
            span["cover_passage_ids"] = covered_ids
            question["gold_passage_ids"].extend(covered_ids)
        question["gold_passage_ids"] = sorted(set(question["gold_passage_ids"]))
        if not question["gold_passage_ids"]:
            raise ValueError(f"No gold passages resolved for {question['id']}")


def rank_ids(ranking):
    return [passage_id for passage_id, _ in ranking]


def passage_retrieval_metrics(ranking_ids, gold_ids, cutoffs=(5, 10, 20), ndcg_k=10):
    gold = set(gold_ids)
    ranks = {doc_id: rank for rank, doc_id in enumerate(ranking_ids, start=1)}
    metrics = {}
    for k in cutoffs:
        hits = sum(1 for doc_id in gold if ranks.get(doc_id, math.inf) <= k)
        metrics[f"recall@{k}"] = hits / len(gold) if gold else 0.0
        metrics[f"precision@{k}"] = hits / k
    first_rank = min((ranks[doc_id] for doc_id in gold if doc_id in ranks), default=None)
    metrics["mrr"] = 1.0 / first_rank if first_rank else 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for doc_id, rank in ranks.items()
        if rank <= ndcg_k and doc_id in gold
    )
    ideal_relevant = min(len(gold), ndcg_k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_relevant + 1))
    metrics[f"ndcg@{ndcg_k}"] = dcg / idcg if idcg else 0.0
    return metrics


def pack_context(ranking, documents_by_id, char_budget):
    """Pack whole, page-labelled passages in rank order under a shared char cap."""
    lines = []
    passage_ids = []
    used_chars = 0
    for passage_id, _ in ranking:
        if passage_id in passage_ids:
            continue
        doc = documents_by_id[passage_id]
        page = doc["page"]
        text = str(doc["text"]).strip().replace("\n", " ")
        prefix = f"[T | source={passage_id} | p. {page}] "
        line = prefix + text
        added = len(line) + (1 if lines else 0)
        if used_chars + added > char_budget:
            continue
        lines.append(line)
        passage_ids.append(passage_id)
        used_chars += added
    return "\n".join(lines), passage_ids, used_chars


def interval_covered(start, end, intervals):
    cursor = start
    for left, right in sorted(intervals):
        if right <= cursor:
            continue
        if left > cursor:
            return False
        cursor = max(cursor, right)
        if cursor >= end:
            return True
    return cursor >= end


def context_span_metrics(context_ids, question, documents_by_id):
    intervals_by_page = {}
    for passage_id in context_ids:
        doc = documents_by_id[passage_id]
        intervals_by_page.setdefault(doc["page"], []).append(
            (doc["char_start"], doc["char_end"])
        )
    covered = 0
    span_rows = []
    for span in question["evidence_spans"]:
        is_covered = interval_covered(
            int(span["char_start"]),
            int(span["char_end"]),
            intervals_by_page.get(int(span["page"]), []),
        )
        covered += int(is_covered)
        span_rows.append({
            "span_id": span["id"],
            "covered": is_covered,
            "page": span["page"],
            "char_start": span["char_start"],
            "char_end": span["char_end"],
        })
    count = len(question["evidence_spans"])
    return {
        "final_context_span_recall": covered / count if count else 0.0,
        "covered_span_count": covered,
        "gold_span_count": count,
        "span_results": span_rows,
    }


def bootstrap_mean_ci(values, seed=20261006, iterations=5000):
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(iterations):
        means.append(statistics.fmean(values[rng.randrange(n)] for _ in range(n)))
    means.sort()
    return [means[int(0.025 * (iterations - 1))], means[int(0.975 * (iterations - 1))]]


def paired_delta_ci(left, right, seed=20261006, iterations=5000):
    deltas = [a - b for a, b in zip(left, right)]
    return statistics.fmean(deltas), bootstrap_mean_ci(deltas, seed, iterations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default=str(QUESTION_PATH))
    parser.add_argument("--passages", default=str(PASSAGE_PATH))
    parser.add_argument("--graph", default=str(GRAPH_PATH))
    parser.add_argument("--index", default=str(INDEX_PATH))
    parser.add_argument("--output", default=str(RESULT_PATH))
    parser.add_argument("--contexts-output", default=str(CONTEXT_PATH))
    parser.add_argument("--context-chars", type=int, default=24000)
    parser.add_argument("--dense-entry-k", type=int, default=10)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    args = parser.parse_args()

    question_path = Path(args.questions)
    passage_path = Path(args.passages)
    graph_path = Path(args.graph)
    index_path = Path(args.index)
    questions = read_jsonl(question_path)
    docs = read_jsonl(passage_path)
    for idx, doc in enumerate(docs):
        doc["id"] = str(doc["id"])
        doc["page"] = int(doc.get("page_start") or (doc.get("source_pages") or [None])[0])
        doc["char_start"] = int(doc["char_start"])
        doc["char_end"] = int(doc["char_end"])
        doc["order"] = idx
        if doc["page"] is None:
            raise ValueError(f"Passage lacks source page: {doc['id']}")
    docs_by_id = {doc["id"]: doc for doc in docs}
    if len(docs_by_id) != len(docs):
        raise ValueError("Passage IDs are not unique.")

    # Confirm that the span text and offsets match the original extracted PDF.
    import fitz
    pdf_path = ROOT / "medbook.pdf"
    with fitz.open(pdf_path) as pdf:
        for question in questions:
            for span in question.get("evidence_spans", []):
                extracted = pdf[int(span["page"]) - 1].get_text("text")
                actual = extracted[int(span["char_start"]):int(span["char_end"])]
                if actual != span["quote"]:
                    raise ValueError(
                        f"Span quote/offset mismatch: {question['id']} p{span['page']}"
                    )
    add_gold_passages(questions, docs)

    print(
        f"Loaded {len(docs)} passages, {len(questions)} questions, "
        f"{sum(len(q['evidence_spans']) for q in questions)} exact spans."
    )
    print(f"Encoding passages and questions with {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    passage_embeddings = model.encode(
        [doc["text"] for doc in docs],
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    query_embeddings = model.encode(
        [question["question"] for question in questions],
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    bm25_index_rows = bm25_rankings(docs, [q["question"] for q in questions])
    dense_index_rows = dense_rankings(passage_embeddings, query_embeddings)
    rrf_index_rows = rrf_rankings(bm25_index_rows, dense_index_rows, len(docs))

    def to_passage_ids(index_rows):
        return [
            [(docs[doc_index]["id"], score) for doc_index, score in row]
            for row in index_rows
        ]

    bm25_rows = to_passage_ids(bm25_index_rows)
    dense_rows = to_passage_ids(dense_index_rows)
    rrf_rows = to_passage_ids(rrf_index_rows)

    # The project method uses the saved HNSW, its BM25-like lexical candidates,
    # passage-graph PPR, and the existing semantic/PPR pruning formula.
    builder = GraphBuilder(
        groq_handler=None,
        jsonl_path=str(passage_path),
        neo4j_uri=None,
        neo4j_user=None,
        neo4j_password=None,
        neo4j_database=None,
    )
    with open(graph_path, "rb") as handle:
        builder.G = pickle.load(handle)
    builder._embedding_model = model
    node_to_doc = {}
    for doc in docs:
        node_id = passage_id_to_node_id(doc["id"])
        if node_id not in builder.G:
            raise ValueError(f"Passage is absent from graph: {node_id}")
        node_to_doc[node_id] = doc
    graph_candidates, graph_finals = resolve_graph_rankings(
        builder, model, [q["question"] for q in questions], node_to_doc,
        index_path, top_k=args.dense_entry_k,
    )

    methods = {
        "bm25": bm25_rows,
        "dense_exact_cosine": dense_rows,
        "bm25_dense_rrf": rrf_rows,
        "project_hybrid_ppr": graph_candidates,
    }
    final_rankings = {
        "bm25": bm25_rows,
        "dense_exact_cosine": dense_rows,
        "bm25_dense_rrf": rrf_rows,
        "project_hybrid_ppr": graph_finals,
    }

    per_question = []
    context_rows = []
    for q_idx, question in enumerate(questions):
        q_result = {
            "id": question["id"],
            "question": question["question"],
            "evaluation_split": (
                "previous_smoke_question" if question["id"].startswith("mb-full-")
                else "newly_annotated_question"
            ),
            "gold_pages": question.get("evidence_pages", []),
            "gold_spans": question["evidence_spans"],
            "gold_passage_ids": question["gold_passage_ids"],
            "methods": {},
        }
        for method_name, rankings in methods.items():
            ranking = rankings[q_idx]
            ordered_ids = rank_ids(ranking)
            retrieval_metrics = passage_retrieval_metrics(
                ordered_ids, question["gold_passage_ids"]
            )
            context, context_ids, used_chars = pack_context(
                final_rankings[method_name][q_idx],
                docs_by_id,
                args.context_chars,
            )
            span_metrics = context_span_metrics(context_ids, question, docs_by_id)
            method_result = {
                **retrieval_metrics,
                **{k: v for k, v in span_metrics.items() if k != "span_results"},
                "candidate_count": len(ordered_ids),
                "candidate_top_pages": [
                    docs_by_id[pid]["page"] for pid in ordered_ids[:20]
                ],
                "ranked_passage_ids": ordered_ids,
                "ranked_scores": [float(score) for _, score in ranking],
                "final_context_passage_ids": context_ids,
                "final_context_pages": [
                    docs_by_id[pid]["page"] for pid in context_ids
                ],
                "final_context_chars": used_chars,
                "span_results": span_metrics["span_results"],
            }
            q_result["methods"][method_name] = method_result
            context_rows.append({
                "question_id": question["id"],
                "method": method_name,
                "question": question["question"],
                "context_passage_ids": context_ids,
                "context_pages": method_result["final_context_pages"],
                "context_chars": used_chars,
                "context": context,
            })
        per_question.append(q_result)

    metric_names = [
        "recall@5", "recall@10", "recall@20",
        "precision@10", "mrr", "ndcg@10", "final_context_span_recall",
    ]

    def summarize_rows(question_rows):
        summary = {}
        for method_name in methods:
            summary[method_name] = {}
            for metric in metric_names:
                values = [row["methods"][method_name][metric] for row in question_rows]
                summary[method_name][metric] = {
                    "mean": statistics.fmean(values),
                    "bootstrap_95pct_ci": bootstrap_mean_ci(
                        values, iterations=args.bootstrap_iterations
                    ),
                }
            span_counts = [
                row["methods"][method_name]["gold_span_count"]
                for row in question_rows
            ]
            covered_counts = [
                row["methods"][method_name]["covered_span_count"]
                for row in question_rows
            ]
            summary[method_name]["final_context_span_recall_micro"] = (
                sum(covered_counts) / sum(span_counts) if sum(span_counts) else 0.0
            )
            summary[method_name]["mean_context_chars"] = statistics.fmean(
                row["methods"][method_name]["final_context_chars"]
                for row in question_rows
            )
            summary[method_name]["mean_context_passages"] = statistics.fmean(
                len(row["methods"][method_name]["final_context_passage_ids"])
                for row in question_rows
            )

        paired = {}
        for method_name in methods:
            if method_name == "bm25":
                continue
            paired[method_name] = {}
            for metric in ("mrr", "ndcg@10", "recall@10", "final_context_span_recall"):
                left = [row["methods"][method_name][metric] for row in question_rows]
                right = [row["methods"]["bm25"][metric] for row in question_rows]
                mean_delta, ci = paired_delta_ci(
                    left, right, iterations=args.bootstrap_iterations
                )
                paired[method_name][metric] = {
                    "mean_delta_vs_bm25": mean_delta,
                    "paired_bootstrap_95pct_ci": ci,
                }
        return {"methods": summary, "paired_deltas_vs_bm25": paired}

    overall = summarize_rows(per_question)
    summary = overall["methods"]
    paired_deltas = overall["paired_deltas_vs_bm25"]
    split_summaries = {}
    for split_name in ("previous_smoke_question", "newly_annotated_question"):
        subset = [row for row in per_question if row["evaluation_split"] == split_name]
        if subset:
            split_summaries[split_name] = {
                "question_count": len(subset),
                **summarize_rows(subset),
            }

    output = {
        "run_metadata": {
            "source_pdf": "medbook.pdf",
            "pdf_sha256": sha256_file(pdf_path),
            "passage_file": str(passage_path.relative_to(ROOT)),
            "passage_file_sha256": sha256_file(passage_path),
            "question_file": str(question_path.relative_to(ROOT)),
            "question_file_sha256": sha256_file(question_path),
            "question_count": len(questions),
            "previous_smoke_question_count": sum(q["id"].startswith("mb-full-") for q in questions),
            "newly_annotated_question_count": sum(q["id"].startswith("mb-span-") for q in questions),
            "evidence_span_count": sum(len(q["evidence_spans"]) for q in questions),
            "passage_count": len(docs),
            "index_mode": "all-book T-passage graph with document-order edges",
            "embedding_model": MODEL_NAME,
            "retrieval_methods": [
                "BM25 Okapi (k1=1.5, b=0.75)",
                "Exact dense cosine over normalized sentence embeddings",
                f"BM25+dense reciprocal rank fusion (RRF k={RRF_K})",
                "Repository hybrid HNSW+lexical entry points, PPR, and alignment pruning",
            ],
            "project_hybrid_parameters": {
                "dense_hnsw_entry_k": args.dense_entry_k,
                "lexical_entry_k": 8,
                "ppr_alpha": 0.5,
                "ppr_max_iter": 10,
                "ppr_top_passages": 5,
                "prune_top_n": 12,
                "reranker_semantic_weight": 0.75,
                "reranker_ppr_weight": 0.25,
                "context_packing": "ranked whole passages, page-labelled, no graph-edge text",
            },
            "context_budget_chars": args.context_chars,
            "ranking_cutoffs": [5, 10, 20],
            "nDCG_cutoff": 10,
            "rrf_constant": RRF_K,
            "bootstrap_iterations": args.bootstrap_iterations,
            "bootstrap_seed": 20261006,
            "generation_calls": 0,
        },
        "summary": summary,
        "paired_deltas_vs_bm25": paired_deltas,
        "split_summaries": split_summaries,
        "questions": per_question,
        "contexts_file": str(Path(args.contexts_output).relative_to(ROOT)),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
    contexts_path = Path(args.contexts_output)
    contexts_path.parent.mkdir(parents=True, exist_ok=True)
    with open(contexts_path, "w", encoding="utf-8") as handle:
        for row in context_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Retrieval benchmark summary:")
    for method_name, metrics in summary.items():
        print(f"\n{method_name}")
        for metric in metric_names:
            row = metrics[metric]
            print(
                f"  {metric}: {row['mean']:.3f} "
                f"(bootstrap 95% CI {row['bootstrap_95pct_ci'][0]:.3f}-"
                f"{row['bootstrap_95pct_ci'][1]:.3f})"
            )
        print(
            "  final_context_span_recall_micro: "
            f"{metrics['final_context_span_recall_micro']:.3f}"
        )
        print(
            "  mean_context_chars/passages: "
            f"{metrics['mean_context_chars']:.0f}/{metrics['mean_context_passages']:.1f}"
        )
    print(f"\nSaved metrics and full rankings to {output_path}")
    print(f"Saved exact per-method contexts to {contexts_path}")


if __name__ == "__main__":
    main()

