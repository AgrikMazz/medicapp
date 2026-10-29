"""Tune and evaluate passage retrieval on the already-built book graph.

This script never rebuilds the book graph. It compares BM25, exact dense cosine,
hybrid RRF, cross-encoder reranking, and PPR expansion, and scores the context
that would actually be sent to generation under a model-token budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import random
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from modules.graph_builder import GraphBuilder  # noqa: E402
from modules.retrieval import (  # noqa: E402
    PassageRetriever,
    TokenBudgetPacker,
    reciprocal_rank_fusion,
)


DEFAULT_GRAPH = ROOT / "data" / "book-graph" / "graph.pkl"
DEFAULT_HNSW_INDEX = ROOT / "data" / "book-graph" / "graph-index.bin"
DEFAULT_SOURCE = ROOT / "data" / "medical-CareForInsane" / "book.txt"
DEFAULT_QUESTIONS = ROOT / "data" / "book-graph" / "eval_questions.reviewed.jsonl"
DEFAULT_OUTPUT = ROOT / "results" / "book-retrieval-tuning.json"
DEFAULT_TUNING = ROOT / "results" / "book-retrieval-selected-config.json"
EMBEDDING_MODEL = "all-MiniLM-L12-v2"
DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized(text):
    return re.sub(r"\s+", " ", str(text).casefold()).strip()


def covered(start, end, intervals):
    cursor = int(start)
    for left, right in sorted(intervals):
        if right <= cursor:
            continue
        if left > cursor:
            return False
        cursor = max(cursor, right)
        if cursor >= end:
            return True
    return cursor >= end


def interval_metrics(ranking, question, retriever, max_k=10):
    spans = question.get("evidence_spans", [])
    passage_by_id = retriever.by_id
    records = []
    for node_id, score in ranking:
        passage = passage_by_id[node_id]
        records.append((passage.char_start, passage.char_end, float(score)))

    relevance = []
    for left, right, _ in records:
        relevance.append(sum(
            left < int(span["char_end"]) and right > int(span["char_start"])
            for span in spans
        ))

    first_relevant = next(
        (rank for rank, value in enumerate(relevance, start=1) if value > 0), None
    )
    ideal_relevance = sorted(
        (
            sum(
                passage.char_start < int(span["char_end"])
                and passage.char_end > int(span["char_start"])
                for span in spans
            )
            for passage in retriever.passages
        ),
        reverse=True,
    )

    def dcg(values):
        return sum(
            (2 ** value - 1) / math.log2(rank + 1)
            for rank, value in enumerate(values[:max_k], start=1)
        )

    ideal_dcg = dcg(ideal_relevance)
    metrics = {
        "mrr": 1 / first_relevant if first_relevant else 0.0,
        "ndcg@10": dcg(relevance) / ideal_dcg if ideal_dcg else 0.0,
    }
    for k in (1, 5, 10, 20):
        chosen = [(start, end) for start, end, _ in records[:k]]
        span_recall = sum(
            covered(span["char_start"], span["char_end"], chosen)
            for span in spans
        ) / len(spans) if spans else 0.0
        metrics[f"evidence_span_recall@{k}"] = span_recall
        metrics[f"complete_evidence@{k}"] = float(
            bool(spans) and all(
                covered(span["char_start"], span["char_end"], chosen)
                for span in spans
            )
        )
    metrics["evidence_span_count"] = len(spans)
    return metrics


def pack_ranking(ranking, retriever, packer, token_budget):
    passage_rows = [
        retriever.passage_context_record(node_id, score)
        for node_id, score in ranking
    ]
    return packer.pack("", passage_rows, token_budget=token_budget)


def hnsw_ppr_ranking(query, builder, retriever, index_path, hnsw_k, expand_k, alpha=0.5):
    """Run saved HNSW + lexical entry points, PPR, and return source passages.

    The index contains T, S, and H nodes. Entry nodes seed PPR; only original
    T passages can appear in the returned ranking and evidence context.
    """
    del builder  # The retriever owns the same loaded graph builder.
    return retriever.hnsw_graph_expand(
        query,
        hnsw_index_path=str(index_path),
        hnsw_candidate_k=int(hnsw_k),
        expand_k=int(expand_k),
        alpha=float(alpha),
        max_iter=10,
    )


def context_metrics(packed, question, retriever):
    spans = question.get("evidence_spans", [])
    intervals = [
        (retriever.by_id[node_id].char_start, retriever.by_id[node_id].char_end)
        for node_id in packed["passage_ids"]
    ]
    context = normalized(packed["context"])
    quote_hits = [
        normalized(span.get("quote", "")) in context
        for span in spans
    ]
    return {
        "packed_context_span_recall": (
            sum(
                covered(span["char_start"], span["char_end"], intervals)
                for span in spans
            ) / len(spans) if spans else 0.0
        ),
        "packed_context_exact_quote_recall": (
            sum(quote_hits) / len(quote_hits) if quote_hits else 0.0
        ),
        "packed_context_all_evidence": float(bool(spans) and all(
            covered(span["char_start"], span["char_end"], intervals)
            for span in spans
        )),
        "packed_context_all_quotes": float(bool(quote_hits) and all(quote_hits)),
        "packed_context_tokens": packed["token_count"],
        "packed_passage_count": len(packed["passage_ids"]),
    }


def bootstrap_ci(values, seed=20261008, iterations=2000):
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    n = len(values)
    samples = sorted(
        statistics.fmean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(iterations)
    )
    return [samples[int(0.025 * (iterations - 1))], samples[int(0.975 * (iterations - 1))]]


def mean_summary(question_rows, method_names, packer, retriever, token_budget, bootstrap_iterations=2000):
    fields = (
        "evidence_span_recall@1", "evidence_span_recall@5",
        "evidence_span_recall@10", "evidence_span_recall@20",
        "complete_evidence@10", "mrr", "ndcg@10",
        "packed_context_span_recall", "packed_context_all_evidence",
        "packed_context_tokens", "packed_passage_count",
    )
    output = {}
    for method in method_names:
        output[method] = {}
        for field in fields:
            values = [row["methods"][method][field] for row in question_rows]
            output[method][field] = statistics.fmean(values) if values else 0.0
            output[method].setdefault("bootstrap_95pct_ci", {})[field] = bootstrap_ci(
                values,
                seed=20261008 + sum(map(ord, method + field)),
                iterations=bootstrap_iterations,
            )
        output[method]["question_count"] = len(question_rows)

    if "bm25" in method_names:
        for method in method_names:
            if method == "bm25":
                continue
            output[method]["delta_vs_bm25"] = {
                field: output[method][field] - output["bm25"][field]
                for field in (
                    "evidence_span_recall@10", "complete_evidence@10", "mrr", "ndcg@10",
                    "packed_context_span_recall", "packed_context_all_evidence",
                )
            }
            output[method]["paired_delta_95pct_ci_vs_bm25"] = {}
            for field in (
                "evidence_span_recall@10", "complete_evidence@10", "mrr", "ndcg@10",
                "packed_context_span_recall", "packed_context_all_evidence",
            ):
                deltas = [
                    row["methods"][method][field] - row["methods"]["bm25"][field]
                    for row in question_rows
                ]
                output[method]["paired_delta_95pct_ci_vs_bm25"][field] = bootstrap_ci(
                    deltas,
                    seed=20261008 + sum(map(ord, method + field + "paired")),
                    iterations=bootstrap_iterations,
                )
    return output


def question_result(question, method_rankings, retriever, packer, token_budget):
    result = {
        "id": question["id"],
        "question": question["question"],
        "category": question.get("question_category"),
        "split": question.get("split"),
        "methods": {},
    }
    for method, ranking in method_rankings.items():
        retrieval = interval_metrics(ranking, question, retriever)
        packed = pack_ranking(ranking, retriever, packer, token_budget)
        result["methods"][method] = {
            **retrieval,
            **context_metrics(packed, question, retriever),
            "ranked_passage_ids": [node_id for node_id, _ in ranking],
            "ranked_scores": [float(score) for _, score in ranking],
            "packed_passage_ids": packed["passage_ids"],
            "packed_context": packed["context"],
        }
    return result


def validate_questions(questions, source_text):
    seen_ids = set()
    for question in questions:
        qid = question.get("id")
        if not qid or qid in seen_ids:
            raise ValueError(f"Missing or duplicate question ID: {qid!r}")
        seen_ids.add(qid)
        if not str(question.get("question", "")).strip():
            raise ValueError(f"Question text is empty: {qid}")
        if question.get("split") not in {"dev", "heldout"}:
            raise ValueError(f"Question {qid} must have split='dev' or 'heldout'.")
        spans = question.get("evidence_spans", [])
        if not spans:
            raise ValueError(f"Question {qid} has no gold source spans.")
        for span in spans:
            start, end = int(span["char_start"]), int(span["char_end"])
            if start < 0 or end <= start or end > len(source_text):
                raise ValueError(f"Invalid span bounds in {qid}: {start}:{end}")
            quote = span.get("quote")
            if not quote or source_text[start:end] != quote:
                raise ValueError(f"Evidence quote does not match source offsets in {qid}.")


def build_rankings(question, builder, retriever, config, index_path, cross_scores=True):
    bm25 = retriever.bm25(question)
    dense = retriever.dense(question)
    hybrid = retriever.rrf(bm25, dense, rrf_k=config.get("rrf_k", 60))
    graph = retriever.graph_expand(
        bm25,
        dense,
        seed_k=config.get("graph_seed_k", 20),
        expand_k=config.get("graph_expand_k", 20),
        alpha=config.get("ppr_alpha", 0.35),
        max_iter=config.get("ppr_max_iter", 20),
        rrf_k=config.get("rrf_k", 60),
    )
    hnsw_graph = hnsw_ppr_ranking(
        question,
        builder,
        retriever,
        index_path,
        hnsw_k=config.get("hnsw_candidate_k", 20),
        expand_k=config.get("hnsw_expand_k", 20),
        alpha=config.get("hnsw_ppr_alpha", 0.5),
    )

    rows = {
        "bm25": bm25,
        "dense_exact": dense,
        "hybrid_rrf": hybrid,
        "graph_ppr_rrf": graph,
        "hnsw_ppr": hnsw_graph,
    }
    candidate_k = int(config.get("cross_encoder_candidate_k", 40))
    if cross_scores and retriever.cross_encoder is not None:
        rows["crossencoder_all"] = retriever.cross_rerank(
            question, [passage.node_id for passage in retriever.passages]
        )
        rows[f"hybrid_crossencoder_{candidate_k}"] = retriever.cross_rerank(
            question, hybrid[:candidate_k]
        )
        rows[f"graph_crossencoder_{candidate_k}"] = retriever.cross_rerank(
            question, graph[:candidate_k]
        )
        rows[f"hnsw_ppr_crossencoder_{candidate_k}"] = retriever.cross_rerank(
            question, hnsw_graph[:candidate_k]
        )
        rows[f"hybrid_crossencoder_rrf_{candidate_k}"] = reciprocal_rank_fusion(
            (bm25, dense, rows[f"hybrid_crossencoder_{candidate_k}"]),
            rrf_k=config.get("rrf_k", 60),
            tie_order=retriever.order,
        )
    return rows


def parameter_grid():
    """Compact development-only sweep with 19 declared candidates."""
    for expand_k in (10, 20, 40):
        for alpha in (0.35, 0.5):
            yield {
                "selected_method": "graph_crossencoder_40",
                "rrf_k": 60,
                "graph_seed_k": 20,
                "graph_expand_k": expand_k,
                "ppr_alpha": alpha,
                "ppr_max_iter": 20,
                "cross_encoder_candidate_k": 40,
            }
    for hnsw_k in (10, 20, 40):
        for expand_k in (10, 20, 40):
            yield {
                "selected_method": "hnsw_ppr_crossencoder_40",
                "rrf_k": 60,
                "hnsw_candidate_k": hnsw_k,
                "hnsw_expand_k": expand_k,
                "hnsw_ppr_alpha": 0.5,
                "cross_encoder_candidate_k": 40,
            }
    for candidate_k in (20, 40, 60):
        yield {
            "selected_method": f"hybrid_crossencoder_{candidate_k}",
            "rrf_k": 60,
            "cross_encoder_candidate_k": candidate_k,
        }
    yield {
        "selected_method": "hybrid_crossencoder_rrf_40",
        "rrf_k": 60,
        "cross_encoder_candidate_k": 40,
    }


def tune_on_dev(questions, builder, retriever, index_path):
    dev = [question for question in questions if question["split"] == "dev"]
    if len(dev) < 5:
        raise ValueError("Tuning requires at least five development questions.")
    bm25_metrics = []
    base_rows = {}
    for index, question in enumerate(dev, start=1):
        query = question["question"]
        bm25 = retriever.bm25(query)
        dense = retriever.dense(query)
        hybrid = retriever.rrf(bm25, dense, rrf_k=60)
        base_rows[question["id"]] = (bm25, dense, hybrid)
        bm25_metrics.append(interval_metrics(bm25, question, retriever))
        # Score each query-passage pair once; every sweep variant reuses these.
        retriever.cross_scores(query)
        print(f"Cross-encoder precompute {index}/{len(dev)}: {question['id']}")
    leaderboard = []
    configs = list(parameter_grid())
    for config_index, candidate_config in enumerate(configs, start=1):
        per_q = []
        for question in dev:
            question_text = question["question"]
            bm25, dense, hybrid = base_rows[question["id"]]
            method = candidate_config["selected_method"]
            if method.startswith("graph_crossencoder"):
                candidates = retriever.graph_expand(
                    bm25,
                    dense,
                    seed_k=candidate_config["graph_seed_k"],
                    expand_k=candidate_config["graph_expand_k"],
                    alpha=candidate_config["ppr_alpha"],
                    max_iter=candidate_config["ppr_max_iter"],
                    rrf_k=candidate_config["rrf_k"],
                )
                ranking = retriever.cross_rerank(
                    question_text, candidates[:candidate_config["cross_encoder_candidate_k"]]
                )
            elif method.startswith("hnsw_ppr_crossencoder"):
                candidates = hnsw_ppr_ranking(
                    question_text,
                    builder,
                    retriever,
                    index_path,
                    hnsw_k=candidate_config["hnsw_candidate_k"],
                    expand_k=candidate_config["hnsw_expand_k"],
                    alpha=candidate_config["hnsw_ppr_alpha"],
                )
                ranking = retriever.cross_rerank(
                    question_text, candidates[:candidate_config["cross_encoder_candidate_k"]]
                )
            else:
                candidate_k = candidate_config["cross_encoder_candidate_k"]
                ce = retriever.cross_rerank(question_text, hybrid[:candidate_k])
                if method.startswith("hybrid_crossencoder_rrf"):
                    ranking = reciprocal_rank_fusion(
                        (bm25, dense, ce), rrf_k=candidate_config["rrf_k"],
                        tie_order=retriever.order,
                    )
                else:
                    ranking = ce
            per_q.append({
                "selected": interval_metrics(ranking, question, retriever),
            })
        fields = ("ndcg@10", "mrr", "evidence_span_recall@10", "complete_evidence@10")
        selected_summary = {
            field: statistics.fmean(row["selected"][field] for row in per_q)
            for field in fields
        }
        # Select only on development nDCG, with MRR and complete evidence as
        # tie-breakers; the held-out split is never inspected here.
        objective = tuple(selected_summary[field] for field in fields)
        row = {
            "config": candidate_config,
            "method": candidate_config["selected_method"],
            "summary": selected_summary,
            "objective": objective,
        }
        leaderboard.append(row)
        if config_index % 3 == 0 or config_index == len(configs):
            print(f"Tuning configurations scored: {config_index}/{len(configs)}")
    leaderboard.sort(key=lambda row: row["objective"], reverse=True)
    best = leaderboard[0]
    selected_config = dict(best["config"])
    bm25_dev = {
        field: statistics.fmean(row[field] for row in bm25_metrics)
        for field in ("ndcg@10", "mrr", "evidence_span_recall@10", "complete_evidence@10")
    }
    return {
        "selected_config": selected_config,
        "selected_dev_summary": best["summary"],
        "bm25_dev_summary": bm25_dev,
        "grid_trials": len(leaderboard),
        "top_10_configs": leaderboard[:10],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--graph", default=str(DEFAULT_GRAPH))
    parser.add_argument("--hnsw-index", default=str(DEFAULT_HNSW_INDEX))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--selected-config", default=str(DEFAULT_TUNING))
    parser.add_argument("--mode", choices=("tune", "final"), default="tune")
    parser.add_argument("--cross-encoder", default=DEFAULT_CROSS_ENCODER)
    parser.add_argument(
        "--tokenizer",
        default=(os.getenv("GROQ_MODEL") or "openai/gpt-oss-20b").strip(),
        help="Tokenizer used to measure serialized evidence context tokens.",
    )
    parser.add_argument("--context-token-budget", type=int, default=4096)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    args = parser.parse_args()

    source_text = Path(args.source).read_text(encoding="utf-8-sig")
    questions = read_jsonl(args.questions)
    validate_questions(questions, source_text)
    dev_questions = [question for question in questions if question["split"] == "dev"]
    heldout_questions = [question for question in questions if question["split"] == "heldout"]
    if not dev_questions or not heldout_questions:
        raise ValueError("The question file must contain both dev and heldout rows.")

    print(f"Loading saved graph (no rebuild): {args.graph}")
    builder = GraphBuilder(
        groq_handler=None,
        jsonl_path=str(Path(args.graph).with_name("chunked_output.jsonl")),
        neo4j_uri=None,
        neo4j_user=None,
        neo4j_password=None,
        neo4j_database=None,
    )
    try:
        with open(args.graph, "rb") as handle:
            builder.G = pickle.load(handle)
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        print("Loading saved HNSW index, cross-encoder, and generator tokenizer...")
        cross_encoder = CrossEncoder(args.cross_encoder)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        packer = TokenBudgetPacker(tokenizer)
        retriever = PassageRetriever(builder, cross_encoder=cross_encoder)
        print(f"Loaded {len(retriever.passages)} source passages and {len(questions)} reviewed questions.")

        tuning = None
        if args.mode == "tune":
            print(f"Tuning on {len(dev_questions)} dev questions; held-out rows are not evaluated.")
            tuning = tune_on_dev(questions, builder, retriever, args.hnsw_index)
            Path(args.selected_config).parent.mkdir(parents=True, exist_ok=True)
            Path(args.selected_config).write_text(
                json.dumps({
                    "question_file_sha256": sha256(args.questions),
                    "source_sha256": sha256(args.source),
                    "graph_sha256": sha256(args.graph),
                    "hnsw_index_sha256": sha256(args.hnsw_index),
                    **tuning,
                }, indent=2),
                encoding="utf-8",
            )
            selected_config = tuning["selected_config"]
            eval_questions = dev_questions
        else:
            selected = json.loads(Path(args.selected_config).read_text(encoding="utf-8"))
            for key, path in (
                ("question_file_sha256", args.questions),
                ("source_sha256", args.source),
                ("graph_sha256", args.graph),
                ("hnsw_index_sha256", args.hnsw_index),
            ):
                if selected.get(key) != sha256(path):
                    raise ValueError(
                        f"Selected config was tuned against a different {key.removesuffix('_sha256').replace('_', ' ')}."
                    )
            selected_config = selected["selected_config"]
            eval_questions = heldout_questions
            print(f"Final evaluation on {len(heldout_questions)} held-out questions; no tuning is performed.")

        if args.mode == "final":
            for index, question in enumerate(eval_questions, start=1):
                retriever.cross_scores(question["question"])
                print(f"Held-out cross-encoder precompute {index}/{len(eval_questions)}: {question['id']}")

        method_rows = []
        for index, question in enumerate(eval_questions, start=1):
            config = dict(selected_config)
            config.setdefault("cross_encoder_candidate_k", 40)
            rows = build_rankings(
                question["question"], builder, retriever, config, args.hnsw_index
            )
            method_rows.append(question_result(
                question, rows, retriever, packer, args.context_token_budget
            ))
            print(f"Scored {index}/{len(eval_questions)}: {question['id']}")

        methods = list(method_rows[0]["methods"]) if method_rows else []
        summary = mean_summary(
            method_rows, methods, packer, retriever, args.context_token_budget,
            bootstrap_iterations=args.bootstrap_iterations,
        )
        category_summaries = {}
        for category in sorted({row.get("category") for row in method_rows if row.get("category")}):
            category_rows = [row for row in method_rows if row.get("category") == category]
            category_summaries[category] = mean_summary(
                category_rows, methods, packer, retriever, args.context_token_budget,
                bootstrap_iterations=args.bootstrap_iterations,
            )
        output = {
            "run_metadata": {
                "mode": args.mode,
                "source_path": str(Path(args.source).resolve().relative_to(ROOT)),
                "source_sha256": sha256(args.source),
                "graph_path": str(Path(args.graph).resolve().relative_to(ROOT)),
                "graph_sha256": sha256(args.graph),
                "hnsw_index_path": str(Path(args.hnsw_index).resolve().relative_to(ROOT)),
                "hnsw_index_sha256": sha256(args.hnsw_index),
                "question_path": str(Path(args.questions).resolve().relative_to(ROOT)),
                "question_sha256": sha256(args.questions),
                "question_count": len(eval_questions),
                "development_question_count": len(dev_questions),
                "heldout_question_count": len(heldout_questions),
                "question_quality": "Assistant-authored and reviewed; evidence quotes and offsets match source text exactly; not externally human-annotated.",
                "passage_count": len(retriever.passages),
                "embedding_model": EMBEDDING_MODEL,
                "cross_encoder": args.cross_encoder,
                "tokenizer": args.tokenizer,
                "context_token_budget": args.context_token_budget,
                "selected_config": selected_config,
                "graph_rebuilt": False,
                "api_calls": 0,
                "metric_definition": {
                    "evidence_span_recall_at_k": "Fraction of annotated source quotes fully covered by the union of the top-k retrieved T-passage source intervals.",
                    "mrr": "Reciprocal rank of the first T passage overlapping any annotated evidence quote.",
                    "ndcg_at_10": "Graded passage relevance equals the number of annotated evidence quotes overlapped by that passage.",
                    "complete_evidence_at_10": "Fraction of questions whose every annotated evidence quote is covered by the top-10 passages.",
                },
            },
            "tuning": tuning,
            "summary": summary,
            "category_summaries": category_summaries,
            "questions": method_rows,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\nRetrieval results")
        for method, metrics in summary.items():
            print(
                f"{method:32s} R@10={metrics['evidence_span_recall@10']:.3f} "
                f"MRR={metrics['mrr']:.3f} nDCG@10={metrics['ndcg@10']:.3f} "
                f"packed-span={metrics['packed_context_span_recall']:.3f} "
                f"vs-BM25={metrics.get('delta_vs_bm25', {}).get('ndcg@10', 0):+.3f}"
            )
        print(f"\nSaved: {output_path}")
        if tuning:
            print(f"Selected dev config: {json.dumps(tuning['selected_config'])}")
            print(f"Saved selected config: {args.selected_config}")
    finally:
        builder.close()


if __name__ == "__main__":
    main()
