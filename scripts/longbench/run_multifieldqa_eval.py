"""Evaluate passage retrieval and answer generation on LongBench MultiFieldQA-en.

Retrieval ranking is run within each supplied article.  The graph condition
uses this project's HNSW -> PPR -> cross-encoder passage path over a temporary
T-only graph per article; it does not claim to evaluate extracted entity or
relation nodes, Leiden communities, or summaries.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import random
import re
import statistics
import string
import sys
import tempfile
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

from modules.chunker import Chunker  # noqa: E402
from modules.graph_builder import GraphBuilder  # noqa: E402
from modules.groq_handler import GroqHandler  # noqa: E402
from modules.retrieval import PassageRetriever, TokenBudgetPacker  # noqa: E402


DATA_PATH = ROOT / "data" / "longbench" / "multifieldqa_en.jsonl"
RESULT_DIR = ROOT / "results"
RETRIEVAL_PATH = RESULT_DIR / "longbench-multifieldqa-en-retrieval.json"
GENERATION_PATH = RESULT_DIR / "longbench-multifieldqa-en-generation.json"
REPORT_PATH = RESULT_DIR / "longbench-multifieldqa-en-report.md"
DATA_URL = "https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip"
ENGLISH_ARTICLES = {"a", "an", "the"}
METHODS = ("bm25", "dense_exact", "hybrid_rrf", "hnsw_ppr", "hnsw_ppr_crossencoder")
GENERATE_METHODS = ("bm25", "hnsw_ppr_crossencoder")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_dataset(path: Path = DATA_PATH) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading the official THUDM/LongBench data archive...", flush=True)
    with tempfile.TemporaryDirectory(prefix="longbench-") as temp_dir:
        archive_path = Path(temp_dir) / "data.zip"
        urllib.request.urlretrieve(DATA_URL, archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            target = next(
                name for name in archive.namelist()
                if name.endswith("/multifieldqa_en.jsonl") or name == "multifieldqa_en.jsonl"
            )
            with archive.open(target) as source, path.open("wb") as destination:
                while True:
                    block = source.read(1024 * 1024)
                    if not block:
                        break
                    destination.write(block)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_answer(text: str) -> str:
    value = str(text).lower()
    value = "".join(character for character in value if character not in string.punctuation)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def answer_f1(prediction: str, references: list[str]) -> tuple[float, float]:
    """LongBench QA F1, plus normalized exact match; both take best reference."""
    prediction_tokens = normalize_answer(prediction).split()
    best_f1 = 0.0
    best_em = 0.0
    for reference in references:
        reference_tokens = normalize_answer(reference).split()
        if not reference_tokens:
            continue
        overlap = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
        precision = overlap / len(prediction_tokens) if prediction_tokens else 0.0
        recall = overlap / len(reference_tokens)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        best_f1 = max(best_f1, f1)
        best_em = max(best_em, float(prediction_tokens == reference_tokens))
    return best_f1, best_em


def bootstrap_mean_ci(values: list[float], seed: int = 42, samples: int = 10000) -> list[float] | None:
    """Return a deterministic percentile 95% CI for a paired mean difference."""
    if not values:
        return None
    rng = random.Random(seed)
    size = len(values)
    means = sorted(
        sum(values[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(samples)
    )
    return [means[int(0.025 * samples)], means[min(samples - 1, int(0.975 * samples))]]


def source_answer_spans(context: str, answers: list[str]) -> list[tuple[int, int]]:
    """Find exact normalized answer-token occurrences, retaining source offsets.

    These are answer-string anchors, not dataset-provided evidence annotations.
    """
    source_tokens = [
        (match.group(0).casefold(), match.start(), match.end())
        for match in re.finditer(r"\w+", context, flags=re.UNICODE)
        if match.group(0).casefold() not in ENGLISH_ARTICLES
    ]
    spans: set[tuple[int, int]] = set()
    for answer in answers:
        answer_tokens = [
            token.casefold() for token in re.findall(r"\w+", str(answer), flags=re.UNICODE)
            if token.casefold() not in ENGLISH_ARTICLES
        ]
        if not answer_tokens or len(answer_tokens) > len(source_tokens):
            continue
        for index in range(len(source_tokens) - len(answer_tokens) + 1):
            if [item[0] for item in source_tokens[index:index + len(answer_tokens)]] == answer_tokens:
                spans.add((source_tokens[index][1], source_tokens[index + len(answer_tokens) - 1][2]))
    return sorted(spans)


def ranked_ids(rows: list[tuple[str, float]]) -> list[str]:
    return [str(node_id) for node_id, _ in rows]


def retrieval_metrics(rows: list[dict], method: str) -> dict:
    eligible = [row for row in rows if row["answer_anchor_passage_ids"]]
    if not eligible:
        return {"eligible_questions": 0}
    output: dict = {"eligible_questions": len(eligible)}
    for cutoff in (1, 5, 10):
        output[f"answer_anchor_hit@{cutoff}"] = sum(
            bool(set(row["rankings"][method][:cutoff]) & set(row["answer_anchor_passage_ids"]))
            for row in eligible
        ) / len(eligible)
    reciprocal_ranks = []
    ndcgs = []
    packed_recall = []
    for row in eligible:
        relevant = set(row["answer_anchor_passage_ids"])
        ranked = row["rankings"][method]
        first_rank = next((rank for rank, node_id in enumerate(ranked, 1) if node_id in relevant), None)
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)
        dcg = sum(
            (1.0 / math.log2(rank + 1))
            for rank, node_id in enumerate(ranked[:10], 1)
            if node_id in relevant
        )
        ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), 10) + 1))
        ndcgs.append(dcg / ideal if ideal else 0.0)
        packed = set(row["packed_passage_ids"].get(method, []))
        packed_recall.append(float(bool(packed & relevant)))
    output["answer_anchor_MRR"] = sum(reciprocal_ranks) / len(reciprocal_ranks)
    output["answer_anchor_nDCG@10"] = sum(ndcgs) / len(ndcgs)
    output["packed_answer_anchor_recall"] = sum(packed_recall) / len(packed_recall)
    return output


def make_context_graph(
    chunks: list[dict], source_id: str, embedder, index_path: Path,
) -> tuple[GraphBuilder, PassageRetriever]:
    builder = GraphBuilder(None, "", None, None, None, None)
    builder._embedding_model = embedder
    for chunk in chunks:
        node_id = f"T_{chunk['id']}"
        builder.G.add_node(
            node_id,
            type="T",
            text=chunk["text"],
            source_chunk_id=chunk["id"],
            source_doc_id=source_id,
            page_start=None,
            page_end=None,
            source_pages=[],
            char_start=chunk["char_start"],
            char_end=chunk["char_end"],
            community=source_id,
        )
    with contextlib.redirect_stdout(io.StringIO()):
        builder.build_G5_semantic_hnsw_index(index_path=str(index_path))
    return builder, PassageRetriever(builder, embedding_model=embedder)


def load_tokenizer():
    from transformers import AutoTokenizer

    configured = (os.getenv("RAG_CONTEXT_TOKENIZER") or "openai/gpt-oss-20b").strip()
    try:
        tokenizer = AutoTokenizer.from_pretrained(configured)
        return tokenizer, configured
    except Exception as error:
        fallback = "gpt2"
        print(f"Tokenizer {configured!r} unavailable ({type(error).__name__}); using {fallback} for packing.")
        return AutoTokenizer.from_pretrained(fallback), fallback


def create_retrieval(args: argparse.Namespace) -> dict:
    ensure_dataset(Path(args.data))
    dataset_path = Path(args.data)
    records = read_jsonl(dataset_path)
    if args.limit:
        records = records[:args.limit]

    from sentence_transformers import CrossEncoder, SentenceTransformer

    print("Loading all-MiniLM-L12-v2 and the cross-encoder...", flush=True)
    embedder = SentenceTransformer("all-MiniLM-L12-v2")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    chunker = Chunker(store_file=str(ROOT / "data" / "longbench" / "unused.jsonl"))
    tokenizer, tokenizer_name = load_tokenizer()
    packer = TokenBudgetPacker(tokenizer)
    index_dir = ROOT / "data" / "longbench" / "hnsw"
    index_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in records:
        doc_hash = hashlib.sha256(item["context"].encode("utf-8")).hexdigest()[:20]
        grouped[doc_hash].append(item)

    output_rows = []
    for doc_number, (doc_hash, items) in enumerate(grouped.items(), start=1):
        context = items[0]["context"]
        chunks = chunker.segment_into_chunks(
            context,
            chunk_size=args.chunk_words,
            overlap=args.overlap_words,
            source_id=doc_hash,
            persist=False,
        )
        if not chunks:
            raise ValueError(f"No chunks generated for context {doc_hash}.")
        index_path = index_dir / f"{doc_hash}.bin"
        builder, retriever = make_context_graph(chunks, doc_hash, embedder, index_path)
        retriever.cross_encoder = cross_encoder
        for item in items:
            query = item["input"]
            rankings: dict[str, list[tuple[str, float]]] = {
                "bm25": retriever.bm25(query),
                "dense_exact": retriever.dense(query),
            }
            rankings["hybrid_rrf"] = retriever.rrf(rankings["bm25"], rankings["dense_exact"])
            with contextlib.redirect_stdout(io.StringIO()):
                hnsw_ppr = retriever.hnsw_graph_expand(
                    query,
                    hnsw_index_path=str(index_path),
                    hnsw_candidate_k=args.hnsw_candidates,
                    expand_k=args.ppr_expand,
                    alpha=args.ppr_alpha,
                    max_iter=args.ppr_iterations,
                )
            rankings["hnsw_ppr"] = hnsw_ppr
            rankings["hnsw_ppr_crossencoder"] = retriever.cross_rerank(
                query, hnsw_ppr[:args.rerank_candidates]
            )

            anchor_spans = source_answer_spans(context, item.get("answers", []))
            anchor_ids = sorted({
                f"T_{chunk['id']}"
                for chunk in chunks
                if any(chunk["char_start"] <= start and chunk["char_end"] >= end for start, end in anchor_spans)
            })
            contexts = {}
            packed_ids = {}
            packed_tokens = {}
            for method, ranking in rankings.items():
                records_for_packer = [retriever.passage_context_record(node_id, score) for node_id, score in ranking]
                packed = packer.pack(query, records_for_packer, token_budget=args.context_tokens)
                contexts[method] = packed["context"]
                packed_ids[method] = packed["passage_ids"]
                packed_tokens[method] = packed["token_count"]
            with contextlib.redirect_stdout(io.StringIO()):
                entry_nodes, _ = builder.get_entry_points(
                    query, hnsw_index_path=str(index_path), top_k=args.hnsw_candidates
                )
            output_rows.append({
                "id": item["_id"],
                "question": query,
                "answers": item.get("answers", []),
                "length_words": len(context.split()),
                "context_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
                "passage_count": len(chunks),
                "anchor_spans": anchor_spans,
                "answer_anchor_passage_ids": anchor_ids,
                "rankings": {name: ranked_ids(rows) for name, rows in rankings.items()},
                "packed_contexts": contexts,
                "packed_passage_ids": packed_ids,
                "packed_tokens": packed_tokens,
                "candidate_counts": {
                    "hnsw_entry_nodes_union": len(entry_nodes),
                    "hnsw_ppr_passages_before_rerank": len(hnsw_ppr),
                    "cross_encoder_scored": min(len(hnsw_ppr), args.rerank_candidates),
                    "cross_encoder_returned": len(rankings["hnsw_ppr_crossencoder"]),
                },
            })
        print(f"Indexed context {doc_number}/{len(grouped)}: {len(chunks)} passages, {len(items)} questions", flush=True)

    metric_rows = [
        {
            "rankings": row["rankings"],
            "packed_passage_ids": row["packed_passage_ids"],
            "answer_anchor_passage_ids": row["answer_anchor_passage_ids"],
        }
        for row in output_rows
    ]
    methods = {method: retrieval_metrics(metric_rows, method) for method in METHODS}
    anchor_count = sum(bool(row["answer_anchor_passage_ids"]) for row in output_rows)
    result = {
        "run_metadata": {
            "benchmark": "LongBench MultiFieldQA-en",
            "dataset_path": str(dataset_path.resolve().relative_to(ROOT)),
            "dataset_sha256": sha256_file(dataset_path),
            "record_count": len(output_rows),
            "distinct_context_count": len(grouped),
            "chunker": "project Chunker, sentence-based, 400 words with 40-word overlap unless overridden",
            "passage_count_total_over_documents": sum(
                len(chunker.segment_into_chunks(item["context"], chunk_size=args.chunk_words, overlap=args.overlap_words, source_id=hashlib.sha256(item["context"].encode()).hexdigest()[:20], persist=False))
                for _, items in grouped.items() for item in items[:1]
            ),
            "embedding_model": "all-MiniLM-L12-v2",
            "hnsw_space": "cosine",
            "hnsw_candidates": args.hnsw_candidates,
            "lexical_entry_candidates": 8,
            "ppr_expand_k": args.ppr_expand,
            "ppr_alpha": args.ppr_alpha,
            "ppr_iterations": args.ppr_iterations,
            "rerank_candidates": args.rerank_candidates,
            "cross_encoder": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "context_token_budget": args.context_tokens,
            "tokenizer": tokenizer_name,
            "graph_scope": "Temporary T-only passage graph with semantic HNSW edges per article; no extracted entity/relation/summary nodes or Leiden communities.",
            "retrieval_label_limit": "LongBench provides answers but no gold evidence spans. Answer-anchor metrics include only examples where a normalized reference answer string occurs contiguously in the source context.",
            "answer_anchor_questions": anchor_count,
        },
        "retrieval_metrics_answer_anchor_subset": methods,
        "questions": output_rows,
    }
    output_path = Path(args.retrieval_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved retrieval run: {output_path}", flush=True)
    return result


def generation_prompt(question: str, context: str) -> str:
    return f"""Answer the question using only the evidence in the supplied context. Give a concise, direct answer in at most 100 words. Do not add citations or explain your reasoning. If the context does not contain enough information, say you do not know.

Question: {question}

Context:
{context}

Answer:"""


def run_generation(args: argparse.Namespace, retrieval: dict) -> dict:
    from modules.generator import GraphAugmentedGenerator

    prior: dict[tuple[str, str], dict] = {}
    output_path = Path(args.generation_output)
    if output_path.exists():
        try:
            old = json.loads(output_path.read_text(encoding="utf-8"))
            if old.get("run_metadata", {}).get("retrieval_sha256") == hashlib.sha256(
                Path(args.retrieval_output).read_bytes()
            ).hexdigest():
                prior = {
                    (row["id"], row["method"]): row
                    for row in old.get("predictions", [])
                    if row.get("answer")
                }
        except Exception:
            prior = {}

    handler = GroqHandler()
    generator = GraphAugmentedGenerator(handler)
    predictions = list(prior.values())
    total = len(retrieval["questions"]) * len(args.generation_methods)
    for method in args.generation_methods:
        for index, row in enumerate(retrieval["questions"], start=1):
            key = (row["id"], method)
            if key in prior:
                continue
            prompt = generation_prompt(row["question"], row["packed_contexts"][method])
            try:
                answer = handler.generate_answer(prompt)
                error = None
                if not answer or answer == "rate_limit_exceeded":
                    error = handler.last_error_type or "generation_failed"
                    answer = ""
            except Exception as exc:
                answer = ""
                error = type(exc).__name__
            f1, em = answer_f1(answer, row["answers"])
            prediction = {
                "id": row["id"],
                "method": method,
                "question": row["question"],
                "answer": answer,
                "references": row["answers"],
                "longbench_qa_f1": f1,
                "normalized_exact_match": em,
                "packed_context_tokens": row["packed_tokens"][method],
                "retrieved_passage_ids": row["rankings"][method],
                "packed_passage_ids": row["packed_passage_ids"][method],
                "provider_model": handler.last_model,
                "provider_key_slot": handler.last_key_slot,
                "generation_error": error,
            }
            predictions.append(prediction)
            print(
                f"Generated {len(predictions)}/{total}: {method} / {row['id']} "
                f"F1={f1:.3f}" + (f" error={error}" if error else ""),
                flush=True,
            )
            checkpoint = summarize_generation(predictions, retrieval, handler, args)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8")

    final = summarize_generation(predictions, retrieval, handler, args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    return final


def summarize_generation(predictions, retrieval, handler, args) -> dict:
    grouped = {}
    successful_rows = [row for row in predictions if row.get("answer") and not row.get("generation_error")]
    provider_breakdown = {}
    for method in args.generation_methods:
        successes = [row for row in predictions if row["method"] == method and not row.get("generation_error")]
        grouped[method] = {
            "attempted": sum(row["method"] == method for row in predictions),
            "successful": len(successes),
            "longbench_qa_f1_percent": round(100 * sum(row["longbench_qa_f1"] for row in successes) / len(successes), 2) if successes else None,
            "normalized_exact_match_percent": round(100 * sum(row["normalized_exact_match"] for row in successes) / len(successes), 2) if successes else None,
        }
    for model in sorted({row.get("provider_model") or "unknown" for row in successful_rows}):
        provider_breakdown[model] = {}
        for method in args.generation_methods:
            rows = [row for row in successful_rows if row["method"] == method and (row.get("provider_model") or "unknown") == model]
            provider_breakdown[model][method] = {
                "count": len(rows),
                "longbench_qa_f1_percent": round(100 * sum(row["longbench_qa_f1"] for row in rows) / len(rows), 2) if rows else None,
            }
    by_method_and_id = {
        (row["method"], row["id"]): row
        for row in successful_rows
    }
    paired = []
    for row in successful_rows:
        if row["method"] != "bm25":
            continue
        graph_row = by_method_and_id.get(("hnsw_ppr_crossencoder", row["id"]))
        if graph_row is not None:
            paired.append({
                "id": row["id"],
                "same_provider_model": row.get("provider_model") == graph_row.get("provider_model"),
                "bm25_f1": row["longbench_qa_f1"],
                "graph_f1": graph_row["longbench_qa_f1"],
                "delta": graph_row["longbench_qa_f1"] - row["longbench_qa_f1"],
            })
    same_model_pairs = [row for row in paired if row["same_provider_model"]]
    return {
        "run_metadata": {
            "benchmark": "LongBench MultiFieldQA-en",
            "mode": "generation on frozen retrieved contexts",
            "retrieval_path": str(Path(args.retrieval_output).resolve().relative_to(ROOT)),
            "retrieval_sha256": hashlib.sha256(Path(args.retrieval_output).read_bytes()).hexdigest(),
            "answer_metric": "Official LongBench qa_f1_score normalization; best reference, reported on 0-100 scale.",
            "prompt": "LongBench answer-only prompt; no generated source citation requested so citation formatting does not alter QA F1.",
            "context_token_budget": retrieval["run_metadata"]["context_token_budget"],
            "configured_generation_methods": list(args.generation_methods),
            "provider": handler.get_runtime_metadata(),
        },
        "summary": grouped,
        "provider_model_breakdown": provider_breakdown,
        "paired_generation_comparison": {
            "questions_with_both_answers": len(paired),
            "same_provider_model_pairs": len(same_model_pairs),
            "mean_graph_minus_bm25_f1_all_pairs": round(sum(row["delta"] for row in paired) / len(paired), 4) if paired else None,
            "all_pairs_bootstrap_95_ci": bootstrap_mean_ci([row["delta"] for row in paired]),
            "mean_graph_minus_bm25_f1_same_model_pairs": round(sum(row["delta"] for row in same_model_pairs) / len(same_model_pairs), 4) if same_model_pairs else None,
            "same_model_pairs_bootstrap_95_ci": bootstrap_mean_ci([row["delta"] for row in same_model_pairs]),
        },
        "predictions": predictions,
    }


def write_report(retrieval: dict, generation: dict | None, report_path: Path) -> None:
    metadata = retrieval["run_metadata"]
    question_rows = retrieval["questions"]
    avg_context_words = statistics.mean(row["length_words"] for row in question_rows)
    avg_entry_nodes = statistics.mean(row["candidate_counts"]["hnsw_entry_nodes_union"] for row in question_rows)
    avg_reranked = statistics.mean(row["candidate_counts"]["cross_encoder_scored"] for row in question_rows)
    max_reranked = max(row["candidate_counts"]["cross_encoder_scored"] for row in question_rows)
    mean_packed_tokens = {
        method: statistics.mean(row["packed_tokens"][method] for row in question_rows)
        for method in ("bm25", "hnsw_ppr_crossencoder")
    }
    lines = [
        "# LongBench MultiFieldQA-en Evaluation",
        "",
        f"Dataset: [official THUDM/LongBench MultiFieldQA-en](https://huggingface.co/datasets/THUDM/LongBench), locally stored at `data/longbench/multifieldqa_en.jsonl`.",
        "",
        "## Data and protocol",
        "",
        f"- Questions evaluated: **{metadata['record_count']}** over **{metadata['distinct_context_count']}** distinct article contexts.",
        f"- Context length: mean **{avg_context_words:.0f} words** per question context; passage corpus totals **{metadata['passage_count_total_over_documents']:,}** chunks across distinct contexts.",
        "- Question categories: all 150 rows are English, but the released record schema has no per-question category label, so no category-wise scores are claimed; questions are drawn from articles in diverse fields.",
        f"- Passage chunking: project sentence chunker, {metadata['chunker']}.",
        f"- Retrieval methods: BM25, exact dense cosine, reciprocal rank fusion, HNSW→PPR, and HNSW→PPR→cross-encoder.",
        f"- Reranking: {metadata['cross_encoder']}; up to {metadata['rerank_candidates']} candidate passages scored.",
        f"- Context packing budget: {metadata['context_token_budget']} tokens using `{metadata['tokenizer']}`.",
        f"- Observed retrieval sizes: mean HNSW/lexical entry union **{avg_entry_nodes:.1f}** passages; cross-encoder scored **{avg_reranked:.1f}** per question on average (maximum {max_reranked}); mean packed tokens were **{mean_packed_tokens['bm25']:.0f}** for BM25 and **{mean_packed_tokens['hnsw_ppr_crossencoder']:.0f}** for the project ranking.",
        f"- Graph coverage: {metadata['graph_scope']}",
        "- Split: the official test set is evaluated as published. No hyperparameter tuning was performed on it.",
        "- Correct answer: LongBench normalized token F1 against the best supplied answer reference. Exact match is also reported as a secondary metric.",
        "- Retrieval evidence limitation: this split supplies answers but no gold evidence spans. Recall/MRR/nDCG are therefore not official gold-evidence metrics here.",
        f"- Answer-anchor subset: **{metadata['answer_anchor_questions']}/{metadata['record_count']}** questions have at least one normalized reference answer string appearing contiguously in the source; retrieval figures below are explicitly a weak diagnostic on that subset.",
        "",
        "## Retrieval results (answer-anchor diagnostic subset)",
        "",
        "| Method | n | Hit@1 | Hit@5 | Hit@10 | MRR | nDCG@10 | Packed answer-anchor recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        values = retrieval["retrieval_metrics_answer_anchor_subset"][method]
        if not values.get("eligible_questions"):
            lines.append(f"| {method} | 0 | — | — | — | — | — | — |")
            continue
        fmt = lambda key: f"{values.get(key, 0):.3f}"
        lines.append(
            f"| {method} | {values['eligible_questions']} | {fmt('answer_anchor_hit@1')} | "
            f"{fmt('answer_anchor_hit@5')} | {fmt('answer_anchor_hit@10')} | "
            f"{fmt('answer_anchor_MRR')} | {fmt('answer_anchor_nDCG@10')} | "
            f"{fmt('packed_answer_anchor_recall')} |"
        )
    lines.extend([
        "",
        "The answer-anchor subset is selected by whether answer text is recoverable verbatim after normalization. It is biased toward extractive answers, can miss paraphrased evidence, and can count an answer mention that is not the actual reasoning evidence. Do not present these figures as dataset gold Recall@k/MRR/nDCG.",
        "",
        "## Generation results",
        "",
    ])
    if generation:
        lines.extend([
            "Generation is scored separately on the frozen packed contexts. BM25 is the controlled retrieval baseline; the project condition uses HNSW→PPR→cross-encoder. Both use the same answer-only prompt and provider fallback configuration.",
            "",
            "| Retrieved context | Completed | LongBench QA F1 | Normalized EM |",
            "|---|---:|---:|---:|",
        ])
        for method, values in generation["summary"].items():
            f1 = "—" if values["longbench_qa_f1_percent"] is None else f"{values['longbench_qa_f1_percent']:.2f}"
            em = "—" if values["normalized_exact_match_percent"] is None else f"{values['normalized_exact_match_percent']:.2f}"
            lines.append(f"| {method} | {values['successful']}/{values['attempted']} | {f1} | {em} |")
        comparison = generation.get("paired_generation_comparison", {})
        if comparison:
            all_ci = comparison.get("all_pairs_bootstrap_95_ci")
            same_ci = comparison.get("same_model_pairs_bootstrap_95_ci")
            all_ci_text = f" [{100 * all_ci[0]:.2f}, {100 * all_ci[1]:.2f}]" if all_ci else ""
            same_ci_text = f" [{100 * same_ci[0]:.2f}, {100 * same_ci[1]:.2f}]" if same_ci else ""
            lines.extend([
                "",
                f"Paired F1 difference (graph minus BM25) over questions answered by both: **{100 * comparison['mean_graph_minus_bm25_f1_all_pairs']:.2f} percentage points**{all_ci_text} across {comparison.get('questions_with_both_answers')} pairs; among pairs answered by the same provider model: **{100 * comparison['mean_graph_minus_bm25_f1_same_model_pairs']:.2f} percentage points**{same_ci_text} across {comparison.get('same_provider_model_pairs')} pairs. Brackets show deterministic 95% percentile bootstrap intervals.",
                "",
                "Provider/model assignment was subject to live rate limits and the configured key/model fallback sequence. This is not a fixed-model controlled generation comparison; use the same-model paired subset as a secondary diagnostic and interpret both aggregate scores cautiously.",
                "",
                "| Provider model | Retrieval context | n | LongBench QA F1 |",
                "|---|---|---:|---:|",
            ])
            for model, methods in generation.get("provider_model_breakdown", {}).items():
                for method, stats in methods.items():
                    score = "—" if stats["longbench_qa_f1_percent"] is None else f"{stats['longbench_qa_f1_percent']:.2f}"
                    lines.append(f"| {model} | {method} | {stats['count']} | {score} |")
        lines.extend([
            "",
            "Failed provider calls are excluded from the F1 denominator and counted in the completed column. Per-question predictions and model/key-slot metadata are in the JSON result; secret values are not saved.",
        ])
    else:
        lines.append("Generation stage has not been run yet.")
    lines.extend([
        "",
        "## Interpretation and limits",
        "",
        "- The test set is small (150 questions) and contains only single-document questions. Results are diagnostic, not a broad significance claim.",
        "- The official set has no train/dev/test partition for this task; avoid tuning on it and then reporting the same score as held-out generalization.",
        "- The temporary benchmark graph contains only source passage nodes and semantic HNSW links. It does not test the project's LLM-extracted entities/relations, Leiden communities, or summary hierarchy.",
        "- The fixed 4,096-token context limit makes this a retrieval/compression comparison; it is not LongBench's full-context model score.",
        "- LongBench defines MultiFieldQA-en as single-document QA and uses normalized QA F1. See the [official task statistics](https://github.com/THUDM/LongBench/blob/main/LongBench/task.md) and [official evaluator](https://github.com/THUDM/LongBench/blob/main/LongBench/eval.py).",
        "",
        "## Artifacts",
        "",
        "- `data/longbench/multifieldqa_en.jsonl` — downloaded official test data.",
        "- `scripts/longbench/run_multifieldqa_eval.py` — repeatable retrieval and generation runner.",
        "- `results/longbench-multifieldqa-en-retrieval.json` — frozen rankings, packed prompts, metrics, and run metadata.",
        "- `results/longbench-multifieldqa-en-generation.json` — per-answer output, scores, provider metadata, and resumable checkpoints.",
        "",
        "Re-run with `python scripts/longbench/run_multifieldqa_eval.py --stage all`. Use `--stage retrieval` to rerun retrieval only or `--stage generation` to reuse the frozen retrieval artifact.",
        "",
    ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("retrieval", "generation", "all"), default="all")
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--retrieval-output", default=str(RETRIEVAL_PATH))
    parser.add_argument("--generation-output", default=str(GENERATION_PATH))
    parser.add_argument("--report", default=str(REPORT_PATH))
    parser.add_argument("--limit", type=int, default=None, help="Optional leading sample limit for a pilot run.")
    parser.add_argument("--chunk-words", type=int, default=400)
    parser.add_argument("--overlap-words", type=int, default=40)
    parser.add_argument("--hnsw-candidates", type=int, default=20)
    parser.add_argument("--ppr-expand", type=int, default=20)
    parser.add_argument("--ppr-alpha", type=float, default=0.5)
    parser.add_argument("--ppr-iterations", type=int, default=10)
    parser.add_argument("--rerank-candidates", type=int, default=40)
    parser.add_argument("--context-tokens", type=int, default=4096)
    parser.add_argument("--generation-methods", nargs="+", choices=GENERATE_METHODS, default=list(GENERATE_METHODS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage in {"retrieval", "all"}:
        retrieval = create_retrieval(args)
    else:
        retrieval_path = Path(args.retrieval_output)
        retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))
        if args.limit:
            retrieval["questions"] = retrieval["questions"][:args.limit]
    generation = None
    if args.stage in {"generation", "all"}:
        generation = run_generation(args, retrieval)
    write_report(retrieval, generation, Path(args.report))
    print(f"Wrote report: {args.report}", flush=True)


if __name__ == "__main__":
    main()
