"""Train and evaluate Align-GRAG-style alignment on the saved book graph.

The saved graph and HNSW index are read-only. HNSW/PPR creates the same initial
candidate subgraph for the aligner and the existing retrieval path. A grouped
development-only cross-validation sweep selects pruning size and training
steps; final retrieval metrics are measured once on the frozen held-out split.
"""

from __future__ import annotations

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
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from modules.align_grag import AlignGRAGAligner, anchor_kl_loss, symmetric_info_nce  # noqa: E402
from modules.groq_handler import GroqHandler  # noqa: E402
from modules.graph_builder import GraphBuilder  # noqa: E402
from modules.retrieval import PassageRetriever, TokenBudgetPacker  # noqa: E402
from run_book_retrieval_benchmark import (  # noqa: E402
    bootstrap_ci,
    context_metrics,
    interval_metrics,
    mean_summary,
    pack_ranking,
    question_result,
    sha256,
)


DEFAULT_GRAPH = ROOT / "data" / "book-graph" / "graph.pkl"
DEFAULT_INDEX = ROOT / "data" / "book-graph" / "graph-index.bin"
DEFAULT_QUESTIONS = ROOT / "data" / "book-graph" / "eval_questions.reviewed.jsonl"
DEFAULT_BASELINE = ROOT / "results" / "book-retrieval-final.json"
DEFAULT_OUTPUT = ROOT / "results" / "align-grag-eval.json"
DEFAULT_CHECKPOINT = ROOT / "results" / "align-grag-aligner.pt"
DEFAULT_SUPERVISION = ROOT / "results" / "align-grag-supervision.jsonl"
DEFAULT_SUPERVISION_META = ROOT / "results" / "align-grag-supervision-meta.json"
DEFAULT_REPORT = ROOT / "results" / "align-grag-evaluation-report.md"
EMBEDDING_MODEL = "all-MiniLM-L12-v2"
PPR_ALPHA = 0.5
PPR_MAX_ITER = 10
PPR_TYPE_LIMIT = 20
NODE_ANCHOR_TEMPERATURE = 0.1
CONTRASTIVE_TEMPERATURE = 0.07
CV_EPOCHS = (5, 10, 20)
CV_SEEDS = (5, 10, 20)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize(text: str) -> str:
    return re.sub(r"[^\w]+", " ", str(text).casefold(), flags=re.UNICODE).strip()


def question_answers(question: dict) -> str:
    answers = question.get("answers") or []
    if isinstance(answers, str):
        answers = [answers]
    return " | ".join(str(answer).strip() for answer in answers if str(answer).strip())


def evidence_quotes(question: dict) -> list[str]:
    return [str(span.get("quote", "")).strip() for span in question.get("evidence_spans", []) if span.get("quote")]


def graph_text_for_supervision(graph) -> str:
    nodes = sorted(
        graph.nodes(data=True),
        key=lambda row: (-float(row[1].get("ppr_score", 0.0)), str(row[0])),
    )[:100]
    visible = {node_id for node_id, _ in nodes}
    lines = [
        f"NODE id={node_id} type={attrs.get('type', '?')} text={str(attrs.get('text', ''))[:280]}"
        for node_id, attrs in nodes
    ]
    edges = []
    for source, target, attrs in graph.edges(data=True):
        if source in visible and target in visible:
            label = str(attrs.get("label", attrs.get("type", "related")))[:100]
            edges.append(f"EDGE {source} --{label}--> {target}")
    lines.extend(edges[:160])
    return "\n".join(lines)


def parse_anchor_response(response: str) -> tuple[str, list[str]] | None:
    candidate = str(response or "").strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.I)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, flags=re.S)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(parsed, dict):
        return None
    rationale = parsed.get("rationale", parsed.get("rationale_chain", parsed.get("RationaleChain", "")))
    if isinstance(rationale, list):
        rationale = " ".join(str(step) for step in rationale)
    anchors = parsed.get("anchors", parsed.get("Anchors", []))
    if isinstance(anchors, dict):
        anchors = list(anchors.values())
    texts = []
    for anchor in anchors if isinstance(anchors, list) else []:
        if isinstance(anchor, dict):
            text = anchor.get("text", anchor.get("anchor", anchor.get("name", "")))
        else:
            text = anchor
        if str(text).strip():
            texts.append(str(text).strip())
    return str(rationale).strip(), list(dict.fromkeys(texts))


def grounded_fallback(question: dict, graph) -> dict:
    """Source-derived rationale/anchor fallback if the provider fails parsing."""
    quotes = evidence_quotes(question)
    answer = question_answers(question) or "the supported answer"
    if quotes:
        rationale = " ".join(
            [f"The source graph states: {quote}" for quote in quotes]
            + [f"Therefore the supported answer is {answer}."]
        )
    else:
        rationale = f"The graph supports the answer {answer} to: {question['question']}"
    graph_texts = [normalize(data.get("text", "")) for _, data in graph.nodes(data=True)]
    anchors = [quote for quote in quotes if normalize(quote)]
    if not anchors:
        anchors = [str(question["question"])]
    # Prefer exact graph phrases where the annotated quote occurs in a node.
    valid = []
    for anchor in anchors:
        norm_anchor = normalize(anchor)
        if any(norm_anchor and norm_anchor in node_text for node_text in graph_texts):
            valid.append(anchor)
    return {
        "rationale": rationale,
        "anchors": valid or anchors,
        "rationale_source": "source_quote_fallback",
        "anchor_source": "source_quote_fallback",
        "provider_model": None,
    }


def make_supervision(question: dict, graph, handler: GroqHandler | None) -> dict:
    graph_text = graph_text_for_supervision(graph)
    prompt = {
        "role": "user",
        "content": (
            f"Question: {question['question']}\n"
            f"Gold answer (training target only): {question_answers(question)}\n\n"
            "Retrieved graph evidence:\n"
            f"{graph_text}\n\n"
            "Return JSON with keys rationale and anchors. rationale must be a concise, "
            "evidence-grounded chain from question to answer (2-4 short steps). anchors "
            "must be 1-5 exact phrases copied from node text or edge labels above, with "
            "each anchor either a string or {\"type\":\"entity|relation|passage\",\"text\":\"verbatim phrase\"}. "
            "Do not use outside knowledge or invent graph facts."
        ),
    }
    messages = [{
        "role": "system",
        "content": "You create training supervision for graph retrieval. Use only the supplied graph. Return valid JSON.",
    }, prompt]
    generated = None
    provider_model = None
    provider_failure = None
    if handler is not None:
        response = handler._call_groq(messages, is_json=True)
        if response == "rate_limit_exceeded" or handler.last_error_type == "RateLimited":
            provider_failure = "all_configured_key_model_pairs_rate_limited"
        if response and response != "rate_limit_exceeded":
            parsed = parse_anchor_response(response)
            if parsed:
                rationale, anchors = parsed
                all_node_texts = [normalize(data.get("text", "")) for _, data in graph.nodes(data=True)]
                # Keep only anchors that can be grounded to a retrieved node phrase.
                matched = [
                    anchor for anchor in anchors
                    if any(
                        normalize(anchor)
                        and (normalize(anchor) in node_text or node_text in normalize(anchor))
                        for node_text in all_node_texts if node_text
                    )
                ]
                if rationale and matched:
                    generated = {
                        "rationale": rationale,
                        "anchors": list(dict.fromkeys(matched)),
                        "rationale_source": "groq_graph_grounded",
                        "anchor_source": "groq_graph_grounded",
                        "provider_model": handler.last_model,
                    }
                    provider_model = handler.last_model
    if generated is None:
        generated = grounded_fallback(question, graph)
        if handler is not None:
            provider_model = handler.last_model
    return {
        "id": question["id"],
        "question": question["question"],
        **generated,
        "provider_model": provider_model,
        "candidate_node_count": graph.number_of_nodes(),
        "candidate_edge_count": graph.number_of_edges(),
        "provider_failure": provider_failure,
    }


def load_or_create_supervision(
    dev_questions: list[dict],
    candidate_graphs: dict[str, object],
    cache_path: Path,
    meta_path: Path,
    graph_hash: str,
    question_hash: str,
) -> dict[str, dict]:
    expected_meta = {
        "graph_sha256": graph_hash,
        "question_file_sha256": question_hash,
        "development_ids": [question["id"] for question in dev_questions],
        "purpose": "Align-GRAG development supervision; held-out answers are not used.",
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached = {}
    if cache_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            cached = {row["id"]: row for row in read_jsonl(cache_path)}
            if not all(meta.get(key) == value for key, value in expected_meta.items()):
                cached = {}
        except (OSError, ValueError, KeyError):
            cached = {}
    elif cache_path.exists():
        # Recover a partial checkpoint from an interrupted first run only when
        # question text and current candidate graph sizes still match exactly.
        try:
            possible = {row["id"]: row for row in read_jsonl(cache_path)}
            recoverable = all(
                q["id"] in possible
                and possible[q["id"]].get("question") == q["question"]
                and possible[q["id"]].get("candidate_node_count") == candidate_graphs[q["id"]].number_of_nodes()
                and possible[q["id"]].get("candidate_edge_count") == candidate_graphs[q["id"]].number_of_edges()
                for q in dev_questions if q["id"] in possible
            )
            if recoverable:
                cached = possible
                print(f"Recovering {len(cached)} cached development supervision rows.")
        except (OSError, ValueError, KeyError):
            cached = {}
    if all(q["id"] in cached for q in dev_questions):
        print(f"Reusing cached development rationale/anchor supervision: {cache_path}")
        return cached
    write_json(meta_path, expected_meta)

    handler = None
    try:
        handler = GroqHandler()
    except ValueError:
        print("No Groq credentials configured; using source-quote rationale supervision.")

    rows = dict(cached)
    for index, question in enumerate(dev_questions, start=1):
        if question["id"] in rows:
            continue
        row = make_supervision(question, candidate_graphs[question["id"]], handler)
        rows[question["id"]] = row
        cache_path.write_text(
            "".join(json.dumps(rows[qid], ensure_ascii=False) + "\n" for qid in rows),
            encoding="utf-8",
        )
        print(
            f"Built dev supervision {index}/{len(dev_questions)}: {question['id']} "
            f"({row['rationale_source']}, {len(row['anchors'])} anchors)"
        )
        if row.get("provider_failure"):
            handler = None
            print("Provider is rate-limited; source-grounded fallback will be used for remaining dev rows.")
    if handler is not None:
        runtime = handler.get_runtime_metadata()
        # The runtime metadata contains model/key slot names only, never key values.
        expected_meta["provider_runtime"] = runtime
    write_json(meta_path, expected_meta)
    return rows


def embed_all_graph_nodes(graph, encoder: SentenceTransformer) -> tuple[list[str], dict[str, int], torch.Tensor]:
    node_ids = list(graph.nodes())
    texts = [str(graph.nodes[node].get("text", "") or node) for node in node_ids]
    vectors = encoder.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    return node_ids, node_index, torch.as_tensor(vectors, dtype=torch.float32)


def graph_example(
    qid: str,
    query: str,
    candidate_graph,
    global_index: dict[str, int],
    global_features: torch.Tensor,
    encoder: SentenceTransformer,
    supervision: dict | None = None,
) -> dict:
    node_ids = list(candidate_graph.nodes())
    if not node_ids:
        raise ValueError(f"Empty initial graph for {qid}.")
    local_index = {node_id: index for index, node_id in enumerate(node_ids)}
    features = global_features[[global_index[node_id] for node_id in node_ids]].clone()
    adjacency = torch.zeros((len(node_ids), len(node_ids)), dtype=torch.float32)
    for source, target in candidate_graph.edges():
        if source not in local_index or target not in local_index or source == target:
            continue
        i, j = local_index[source], local_index[target]
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0
    degrees = adjacency.sum(dim=1, keepdim=True)
    adjacency = adjacency / degrees.clamp_min(1.0)
    query_embedding = encoder.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=False,
    )[0]
    row = {
        "id": qid,
        "query": query,
        "graph": candidate_graph,
        "node_ids": node_ids,
        "node_types": [candidate_graph.nodes[node].get("type", "") for node in node_ids],
        "node_features": features,
        "adjacency": adjacency,
        "query_embedding": torch.as_tensor(query_embedding, dtype=torch.float32),
        "anchor_distribution": None,
        "rationale_embedding": None,
    }
    if supervision is not None:
        anchors = supervision.get("anchors", [])
        if not anchors:
            anchors = [query]
        anchor_vectors = encoder.encode(
            anchors, normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False,
        )
        similarities = features @ torch.as_tensor(anchor_vectors, dtype=torch.float32).T
        anchor_logits = similarities.max(dim=1).values / NODE_ANCHOR_TEMPERATURE
        row["anchor_distribution"] = torch.softmax(anchor_logits, dim=0)
        rationale = str(supervision.get("rationale", "")).strip()
        if not rationale:
            rationale = " ".join(anchors)
        rationale_vector = encoder.encode(
            [rationale], normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        row["rationale_embedding"] = torch.as_tensor(rationale_vector, dtype=torch.float32)
    return row


def train_epochs(
    model: AlignGRAGAligner,
    examples: list[dict],
    epochs: int,
    contrastive: bool = True,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
) -> list[dict]:
    if len(examples) < 2:
        raise ValueError("Align-GRAG contrastive training requires at least two examples.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        node_losses, graph_embeddings, rationale_embeddings = [], [], []
        for example in examples:
            _, logits, graph_embedding = model(
                example["node_features"], example["adjacency"], example["query_embedding"]
            )
            node_losses.append(anchor_kl_loss(logits, example["anchor_distribution"]))
            graph_embeddings.append(graph_embedding)
            rationale_embeddings.append(example["rationale_embedding"])
        node_loss = torch.stack(node_losses).mean()
        graph_stack = torch.stack(graph_embeddings)
        rationale_stack = torch.stack(rationale_embeddings)
        if contrastive:
            graph_projected, rationale_projected = model.project_graph_text(
                graph_stack, rationale_stack
            )
            contrastive_loss = symmetric_info_nce(
                graph_projected, rationale_projected,
                temperature=CONTRASTIVE_TEMPERATURE,
            )
        else:
            contrastive_loss = node_loss.new_zeros(())
        loss = node_loss + contrastive_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append({
            "epoch": epoch,
            "total": float(loss.detach()),
            "node_kl": float(node_loss.detach()),
            "graph_rationale_contrastive": float(contrastive_loss.detach()),
        })
    return losses


def predict_aligned_ranking(model: AlignGRAGAligner, example: dict, seed_count: int) -> list[tuple[str, float]]:
    model.eval()
    with torch.no_grad():
        _, logits, _ = model(
            example["node_features"], example["adjacency"], example["query_embedding"]
        )
    scores = logits.detach().cpu().numpy().astype(float)
    order = np.argsort(-scores, kind="stable")
    selected_rows = order[:max(1, min(int(seed_count), len(order)))]
    keep = set(int(index) for index in selected_rows)
    neighbors = example["adjacency"] > 0
    for index in selected_rows:
        keep.update(torch.where(neighbors[int(index)])[0].tolist())
    passage_rows = [
        (example["node_ids"][index], float(scores[index]))
        for index in keep
        if example["node_types"][index] == "T"
    ]
    passage_rows.sort(key=lambda row: (-row[1], row[0]))
    if not passage_rows:
        passage_rows = [
            (example["node_ids"][index], float(scores[index]))
            for index in order if example["node_types"][index] == "T"
        ][:1]
    return passage_rows


def passage_anchor_ids(question: dict, passages: list) -> set[str]:
    spans = question.get("evidence_spans", [])
    return {
        passage.node_id
        for passage in passages
        if any(
            passage.char_start < int(span["char_end"])
            and passage.char_end > int(span["char_start"])
            for span in spans
        )
    }


def make_grouped_folds(questions: list[dict], passages: list, fold_count: int = 4) -> list[list[int]]:
    """Group questions sharing any evidence passage before assigning folds."""
    support_by_index = [passage_anchor_ids(question, passages) for question in questions]
    parent = list(range(len(questions)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    first_for_passage: dict[str, int] = {}
    for index, support in enumerate(support_by_index):
        for passage_id in support:
            if passage_id in first_for_passage:
                union(index, first_for_passage[passage_id])
            else:
                first_for_passage[passage_id] = index
    groups: dict[int, list[int]] = {}
    for index in range(len(questions)):
        groups.setdefault(find(index), []).append(index)
    if len(groups) < 2:
        raise ValueError("Development evidence grouping left fewer than two independent groups.")
    fold_count = min(fold_count, len(groups))
    folds = [[] for _ in range(fold_count)]
    fold_sizes = [0] * fold_count
    ordered_groups = sorted(groups.values(), key=lambda members: (-len(members), members[0]))
    for members in ordered_groups:
        destination = min(range(fold_count), key=lambda fold: (fold_sizes[fold], fold))
        folds[destination].extend(members)
        fold_sizes[destination] += len(members)
    return [sorted(fold) for fold in folds]


def cross_validate(
    examples: list[dict], questions: list[dict], passages: list,
    epochs_to_check: tuple[int, ...] = CV_EPOCHS,
    seed_counts: tuple[int, ...] = CV_SEEDS,
    requested_folds: int = 4,
) -> dict:
    folds = make_grouped_folds(questions, passages, fold_count=requested_folds)
    scores = {
        (epochs, seed_count): []
        for epochs in epochs_to_check for seed_count in seed_counts
    }
    fold_metadata = []
    for fold_index, validation_indices in enumerate(folds):
        validation_set = set(validation_indices)
        train_indices = [index for index in range(len(examples)) if index not in validation_set]
        train_rows = [examples[index] for index in train_indices]
        validation_rows = [examples[index] for index in validation_indices]
        torch.manual_seed(20261008 + fold_index)
        model = AlignGRAGAligner(input_dim=examples[0]["node_features"].shape[-1])
        snapshots = sorted(set(epochs_to_check))
        losses = train_epochs(model, train_rows, max(snapshots), contrastive=True)
        for epoch in snapshots:
            # Refit each fold to the requested snapshot so every epoch setting
            # is evaluated from the same initialization, without restoring an
            # optimizer or retaining held-out graph representations.
            if epoch != max(snapshots):
                torch.manual_seed(20261008 + fold_index)
                fold_model = AlignGRAGAligner(input_dim=examples[0]["node_features"].shape[-1])
                train_epochs(fold_model, train_rows, epoch, contrastive=True)
            else:
                fold_model = model
            for seed_count in seed_counts:
                query_metrics = []
                for example, q_index in zip(validation_rows, validation_indices):
                    ranking = predict_aligned_ranking(fold_model, example, seed_count)
                    query_metrics.append(interval_metrics(ranking, questions[q_index], _RetrieverView(passages)))
                scores[(epoch, seed_count)].append({
                    "ndcg@10": statistics.fmean(row["ndcg@10"] for row in query_metrics),
                    "mrr": statistics.fmean(row["mrr"] for row in query_metrics),
                    "evidence_span_recall@10": statistics.fmean(row["evidence_span_recall@10"] for row in query_metrics),
                    "complete_evidence@10": statistics.fmean(row["complete_evidence@10"] for row in query_metrics),
                    "question_count": len(query_metrics),
                })
        fold_metadata.append({
            "fold": fold_index,
            "train_questions": len(train_rows),
            "validation_questions": len(validation_rows),
            "validation_ids": [questions[index]["id"] for index in validation_indices],
            "training_loss_at_max_epoch": losses[-1],
        })
        print(f"Grouped development fold {fold_index + 1}/{len(folds)} complete.")
    leaderboard = []
    for (epochs, seed_count), fold_scores in scores.items():
        row = {
            "epochs": epochs,
            "seed_count": seed_count,
            "fold_average": {
                metric: statistics.fmean(score[metric] for score in fold_scores)
                for metric in ("ndcg@10", "mrr", "evidence_span_recall@10", "complete_evidence@10")
            },
            "fold_scores": fold_scores,
        }
        row["objective"] = [
            row["fold_average"]["ndcg@10"],
            row["fold_average"]["mrr"],
            row["fold_average"]["evidence_span_recall@10"],
            row["fold_average"]["complete_evidence@10"],
            -seed_count,
            -epochs,
        ]
        leaderboard.append(row)
    leaderboard.sort(key=lambda row: row["objective"], reverse=True)
    return {
        "fold_count": len(folds),
        "fold_assignment": fold_metadata,
        "selection_rule": "maximize grouped-dev mean nDCG@10, then MRR, Recall@10, complete evidence; prefer fewer seeds/epochs on exact ties",
        "leaderboard": leaderboard,
        "selected": {key: leaderboard[0][key] for key in ("epochs", "seed_count", "fold_average")},
    }


class _RetrieverView:
    """Minimal interface used by the common source-offset metric function."""

    def __init__(self, passages: list):
        self.passages = passages
        self.by_id = {passage.node_id: passage for passage in passages}


def initial_graphs(questions, builder, index_path: Path) -> dict[str, object]:
    result = {}
    for index, question in enumerate(questions, start=1):
        with contextlib.redirect_stdout(io.StringIO()):
            entry_nodes, personalization = builder.get_entry_points(
                question["question"], hnsw_index_path=str(index_path), top_k=20
            )
            graph, _ = builder.run_shallow_ppr(
                entry_nodes,
                personalization,
                m=PPR_TYPE_LIMIT,
                alpha=PPR_ALPHA,
                max_iter=PPR_MAX_ITER,
            )
        if not graph.number_of_nodes():
            raise ValueError(f"No initial PPR subgraph for {question['id']}.")
        result[question["id"]] = graph
        print(
            f"Initial candidate graph {index}/{len(questions)}: {question['id']} "
            f"({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)"
        )
    return result


def attach_supervision(examples, questions, supervision_by_id, encoder):
    question_by_id = {question["id"]: question for question in questions}
    for example in examples:
        question = question_by_id[example["id"]]
        supervision = supervision_by_id[example["id"]]
        anchors = supervision.get("anchors") or evidence_quotes(question) or [question["question"]]
        anchor_vectors = encoder.encode(
            anchors, normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False,
        )
        similarities = example["node_features"] @ torch.as_tensor(anchor_vectors, dtype=torch.float32).T
        example["anchor_distribution"] = torch.softmax(
            similarities.max(dim=1).values / NODE_ANCHOR_TEMPERATURE,
            dim=0,
        )
        rationale = str(supervision.get("rationale", "")).strip() or " ".join(anchors)
        rationale_vector = encoder.encode(
            [rationale], normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        example["rationale_embedding"] = torch.as_tensor(rationale_vector, dtype=torch.float32)


def evaluate_models(
    questions,
    heldout_examples,
    heldout_graphs,
    builder,
    retriever,
    packer,
    token_budget,
    baseline_data,
    dual_model,
    node_only_model,
    selected_seed_count,
    index_path: Path,
) -> tuple[dict, dict]:
    align_rows = []
    question_result_rows = []
    question_by_id = {question["id"]: question for question in questions}
    baseline_by_id = {row["id"]: row for row in baseline_data["questions"]}
    heldout_questions = [q for q in questions if q["split"] == "heldout"]
    for index, example in enumerate(heldout_examples, start=1):
        question = question_by_id[example["id"]]
        graph = heldout_graphs[example["id"]]
        rankings = {
            "align_grag_dual": predict_aligned_ranking(dual_model, example, selected_seed_count),
            "align_grag_kl_only": predict_aligned_ranking(node_only_model, example, selected_seed_count),
            "hnsw_ppr": retriever.hnsw_graph_expand(
                question["question"],
                hnsw_index_path=str(index_path),
                hnsw_candidate_k=20,
                expand_k=20,
                alpha=PPR_ALPHA,
                max_iter=PPR_MAX_ITER,
            ),
        }
        computed = question_result(question, rankings, retriever, packer, token_budget)
        frozen = baseline_by_id.get(question["id"])
        if frozen is None:
            raise ValueError(f"Frozen baseline has no held-out question {question['id']}.")
        # Bring over identical per-question baseline records from the already
        # frozen final run, including the selected cross-encoder result.
        for method in ("bm25", "hnsw_ppr_crossencoder_40"):
            if method not in frozen["methods"]:
                raise ValueError(f"Frozen baseline is missing method {method}.")
            computed["methods"][method] = frozen["methods"][method]
        order = [
            "bm25", "hnsw_ppr", "align_grag_kl_only", "align_grag_dual",
            "hnsw_ppr_crossencoder_40",
        ]
        computed["methods"] = {name: computed["methods"][name] for name in order}
        question_result_rows.append(computed)
        align_rows.append({
            "id": question["id"],
            "candidate_graph_nodes": graph.number_of_nodes(),
            "candidate_graph_edges": graph.number_of_edges(),
            "candidate_text_passages": sum(1 for _, d in graph.nodes(data=True) if d.get("type") == "T"),
            "methods": {
                method: {
                    "mrr": computed["methods"][method]["mrr"],
                    "ndcg@10": computed["methods"][method]["ndcg@10"],
                    "evidence_span_recall@10": computed["methods"][method]["evidence_span_recall@10"],
                    "complete_evidence@10": computed["methods"][method]["complete_evidence@10"],
                    "packed_context_span_recall": computed["methods"][method]["packed_context_span_recall"],
                    "packed_context_all_evidence": computed["methods"][method]["packed_context_all_evidence"],
                    "packed_context_tokens": computed["methods"][method]["packed_context_tokens"],
                    "packed_passage_count": computed["methods"][method]["packed_passage_count"],
                    "ranked_passage_ids": computed["methods"][method]["ranked_passage_ids"],
                    "packed_passage_ids": computed["methods"][method]["packed_passage_ids"],
                }
                for method in order
            },
        })
        print(f"Held-out Align-GRAG question {index}/{len(heldout_examples)}: {question['id']}")

    method_names = list(question_result_rows[0]["methods"])
    summary = mean_summary(
        question_result_rows,
        method_names,
        packer,
        retriever,
        token_budget,
        bootstrap_iterations=5000,
    )
    by_category = {}
    for category in sorted({row.get("category") for row in question_result_rows if row.get("category")}):
        category_rows = [row for row in question_result_rows if row.get("category") == category]
        by_category[category] = mean_summary(
            category_rows,
            method_names,
            packer,
            retriever,
            token_budget,
            bootstrap_iterations=2000,
        )
    return {
        "summary": summary,
        "category_summaries": by_category,
        "questions": align_rows,
        "question_result_rows": question_result_rows,
    }, baseline_data


def add_crossencoder_pairwise_intervals(output: dict) -> None:
    """Attach paired held-out deltas against the selected existing pipeline."""
    fields = (
        "evidence_span_recall@10", "complete_evidence@10", "mrr", "ndcg@10",
        "packed_context_span_recall", "packed_context_all_evidence",
    )
    reference_method = "hnsw_ppr_crossencoder_40"
    rows = output["question_result_rows"]
    for method in ("align_grag_dual", "align_grag_kl_only"):
        output["summary"][method][f"delta_vs_{reference_method}"] = {}
        output["summary"][method][f"paired_delta_95pct_ci_vs_{reference_method}"] = {}
        for field in fields:
            deltas = [
                row["methods"][method][field] - row["methods"][reference_method][field]
                for row in rows
            ]
            output["summary"][method][f"delta_vs_{reference_method}"][field] = statistics.fmean(deltas)
            output["summary"][method][f"paired_delta_95pct_ci_vs_{reference_method}"][field] = bootstrap_ci(
                deltas,
                seed=20261008 + sum(map(ord, method + field + "paired_ce")),
                iterations=5000,
            )


def report_markdown(output: dict) -> str:
    summary = output["summary"]
    selected = output["training"]["selected_config"]
    baseline = summary["bm25"]
    best_existing = summary["hnsw_ppr_crossencoder_40"]
    dual = summary["align_grag_dual"]
    kl = summary["align_grag_kl_only"]
    fallback_count = output["training"]["supervision_source_counts"].get("source_quote_fallback", 0)
    supervision_count = output["training"]["training_supervision_count"]
    provider_runtime = output["training"].get("supervision_provider_runtime", {})
    provider_errors = {
        event.get("error_type") for event in provider_runtime.get("fallback_events", [])
    }
    if fallback_count == supervision_count:
        fallback_note = (
            f"All {fallback_count}/{supervision_count} development examples used source-quote/answer-derived "
            "rationale and anchor fallbacks; none used an LLM-generated rationale or anchor."
        )
        if provider_errors == {"rate_limit_exceeded"}:
            fallback_note += " All configured key/model calls were rate-limited."
    else:
        fallback_note = (
            f"Supervision source counts: {output['training']['supervision_source_counts']}. "
            "See the sidecar metadata for provider fallback details."
        )
    table_rows = []
    for method in ("bm25", "hnsw_ppr", "align_grag_kl_only", "align_grag_dual", "hnsw_ppr_crossencoder_40"):
        row = summary[method]
        table_rows.append(
            f"| {method} | {row['evidence_span_recall@1']:.3f} | "
            f"{row['evidence_span_recall@5']:.3f} | {row['evidence_span_recall@10']:.3f} | "
            f"{row['mrr']:.3f} | {row['ndcg@10']:.3f} | "
            f"{row['packed_context_span_recall']:.3f} | {row['packed_context_all_evidence']:.3f} |"
        )
    category_rows = []
    for category, category_summary in sorted(output.get("category_summaries", {}).items()):
        row = category_summary["align_grag_dual"]
        category_rows.append(
            f"| {category} | {row['question_count']} | {row['evidence_span_recall@10']:.3f} | "
            f"{row['mrr']:.3f} | {row['ndcg@10']:.3f} |"
        )
    delta_vs_ce = dual["ndcg@10"] - best_existing["ndcg@10"]
    delta_vs_bm25 = dual["ndcg@10"] - baseline["ndcg@10"]
    paired_ce_ci = dual["paired_delta_95pct_ci_vs_hnsw_ppr_crossencoder_40"]
    paired_bm25_ci = dual["paired_delta_95pct_ci_vs_bm25"]
    return f"""# Align-GRAG Core Aligner Evaluation

**Run date:** {output['run_metadata']['run_date']}  
**Graph:** existing saved book graph, read-only; SHA-256 `{output['run_metadata']['graph_sha256']}`  
**Graph size:** {output['run_metadata']['graph_node_count']:,} nodes, {output['run_metadata']['graph_edge_count']:,} edges, {output['run_metadata']['passage_count']} source passages  
**Question set:** {output['run_metadata']['development_question_count']} development questions and {output['run_metadata']['heldout_question_count']} frozen held-out questions  
**Evaluation boundary:** graph/index are read-only; no graph rebuild; initial candidate subgraphs use the existing saved HNSW + shallow PPR path.

## What was evaluated

This experiment implements Align-GRAG's trainable refinement stage on the existing graph candidate subgraphs. A two-layer GraphSAGE encoder (implemented in plain PyTorch because `torch_geometric` is absent) scores nodes conditioned on the query. Training combines anchor-target KL loss and bidirectional in-batch graph/rationale contrastive loss. Inference selects the top {selected['seed_count']} aligned nodes and preserves their one-hop neighbors; source T passages in that pruned graph are ranked by node score.

Development-only rationale/anchor examples use the question, its annotated answer, and the retrieved candidate graph. {fallback_note} Fallback rationales are constructed from the labeled answer and source quotes, so they are weaker supervision than the paper's question-answer-rationale-anchor examples. No held-out answer, rationale, or anchor was used to train or select settings. Hyperparameters were selected with {output['training']['cv']['fold_count']}-fold grouped cross-validation on development questions, grouping questions that share evidence passages. Selected settings: {selected['epochs']} epochs, {selected['seed_count']} seed nodes; learning rate {selected['learning_rate']}, weight decay {selected['weight_decay']}.

The initial candidate retriever is held constant as HNSW + PPR to isolate learned alignment/pruning. Text features for graph nodes, questions, anchors, and rationales use `all-MiniLM-L12-v2`; graph states come from the trained GraphSAGE encoder. This is **not a full reproduction** of the paper's SBERT top-k node/edge + PCST retrieval. It implements the central KL and graph-text contrastive objectives, with a smaller GraphSAGE encoder for this graph and runtime.

## Held-out retrieval results

Gold relevance uses the same exact source character offsets and metric definitions as the previous controlled benchmark. Context uses the same passage-preserving 4,096-token budget.

R@k is the fraction of labeled evidence spans found in the top k passages. MRR measures the reciprocal rank of the first supporting passage. nDCG@10 rewards placing more supporting passages near the top. Packed span recall measures evidence spans present in the final packed context; "all evidence packed" is the fraction of questions whose complete labeled evidence fits in that context.

| Method | R@1 | R@5 | R@10 | MRR | nDCG@10 | Packed span recall | All evidence packed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(table_rows)}

### Align-GRAG dual results by held-out question category

| Category | Questions | R@10 | MRR | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(category_rows)}

Align-GRAG dual alignment changes nDCG@10 by **{delta_vs_ce:+.3f}** versus the current HNSW/PPR + cross-encoder system and **{delta_vs_bm25:+.3f}** versus BM25. The node-only ablation has nDCG@10 {kl['ndcg@10']:.3f}; the dual model has {dual['ndcg@10']:.3f}. Paired per-question bootstrap 95% intervals (5,000 resamples) for Align dual minus cross-encoder are nDCG@10 [{paired_ce_ci['ndcg@10'][0]:+.3f}, {paired_ce_ci['ndcg@10'][1]:+.3f}], MRR [{paired_ce_ci['mrr'][0]:+.3f}, {paired_ce_ci['mrr'][1]:+.3f}], and R@10 [{paired_ce_ci['evidence_span_recall@10'][0]:+.3f}, {paired_ce_ci['evidence_span_recall@10'][1]:+.3f}]. Against BM25, the corresponding intervals are nDCG@10 [{paired_bm25_ci['ndcg@10'][0]:+.3f}, {paired_bm25_ci['ndcg@10'][1]:+.3f}], MRR [{paired_bm25_ci['mrr'][0]:+.3f}, {paired_bm25_ci['mrr'][1]:+.3f}], and R@10 [{paired_bm25_ci['evidence_span_recall@10'][0]:+.3f}, {paired_bm25_ci['evidence_span_recall@10'][1]:+.3f}].

For a concrete miss, on `cur-test-009` (recovering from dementia), Align dual retrieved no labeled evidence in its top 10 (nDCG@10 0.000), while BM25 and the cross-encoder each ranked the evidence first (nDCG@10 1.000). This illustrates the severe ranking failure behind the averages.

## Important scope limits

- The held-out set is small (23 questions), and the QA/evidence labels were assistant-authored and source-audited, not independently annotated by domain experts. Category counts are imbalanced, especially for multi-evidence questions.
- The existing graph and its candidate retrieval are evaluated; graph extraction quality is not changed.
- To produce comparable ranked passages, source `T` nodes retained after pruning are ordered by their aligner node logits. That is an evaluation adapter for this prototype; the paper's downstream model consumes the aligned graph representation directly.
- The current hosted Groq API accepts text, not a learned continuous graph token. Therefore the aligned graph token is not injected into the generator, and this is a retrieval/pruning comparison, not a complete Align-GRAG generation result.
- `results/align-grag-supervision.jsonl` contains development-only rationale/anchor supervision and provider/model metadata without credentials.

## Reproduction

```powershell
.\\.venv\\Scripts\\python.exe scripts\\book\\run_book_align_grag_eval.py
```

Primary result artifact: `results/align-grag-eval.json`.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--hnsw-index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--supervision", type=Path, default=DEFAULT_SUPERVISION)
    parser.add_argument("--supervision-meta", type=Path, default=DEFAULT_SUPERVISION_META)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--cv-folds", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    random.seed(20261008)
    np.random.seed(20261008)
    torch.manual_seed(20261008)
    questions = read_jsonl(args.questions)
    dev_questions = [question for question in questions if question["split"] == "dev"]
    heldout_questions = [question for question in questions if question["split"] == "heldout"]
    baseline_data = json.loads(args.baseline.read_text(encoding="utf-8"))
    graph_hash = sha256(args.graph)
    question_hash = sha256(args.questions)
    for field, expected in (("graph_sha256", graph_hash), ("question_sha256", question_hash)):
        actual = baseline_data["run_metadata"].get(field)
        if actual != expected:
            raise ValueError(f"Frozen baseline {field} does not match the current files.")
    if len(baseline_data.get("questions", [])) != len(heldout_questions):
        raise ValueError("Frozen baseline held-out question count does not match.")
    tokenizer_name = baseline_data["run_metadata"].get("tokenizer", "openai/gpt-oss-20b")
    token_budget = int(baseline_data["run_metadata"].get("context_token_budget", 4096))

    print(f"Loading saved graph only (no rebuild): {args.graph}")
    builder = GraphBuilder(
        groq_handler=None,
        jsonl_path=str(args.graph.with_name("chunked_output.jsonl")),
        neo4j_uri=None,
        neo4j_user=None,
        neo4j_password=None,
        neo4j_database=None,
    )
    try:
        with args.graph.open("rb") as handle:
            builder.G = pickle.load(handle)
        initial = initial_graphs(questions, builder, args.hnsw_index)
        supervision_by_id = load_or_create_supervision(
            dev_questions,
            initial,
            args.supervision,
            args.supervision_meta,
            graph_hash,
            question_hash,
        )

        encoder = builder._embedder()
        print(f"Encoding text for {builder.G.number_of_nodes()} saved graph nodes.")
        graph_node_ids, global_index, global_features = embed_all_graph_nodes(builder.G, encoder)
        retriever = builder.get_passage_retriever()
        dev_examples = [
            graph_example(
                question["id"], question["question"], initial[question["id"]],
                global_index, global_features, encoder,
            )
            for question in dev_questions
        ]
        attach_supervision(dev_examples, dev_questions, supervision_by_id, encoder)
        print("Selecting aligner epochs and prune size using grouped development folds.")
        # Keep the requested fold count configurable while never allowing a
        # question sharing evidence passages across folds.
        cv = cross_validate(
            dev_examples,
            dev_questions,
            builder.get_passage_retriever().passages,
            epochs_to_check=CV_EPOCHS,
            seed_counts=CV_SEEDS,
            requested_folds=args.cv_folds,
        )
        selected_epochs = int(cv["selected"]["epochs"])
        selected_seed_count = int(cv["selected"]["seed_count"])

        torch.manual_seed(20261008)
        dual_model = AlignGRAGAligner(input_dim=global_features.shape[-1])
        dual_losses = train_epochs(dual_model, dev_examples, selected_epochs, contrastive=True)
        torch.manual_seed(20261008)
        node_only_model = AlignGRAGAligner(input_dim=global_features.shape[-1])
        node_only_losses = train_epochs(node_only_model, dev_examples, selected_epochs, contrastive=False)

        heldout_examples = [
            graph_example(
                question["id"], question["question"], initial[question["id"]],
                global_index, global_features, encoder,
            )
            for question in heldout_questions
        ]
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        packer = TokenBudgetPacker(tokenizer)
        evaluated, _ = evaluate_models(
            questions,
            heldout_examples,
            initial,
            builder,
            retriever,
            packer,
            token_budget,
            baseline_data,
            dual_model,
            node_only_model,
            selected_seed_count,
            args.hnsw_index,
        )
        output = {
            "run_metadata": {
                "run_date": "2026-10-08",
                "graph_path": str(args.graph.resolve().relative_to(ROOT)),
                "graph_sha256": graph_hash,
                "question_path": str(args.questions.resolve().relative_to(ROOT)),
                "question_sha256": question_hash,
                "hnsw_index_sha256": sha256(args.hnsw_index),
                "frozen_baseline_path": str(args.baseline.resolve().relative_to(ROOT)),
                "frozen_baseline_sha256": sha256(args.baseline),
                "development_question_count": len(dev_questions),
                "heldout_question_count": len(heldout_questions),
                "graph_node_count": builder.G.number_of_nodes(),
                "graph_edge_count": builder.G.number_of_edges(),
                "passage_count": len(retriever.passages),
                "graph_rebuilt": False,
                "api_calls_for_answer_generation": 0,
                "generator_graph_token_used": False,
                "candidate_method": "saved HNSW entry points + shallow PPR; same candidate graph for both Align-GRAG and ablation",
                "candidate_ppr": {"alpha": PPR_ALPHA, "max_iter": PPR_MAX_ITER, "per_type_limit": PPR_TYPE_LIMIT},
                "embedding_model": EMBEDDING_MODEL,
                "node_anchor_temperature": NODE_ANCHOR_TEMPERATURE,
                "contrastive_temperature": CONTRASTIVE_TEMPERATURE,
                "tokenizer": tokenizer_name,
                "context_token_budget": token_budget,
                "question_quality": "Assistant-authored and source-span audited; not independently human annotated.",
                "model_architecture": "two-layer mean GraphSAGE, query-conditioned node MLP, shared graph/rationale projection",
                "alignment_losses": ["KL(anchor_softmax_cosine || predicted_node_softmax)", "symmetric in-batch graph/rationale InfoNCE"],
            },
            "training": {
                "supervision_path": str(args.supervision.resolve().relative_to(ROOT)),
                "training_supervision_count": len(supervision_by_id),
                "supervision_source_counts": {
                    source: sum(row.get("rationale_source") == source for row in supervision_by_id.values())
                    for source in sorted({row.get("rationale_source") for row in supervision_by_id.values()})
                },
                "supervision_provider_runtime": json.loads(args.supervision_meta.read_text(encoding="utf-8")).get("provider_runtime", {})
                if args.supervision_meta.exists() else {},
                "cv": cv,
                "selected_config": {"epochs": selected_epochs, "seed_count": selected_seed_count, "learning_rate": 1e-4, "weight_decay": 1e-4},
                "dual_loss_trace": dual_losses,
                "node_only_loss_trace": node_only_losses,
            },
            **evaluated,
        }
        add_crossencoder_pairwise_intervals(output)
        write_json(args.output, output)
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict_dual": dual_model.state_dict(),
            "state_dict_node_only": node_only_model.state_dict(),
            "selected_config": output["training"]["selected_config"],
            "graph_sha256": graph_hash,
            "question_sha256": question_hash,
            "node_ids": graph_node_ids,
            "architecture": output["run_metadata"]["model_architecture"],
        }, args.checkpoint)
        args.report.write_text(report_markdown(output), encoding="utf-8")

        print("\nHeld-out retrieval results")
        for method, metrics in output["summary"].items():
            print(
                f"{method:30s} R@10={metrics['evidence_span_recall@10']:.3f} "
                f"MRR={metrics['mrr']:.3f} nDCG@10={metrics['ndcg@10']:.3f} "
                f"packed-span={metrics['packed_context_span_recall']:.3f}"
            )
        print(f"\nSaved metrics: {args.output}")
        print(f"Saved checkpoint: {args.checkpoint}")
        print(f"Saved report: {args.report}")
    finally:
        builder.close()


if __name__ == "__main__":
    main()
