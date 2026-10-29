"""Generate answers from frozen retrieved contexts and score them separately.

This script consumes the final retrieval artifact; it does not retrieve again,
change rankings, or rebuild the graph. It makes Groq calls through GroqHandler.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import string
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

import sys
sys.path.insert(0, str(ROOT))
from modules.generator import GraphAugmentedGenerator  # noqa: E402
from modules.groq_handler import GroqHandler  # noqa: E402


DEFAULT_RETRIEVAL = ROOT / "results" / "book-retrieval-final.json"
DEFAULT_QUESTIONS = ROOT / "data" / "book-graph" / "eval_questions.reviewed.jsonl"
DEFAULT_OUTPUT = ROOT / "results" / "book-retrieval-generation-final.json"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_answer(text):
    text = str(text).lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_scores(prediction, references):
    prediction = re.sub(
        r"(?:\[[^\]]+\]|\u3010[^\u3011]*\u3011)",
        " ",
        str(prediction).strip(),
        flags=re.I,
    )
    pred = normalize_answer(prediction).split()
    best_f1 = best_em = best_rouge_l = 0.0
    for reference in references:
        gold = normalize_answer(reference).split()
        if not gold:
            continue
        common = Counter(pred) & Counter(gold)
        overlap = sum(common.values())
        precision = overlap / len(pred) if pred else 0.0
        recall = overlap / len(gold)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        best_f1 = max(best_f1, f1)
        best_em = max(best_em, float(pred == gold))

        previous = [0] * (len(gold) + 1)
        for token in pred:
            current = [0]
            for index, gold_token in enumerate(gold, start=1):
                current.append(
                    previous[index - 1] + 1
                    if token == gold_token
                    else max(previous[index], current[-1])
                )
            previous = current
        lcs = previous[-1]
        rouge_precision = lcs / len(pred) if pred else 0.0
        rouge_recall = lcs / len(gold)
        rouge_l = (
            2 * rouge_precision * rouge_recall / (rouge_precision + rouge_recall)
            if rouge_precision + rouge_recall else 0.0
        )
        best_rouge_l = max(best_rouge_l, rouge_l)
    return {
        "token_f1": best_f1,
        "exact_match": best_em,
        "rouge_l": best_rouge_l,
    }


def citation_scores(answer, context, question):
    cited = re.findall(
        r"(?:\[|\u3010)\s*source\s*=\s*([A-Za-z0-9._:-]+)\s*(?:\]|\u3011)",
        answer,
        flags=re.I,
    )
    context_ids = set(re.findall(
        r"\[T\s*\|\s*source=([^\s|\]]+)", context, flags=re.I
    ))
    gold_ids = {
        str(span.get("passage_id", "")).removeprefix("T_")
        for span in question.get("evidence_spans", [])
        if span.get("passage_id")
    }
    cited_set = set(cited)
    valid = sum(source_id in context_ids for source_id in cited)
    gold_hits = len(cited_set & gold_ids)
    return {
        "citation_count": len(cited),
        "citation_present": float(bool(cited)),
        "context_citation_precision": valid / len(cited) if cited else 0.0,
        "gold_evidence_citation_precision": gold_hits / len(cited) if cited else 0.0,
        "gold_evidence_citation_recall": gold_hits / len(gold_ids) if gold_ids else 0.0,
        "cited_source_ids": cited,
        "context_source_ids": sorted(context_ids),
        "gold_evidence_source_ids": sorted(gold_ids),
    }


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", default=str(DEFAULT_RETRIEVAL))
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    retrieval = json.loads(Path(args.retrieval).read_text(encoding="utf-8"))
    questions = read_jsonl(args.questions)
    if retrieval["run_metadata"].get("mode") != "final":
        raise ValueError("Generation evaluation requires a frozen final retrieval run.")
    if retrieval["run_metadata"].get("question_sha256") != sha256(args.questions):
        raise ValueError("Question file differs from the frozen retrieval run.")
    references = {item["id"]: item for item in questions}
    method = retrieval["run_metadata"]["selected_config"]["selected_method"]
    stored_rows = {item["id"]: item for item in retrieval["questions"]}
    eval_ids = [row["id"] for row in retrieval["questions"]]
    if any(question_id not in references for question_id in eval_ids):
        raise ValueError("The retrieval artifact and question file have different IDs.")

    prior = None
    prior_metadata = {}
    output_path = Path(args.output)
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if (
            existing.get("run_metadata", {}).get("retrieval_sha256") == sha256(args.retrieval)
            and existing.get("run_metadata", {}).get("question_sha256") == sha256(args.questions)
            and existing.get("run_metadata", {}).get("method") == method
        ):
            prior = {row["id"]: row for row in existing.get("questions", []) if row.get("answer")}
            prior_metadata = existing.get("run_metadata", {})

    groq = GroqHandler()
    generator = GraphAugmentedGenerator(groq)
    predictions = []
    for index, question_id in enumerate(eval_ids, start=1):
        item = references[question_id]
        retrieved = stored_rows[question_id]["methods"][method]
        context = retrieved["packed_context"]
        if prior and question_id in prior:
            row = prior[question_id]
            row["references"] = item.get("answers", [])
            row["answer_metrics"] = answer_scores(row["answer"], item.get("answers", []))
            row["citation_metrics"] = citation_scores(row["answer"], context, item)
            predictions.append(row)
            print(f"Reusing saved generation {index}/{len(eval_ids)}: {question_id}")
            continue

        prompt_answer = None
        for attempt in range(3):
            prompt_answer = generator.generate_answer(
                item["question"], None, textual_context=context
            )
            if prompt_answer and prompt_answer != "rate_limit_exceeded":
                break
            if attempt < 2:
                time.sleep(max(2.2, GroqHandler._retry_delay(groq.last_rate_limit_info)))
        if not prompt_answer or prompt_answer == "rate_limit_exceeded":
            raise RuntimeError(
                f"Generation failed for {question_id} after key/model fallback and retries."
            )

        row = {
            "id": question_id,
            "question": item["question"],
            "category": item.get("question_category"),
            "answer": prompt_answer.strip(),
            "references": item.get("answers", []),
            "answer_metrics": answer_scores(prompt_answer, item.get("answers", [])),
            "citation_metrics": citation_scores(prompt_answer, context, item),
            "retrieved_passage_ids": retrieved["ranked_passage_ids"],
            "packed_passage_ids": retrieved["packed_passage_ids"],
            "packed_context_tokens": retrieved["packed_context_tokens"],
            "retrieval_evidence_span_recall@10": retrieved["evidence_span_recall@10"],
            "retrieval_packed_context_span_recall": retrieved["packed_context_span_recall"],
        }
        predictions.append(row)
        checkpoint = {
            "run_metadata": {
                "mode": "generation_on_frozen_retrieved_contexts",
                "retrieval_path": str(Path(args.retrieval).resolve().relative_to(ROOT)),
                "retrieval_sha256": sha256(args.retrieval),
                "question_path": str(Path(args.questions).resolve().relative_to(ROOT)),
                "question_sha256": sha256(args.questions),
                "method": method,
                "question_count": len(eval_ids),
                "generation_completed": len(predictions),
                "retrieval_recomputed": False,
                "graph_rebuilt": False,
                "api_calls": groq.successful_requests,
                "api_token_usage": dict(groq.total_usage),
                "llm_runtime": groq.get_runtime_metadata(),
            },
            "questions": predictions,
        }
        atomic_write(output_path, checkpoint)
        print(f"Generated {index}/{len(eval_ids)}: {question_id}")

    numeric_fields = ("token_f1", "exact_match", "rouge_l")
    summary = {
        field: sum(row["answer_metrics"][field] for row in predictions) / len(predictions)
        for field in numeric_fields
    }
    for field in (
        "citation_present",
        "context_citation_precision",
        "gold_evidence_citation_precision",
        "gold_evidence_citation_recall",
    ):
        summary[field] = sum(row["citation_metrics"][field] for row in predictions) / len(predictions)
    summary["answers_generated"] = len(predictions)
    summary["questions_with_any_citation"] = sum(
        row["citation_metrics"]["citation_present"] for row in predictions
    )
    final = {
        "run_metadata": {
            "mode": "generation_on_frozen_retrieved_contexts",
            "retrieval_path": str(Path(args.retrieval).resolve().relative_to(ROOT)),
            "retrieval_sha256": sha256(args.retrieval),
            "question_path": str(Path(args.questions).resolve().relative_to(ROOT)),
            "question_sha256": sha256(args.questions),
            "method": method,
            "question_count": len(eval_ids),
            "retrieval_recomputed": False,
            "graph_rebuilt": False,
            "api_calls": groq.successful_requests,
            "api_token_usage": dict(groq.total_usage),
            "llm_runtime": groq.get_runtime_metadata(),
            "generation_limitations": "Lexical answer scores and source-ID citation agreement are diagnostics; no human or claim-level faithfulness judge was run.",
        },
        "summary": summary,
        "questions": predictions,
    }
    if prior_metadata and groq.successful_requests == 0:
        final["run_metadata"]["api_calls"] = prior_metadata.get("api_calls", 0)
        final["run_metadata"]["api_token_usage"] = prior_metadata.get("api_token_usage", {})
        final["run_metadata"]["llm_runtime"] = prior_metadata.get("llm_runtime", {})
    atomic_write(output_path, final)
    print("\nGeneration results")
    for field, value in summary.items():
        print(f"{field}: {value:.3f}" if isinstance(value, float) else f"{field}: {value}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
