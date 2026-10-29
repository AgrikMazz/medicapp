"""Small metric helpers matching the evaluation reported in the NodeRAG paper.

Input is JSONL: one result per system/query. This script scores saved outputs;
it does not run a retriever, generator, or LLM judge.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def average(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    return mean(values) if values else None


def summarize(rows):
    """Summarize one dataset/system group from per-question result rows."""
    result = {
        "questions": len(rows),
        "retrieval_tokens_avg": average(rows, "retrieval_tokens"),
        "query_seconds_avg": average(rows, "query_seconds"),
    }

    # For HotpotQA/MuSiQue, `correct` must be the LLM judge's 0/1 verdict.
    judged = [row["correct"] for row in rows if row.get("correct") is not None]
    if judged:
        result["accuracy"] = mean(float(value) for value in judged)

    # MultiHop-RAG's published `Sco` is its benchmark answer score.
    scores = [row["score"] for row in rows if row.get("score") is not None]
    if scores:
        result["score"] = mean(float(value) for value in scores)

    # Arena outcome is relative to the human-written LFRQA/gold answer.
    arena = [row["gold_outcome"] for row in rows if row.get("gold_outcome")]
    if arena:
        result["gold_win_rate"] = sum(x == "win" for x in arena) / len(arena)
        result["gold_tie_rate"] = sum(x == "tie" for x in arena) / len(arena)
        result["gold_win_tie_rate"] = sum(x in {"win", "tie"} for x in arena) / len(arena)

    # Index metrics are run-level values; include one value per run, not per query.
    for key in ("index_seconds", "storage_mb"):
        value = average(rows, key)
        if value is not None:
            result[key + "_avg"] = value
    return result


def summarize_pairwise(rows):
    """Count pairwise judge outcomes. outcome is relative to method_a."""
    grouped = defaultdict(list)
    for row in rows:
        key = (row["dataset"], row["method_a"], row["method_b"])
        grouped[key].append(row["outcome"])

    result = {}
    for (dataset, method_a, method_b), outcomes in grouped.items():
        n = len(outcomes)
        result[f"{dataset}: {method_a} vs {method_b}"] = {
            "questions": n,
            "win_rate_a": outcomes.count("win") / n,
            "tie_rate": outcomes.count("tie") / n,
            "win_rate_b": outcomes.count("loss") / n,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="JSONL result file")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.results.read_text(encoding="utf-8").splitlines() if line.strip()]
    regular = [row for row in rows if row.get("kind", "result") == "result"]
    pairwise = [row for row in rows if row.get("kind") == "pairwise"]

    groups = defaultdict(list)
    for row in regular:
        groups[(row["dataset"], row["method"])].append(row)

    report = {
        f"{dataset} / {method}": summarize(group)
        for (dataset, method), group in sorted(groups.items())
    }
    if pairwise:
        report["pairwise"] = summarize_pairwise(pairwise)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
