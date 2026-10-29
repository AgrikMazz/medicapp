"""Build and evaluate page-grounded GraphRAG runs on medbook.pdf."""

import json
import os
import sys
import argparse
import fitz

from dotenv import load_dotenv

for output_stream in (sys.stdout, sys.stderr):
    if hasattr(output_stream, "reconfigure"):
        output_stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(ROOT, ".env"))
sys.path.insert(0, ROOT)

from modules.chunker import Chunker
from modules.entity_handler import EntityHandler
from modules.evaluator import NodeRAGEvaluator
from modules.generator import GraphAugmentedGenerator
from modules.graph_builder import GraphBuilder
from modules.groq_handler import GroqHandler


PDF_PATH = os.path.join(ROOT, "medbook.pdf")
DEMO_PAGES = [3, 4, 10, 11, 12, 13, 18]


class SavedAnswerGenerator:
    def __init__(self, answers_by_question):
        self.answers_by_question = answers_by_question

    def generate_answer(self, query, _graph):
        return self.answers_by_question.get(query, "")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-book",
        action="store_true",
        help="Build/evaluate all PDF pages in separate medbook-full artifacts.",
    )
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Reuse the existing demo graph and HNSW index; only run the annotated questions.",
    )
    run_mode.add_argument(
        "--rescore-only",
        action="store_true",
        help="Recompute retrieval/answer metrics from saved predictions without making Groq calls.",
    )
    args = parser.parse_args()

    artifact_name = "medbook-full" if args.full_book else "medbook-demo"
    artifact_dir = os.path.join(ROOT, "data", artifact_name)
    question_path = os.path.join(ROOT, "data", f"{artifact_name}_questions.jsonl")
    result_path = os.path.join(ROOT, "results", f"{artifact_name}-results.json")
    pdf_pages = None if args.full_book else DEMO_PAGES

    if not os.path.isfile(PDF_PATH):
        raise FileNotFoundError(f"Expected the source PDF at {PDF_PATH}")
    if not os.path.isfile(question_path):
        raise FileNotFoundError(f"Expected evaluation questions at {question_path}")

    groq = GroqHandler()
    chunker = Chunker(store_file=os.path.join(artifact_dir, "chunked_output.jsonl"))
    generator = GraphAugmentedGenerator(
        groq,
        max_context_chars=int(os.getenv("ANSWER_CONTEXT_CHAR_LIMIT", "24000")),
    )
    builder = GraphBuilder(
        groq_handler=groq,
        jsonl_path=chunker.store_file,
        neo4j_uri=None,
        neo4j_user=None,
        neo4j_password=None,
        neo4j_database=None,
    )
    entity_handler = EntityHandler(
        groq_handler=groq,
        graph_builder=builder,
        generator=generator,
        batch_size=8,
        wait_time=0,
        progress_file=os.path.join(artifact_dir, "progress.json"),
        failed_file=os.path.join(artifact_dir, "permanent_failed.json"),
        store_file=chunker.store_file,
        community_file=os.path.join(artifact_dir, "community_map.json"),
        max_retries=2,
    )
    evaluator = NodeRAGEvaluator(chunker, builder, generator, entity_handler)

    try:
        previous_results = None
        if args.rescore_only:
            if not os.path.isfile(result_path):
                raise FileNotFoundError("No saved demo results are available to rescore.")
            with open(result_path, "r", encoding="utf-8") as previous_file:
                previous_results = json.load(previous_file)
            answer_map = {
                item["question"]: item.get("prediction", "")
                for item in previous_results.get("predictions", [])
            }
            evaluator.generator = SavedAnswerGenerator(answer_map)

        if not args.evaluate_only and not args.rescore_only:
            evaluator.build_from_paragraphs(
                eval=False,
                pdf_path=PDF_PATH,
                page_numbers=pdf_pages,
                artifact_dir=artifact_dir,
            )
        results = evaluator.evaluate_pdf_questions(
            questions_path=question_path,
            hnsw_index_path=os.path.join(artifact_dir, "graph-index.bin"),
            graph_path=os.path.join(artifact_dir, "graph.pkl"),
        )
        if pdf_pages is None:
            with fitz.open(PDF_PATH) as pdf:
                indexed_pages = list(range(1, len(pdf) + 1))
        else:
            indexed_pages = pdf_pages
        results["run_metadata"] = {
            "source_pdf": os.path.basename(PDF_PATH),
            "indexed_pdf_pages": indexed_pages,
            "indexed_page_count": len(indexed_pages),
            "model": os.getenv("GROQ_MODEL"),
            "llm_runtime": groq.get_runtime_metadata(),
            "reasoning_effort": os.getenv("GROQ_REASONING_EFFORT"),
            "question_set": os.path.basename(question_path),
            "successful_api_requests": groq.successful_requests,
            "api_token_usage": groq.total_usage,
        }
        if args.rescore_only:
            results["run_metadata"] = previous_results.get("run_metadata", {})
            results["run_metadata"]["rescored_locally"] = True
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, "w", encoding="utf-8") as output_file:
            json.dump(results, output_file, indent=2, ensure_ascii=False)

        print("Evaluation summary:")
        print(json.dumps(results.get("summary", {}), indent=2))
        print(f"Detailed results: {result_path}")
    finally:
        builder.close()


if __name__ == "__main__":
    main()
