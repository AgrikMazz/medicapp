"""Build a full-PDF passage graph and evaluate the annotated questions."""

import json
import os
import sys
import argparse

from dotenv import load_dotenv
import fitz

for output_stream in (sys.stdout, sys.stderr):
    if hasattr(output_stream, "reconfigure"):
        output_stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(ROOT, ".env"))
sys.path.insert(0, ROOT)

from modules.chunker import Chunker
from modules.evaluator import NodeRAGEvaluator
from modules.generator import GraphAugmentedGenerator
from modules.graph_builder import GraphBuilder
from modules.groq_handler import GroqHandler


PDF_PATH = os.path.join(ROOT, "medbook.pdf")
ARTIFACT_DIR = os.path.join(ROOT, "data", "medbook-full-text")
CHUNK_PATH = os.path.join(ARTIFACT_DIR, "chunked_output.jsonl")
GRAPH_PATH = os.path.join(ARTIFACT_DIR, "graph.pkl")
INDEX_PATH = os.path.join(ARTIFACT_DIR, "graph-index.bin")
QUESTION_PATH = os.path.join(ROOT, "data", "medbook-full_questions.jsonl")
RESULT_PATH = os.path.join(ROOT, "results", "medbook-full-text-results.json")
RETRIEVAL_RESULT_PATH = os.path.join(ROOT, "results", "medbook-full-retrieval-results.json")
GENERATION_RESULT_PATH = os.path.join(ROOT, "results", "medbook-full-generation-results.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Reuse the saved full-text graph and HNSW index.",
    )
    evaluation_mode = parser.add_mutually_exclusive_group()
    evaluation_mode.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Measure retrieval and save its exact generation contexts without making model calls.",
    )
    evaluation_mode.add_argument(
        "--generation-only",
        action="store_true",
        help="Generate and score answers from the exact contexts saved by a retrieval-only pass.",
    )
    args = parser.parse_args()
    if args.generation_only:
        args.evaluate_only = True

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    chunker = Chunker(store_file=CHUNK_PATH)
    with fitz.open(PDF_PATH) as pdf:
        page_count = len(pdf)
    if not args.evaluate_only:
        pages = chunker.pdf_to_pages(PDF_PATH)
        chunks = chunker.segment_into_chunks(pages, source_id="medbook", persist=False)
        chunker.write_chunks(chunks)
        print(f"Extracted {len(pages)} pages into {len(chunks)} page-aware source chunks.")
    else:
        with open(CHUNK_PATH, "r", encoding="utf-8") as chunk_file:
            chunk_count = sum(1 for line in chunk_file if line.strip())
        print(f"Reusing {page_count} pages and {chunk_count} saved source chunks.")

    groq = GroqHandler()
    builder = GraphBuilder(
        groq_handler=groq,
        jsonl_path=CHUNK_PATH,
        neo4j_uri=None,
        neo4j_user=None,
        neo4j_password=None,
        neo4j_database=None,
    )
    generator = GraphAugmentedGenerator(
        groq,
        max_context_chars=int(os.getenv("ANSWER_CONTEXT_CHAR_LIMIT", "24000")),
    )
    evaluator = NodeRAGEvaluator(chunker, builder, generator, entity_handler=None)

    try:
        saved_contexts = None
        if args.generation_only:
            with open(RETRIEVAL_RESULT_PATH, "r", encoding="utf-8") as retrieval_file:
                retrieval_results = json.load(retrieval_file)
            saved_contexts = {
                item["question"]: item["generation_context"]
                for item in retrieval_results.get("predictions", [])
                if item.get("generation_context")
            }

        if not args.evaluate_only:
            builder.G.clear()
            builder.build_G4_text_attachment(chunk_file=CHUNK_PATH)

            passage_nodes = [
                (node_id, data)
                for node_id, data in builder.G.nodes(data=True)
                if data.get("type") == "T"
            ]
            passage_nodes.sort(key=lambda item: (
                item[1].get("page_start") or 0,
                item[1].get("char_start") or 0,
                item[0],
            ))
            for (previous_id, _), (node_id, _) in zip(passage_nodes, passage_nodes[1:]):
                builder.G.add_edge(previous_id, node_id, label="next_passage")

            builder.build_G5_semantic_hnsw_index(index_path=INDEX_PATH)
            builder.save_pickle(GRAPH_PATH)
            print(f"Built passage graph with {builder.G.number_of_nodes()} nodes and {builder.G.number_of_edges()} edges.")

        results = evaluator.evaluate_pdf_questions(
            questions_path=QUESTION_PATH,
            hnsw_index_path=INDEX_PATH,
            graph_path=GRAPH_PATH,
            retrieval_only=args.retrieval_only,
            generation_contexts=saved_contexts,
        )
        results["run_metadata"] = {
            "source_pdf": os.path.basename(PDF_PATH),
            "indexed_page_count": page_count,
            "indexed_pdf_pages": list(range(1, page_count + 1)),
            "source_chunk_count": len(builder.G),
            "index_mode": "full-text passage graph; no LLM entity/relation/community enrichment",
            "evaluation_mode": (
                "retrieval-only" if args.retrieval_only else
                "generation-replay" if args.generation_only else
                "retrieval-plus-generation"
            ),
            "model": os.getenv("GROQ_MODEL"),
            "llm_runtime": groq.get_runtime_metadata(),
            "reasoning_effort": os.getenv("GROQ_REASONING_EFFORT"),
            "question_set": os.path.basename(QUESTION_PATH),
            "successful_api_requests": groq.successful_requests,
            "api_token_usage": groq.total_usage,
        }
        result_path = (
            RETRIEVAL_RESULT_PATH if args.retrieval_only else
            GENERATION_RESULT_PATH if args.generation_only else
            RESULT_PATH
        )
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, "w", encoding="utf-8") as result_file:
            json.dump(results, result_file, indent=2, ensure_ascii=False)

        print("Evaluation summary:")
        print(json.dumps(results.get("summary", {}), indent=2))
        print(f"Detailed results: {result_path}")
    finally:
        builder.close()


if __name__ == "__main__":
    main()
