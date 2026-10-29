"""Build a typed graph from a plain-text book and evaluate book-grounded QA."""

import argparse
import hashlib
import json
import os
import re
import sys

from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(ROOT, ".env"))
sys.path.insert(0, ROOT)

for output_stream in (sys.stdout, sys.stderr):
    if hasattr(output_stream, "reconfigure"):
        output_stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from modules.chunker import Chunker
from modules.entity_handler import EntityHandler
from modules.evaluator import NodeRAGEvaluator
from modules.generator import GraphAugmentedGenerator
from modules.graph_builder import GraphBuilder
from modules.groq_handler import GroqHandler


DEFAULT_TEXT = os.path.join(ROOT, "data", "medical-CareForInsane", "book.txt")


def source_windows(text, max_chars=5000):
    """Split text at paragraph boundaries while retaining exact source offsets."""
    paragraphs = []
    for match in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", text, flags=re.DOTALL):
        paragraphs.append((match.start(), match.end(), match.group(0)))

    windows = []
    current = []
    current_start = None
    current_end = None
    current_chars = 0
    for start, end, paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                windows.append({"start": current_start, "end": current_end, "text": text[current_start:current_end]})
                current, current_start, current_end, current_chars = [], None, None, 0
            for block_start in range(start, end, max_chars):
                block_end = min(end, block_start + max_chars)
                windows.append({"start": block_start, "end": block_end, "text": text[block_start:block_end]})
            continue
        if current and current_chars + len(paragraph) > max_chars:
            windows.append({"start": current_start, "end": current_end, "text": text[current_start:current_end]})
            current, current_start, current_end, current_chars = [], None, None, 0
        if current_start is None:
            current_start = start
        current.append(paragraph)
        current_end = end
        current_chars += len(paragraph)
    if current:
        windows.append({"start": current_start, "end": current_end, "text": text[current_start:current_end]})

    # A text file without blank-line paragraph breaks still gets evaluated.
    if not windows and text.strip():
        for start in range(0, len(text), max_chars):
            end = min(len(text), start + max_chars)
            windows.append({"start": start, "end": end, "text": text[start:end]})
    return windows


def generate_synthetic_questions(text, groq, count=24):
    """Create an explicitly synthetic QA pilot with exact evidence offsets."""
    windows = source_windows(text)
    if not windows:
        raise ValueError("Cannot generate evaluation questions from an empty book.")

    questions = []
    seen = set()
    attempts = 0
    max_attempts = max(count * 3, 12)
    while len(questions) < count and attempts < max_attempts:
        attempt_index = attempts
        index = attempt_index % len(windows)
        first = windows[index]
        cross_section = len(windows) > 1 and attempt_index % 2 == 0
        second = windows[len(windows) - 1 - index] if cross_section else first
        if cross_section and first["start"] == second["start"]:
            cross_section = False
        attempts += 1

        if cross_section:
            instruction = (
                "Write one answerable question that requires combining information from BOTH excerpts. "
                "Provide one short, exact, contiguous evidence quote from each excerpt."
            )
            context = f"<ExcerptA>\n{first['text']}\n</ExcerptA>\n<ExcerptB>\n{second['text']}\n</ExcerptB>"
        else:
            instruction = (
                "Write one answerable question about an important fact in the excerpt. "
                "Provide one short, exact, contiguous evidence quote from the excerpt."
            )
            context = f"<Excerpt>\n{first['text']}\n</Excerpt>"

        messages = [
            {
                "role": "system",
                "content": (
                    "Create a concise evaluation example using only the supplied text. "
                    "Do not use outside knowledge. Return a JSON object with exactly these fields: "
                    '{"question":"...","answer":"...","evidence_quotes":["..."]}. '
                    "For cross-excerpt questions, return exactly two quotes, one from each excerpt. "
                    "Copy quotes verbatim; keep each quote between 8 and 30 words."
                ),
            },
            {"role": "user", "content": f"{instruction}\n\n{context}"},
        ]
        raw = groq._call_groq(messages, is_json=True)
        if not raw or raw == "rate_limit_exceeded":
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        quotes = item.get("evidence_quotes", [])
        if isinstance(quotes, str):
            quotes = [quotes]
        if not question or not answer or not isinstance(quotes, list):
            continue
        expected_quote_count = 2 if cross_section else 1
        if len(quotes) != expected_quote_count:
            continue

        evidence_spans = []
        valid = True
        for quote_index, raw_quote in enumerate(quotes):
            quote = str(raw_quote).strip()
            if not quote:
                valid = False
                break
            # Keep evidence local to the excerpt from which it was requested.
            source_window = first if not cross_section or quote_index == 0 else second
            local_at = source_window["text"].find(quote)
            if local_at < 0:
                valid = False
                break
            char_start = source_window["start"] + local_at
            evidence_spans.append({
                "char_start": char_start,
                "char_end": char_start + len(quote),
                "quote": quote,
            })
        normalized_question = re.sub(r"\W+", " ", question.lower()).strip()
        if not valid or normalized_question in seen:
            continue
        seen.add(normalized_question)
        questions.append({
            "id": f"synthetic-{len(questions) + 1:03d}",
            "question": question,
            "answers": [answer],
            "answerable": True,
            "question_category": "cross_section_synthetic" if cross_section else "single_section_synthetic",
            "evidence_spans": evidence_spans,
            "question_source": "LLM-generated; not human-reviewed",
        })
        print(f"Created synthetic question {len(questions)}/{count} ({questions[-1]['question_category']}).")

    if len(questions) < min(count, 6):
        raise RuntimeError(
            f"Only {len(questions)} valid synthetic questions were produced after {attempts} attempts. "
            "Provide an annotated questions JSONL or retry after checking model availability."
        )
    return questions


def write_jsonl(items, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as output_file:
        for item in items:
            output_file.write(json.dumps(item, ensure_ascii=False) + "\n")


def verify_full_enrichment(chunk_path, community_path):
    """Refuse to label an interrupted partial graph as a full enriched build."""
    with open(chunk_path, "r", encoding="utf-8") as chunk_file:
        chunks = [json.loads(line) for line in chunk_file if line.strip()]
    incomplete_chunks = [
        chunk.get("id", "unknown")
        for chunk in chunks
        if not isinstance(chunk.get("entities"), list)
        or not isinstance(chunk.get("relations"), list)
        or not isinstance(chunk.get("summary"), str)
        or not chunk.get("summary", "").strip()
    ]
    with open(community_path, "r", encoding="utf-8") as community_file:
        communities = json.load(community_file)
    incomplete_communities = [
        cid for cid, community in communities.items()
        if not community.get("HL_summary") or not community.get("title")
    ]
    if incomplete_chunks or incomplete_communities:
        raise RuntimeError(
            "Graph enrichment is incomplete; refusing to report a full run. "
            f"Chunks missing extraction fields: {len(incomplete_chunks)}/{len(chunks)}; "
            f"communities missing summaries/titles: {len(incomplete_communities)}/{len(communities)}. "
            "Check results for fallback exhaustion, then resume/rebuild."
        )
    return chunks, communities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_TEXT, help="UTF-8 plain-text book file.")
    parser.add_argument("--questions", help="Annotated question JSONL; omit to make a synthetic pilot set.")
    parser.add_argument("--question-count", type=int, default=24, help="Synthetic QA count if no question file is supplied.")
    parser.add_argument("--artifact-dir", help="Defaults to data/<book-name>-graph.")
    parser.add_argument("--result", help="Defaults to results/<book-name>-text-results.json.")
    parser.add_argument("--skip-build", action="store_true", help="Reuse saved graph/index artifacts.")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Score retrieval and skip answer generation.",
    )
    parser.add_argument(
        "--context-char-limit",
        type=int,
        default=int(os.getenv("ANSWER_CONTEXT_CHAR_LIMIT", "24000")),
        help="Maximum characters of retrieved graph context sent to answer generation.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Book text does not exist: {args.input}")
    with open(args.input, "r", encoding="utf-8-sig") as source_file:
        source_text = source_file.read()
    if not source_text.strip():
        raise ValueError(f"Book text is empty (0 usable characters): {args.input}")

    book_name = os.path.splitext(os.path.basename(args.input))[0]
    artifact_dir = args.artifact_dir or os.path.join(ROOT, "data", f"{book_name}-graph")
    result_path = args.result or os.path.join(ROOT, "results", f"{book_name}-text-results.json")
    question_path = args.questions or os.path.join(artifact_dir, "eval_questions.generated.jsonl")
    chunk_path = os.path.join(artifact_dir, "chunked_output.jsonl")
    community_path = os.path.join(artifact_dir, "community_map.json")
    graph_path = os.path.join(artifact_dir, "graph.pkl")
    index_path = os.path.join(artifact_dir, "graph-index.bin")

    groq = GroqHandler()
    chunker = Chunker(store_file=chunk_path)
    generator = GraphAugmentedGenerator(
        groq,
        max_context_chars=args.context_char_limit,
    )
    builder = GraphBuilder(
        groq_handler=groq,
        jsonl_path=chunk_path,
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
        store_file=chunk_path,
        community_file=community_path,
        max_retries=2,
    )
    evaluator = NodeRAGEvaluator(chunker, builder, generator, entity_handler)

    try:
        with open(args.input, "rb") as source_file:
            source_sha256 = hashlib.sha256(source_file.read()).hexdigest()

        if args.questions:
            if not os.path.isfile(args.questions):
                raise FileNotFoundError(f"Question file does not exist: {args.questions}")
            question_path = args.questions
        elif os.path.isfile(question_path):
            with open(question_path, "r", encoding="utf-8") as questions_file:
                cached_questions = [json.loads(line) for line in questions_file if line.strip()]
            if not cached_questions or any(q.get("source_sha256") != source_sha256 for q in cached_questions):
                cached_questions = generate_synthetic_questions(source_text, groq, count=args.question_count)
                for question in cached_questions:
                    question["source_sha256"] = source_sha256
                write_jsonl(cached_questions, question_path)
                print(f"Replaced stale synthetic QA annotations at {question_path}")
        else:
            synthetic = generate_synthetic_questions(source_text, groq, count=args.question_count)
            for question in synthetic:
                question["source_sha256"] = source_sha256
            write_jsonl(synthetic, question_path)
            print(f"Saved {len(synthetic)} synthetic QA annotations to {question_path}")

        if not args.skip_build:
            os.makedirs(artifact_dir, exist_ok=True)
            evaluator.build_from_paragraphs(
                eval=False,
                text_path=args.input,
                artifact_dir=artifact_dir,
            )
            chunks, communities = verify_full_enrichment(chunk_path, community_path)
        else:
            if not all(os.path.isfile(path) for path in (chunk_path, graph_path, index_path)):
                raise FileNotFoundError("--skip-build requested but saved chunk/graph/index artifacts are incomplete.")
            chunks, communities = verify_full_enrichment(chunk_path, community_path)

        results = evaluator.evaluate_pdf_questions(
            questions_path=question_path,
            hnsw_index_path=index_path,
            graph_path=graph_path,
            retrieval_only=args.retrieval_only,
        )
        with open(args.input, "rb") as source_file:
            source_sha256 = hashlib.sha256(source_file.read()).hexdigest()
        results["run_metadata"] = {
            "source_text": os.path.relpath(args.input, ROOT),
            "source_sha256": source_sha256,
            "source_characters": len(source_text),
            "source_chunk_count": len(chunks),
            "community_count": len(communities),
            "index_mode": "repository full-text enrichment (T/S/N/R + community H/O); A attributes are not wired in",
            "evaluation_mode": "retrieval_only" if args.retrieval_only else "retrieval_and_generation",
            "answer_context_character_limit": args.context_char_limit,
            "evaluation_question_file": os.path.relpath(question_path, ROOT) if os.path.isabs(question_path) else question_path,
            "evaluation_questions_are_synthetic": not bool(args.questions),
            "evaluation_warning": (
                "Synthetic questions/evidence were generated by an LLM and were not human-reviewed. "
                "Treat as a pilot, not a held-out benchmark result."
                if not args.questions else None
            ),
            "successful_api_requests": groq.successful_requests,
            "api_token_usage": groq.total_usage,
            "llm_runtime": groq.get_runtime_metadata(),
        }
        result_parent = os.path.dirname(result_path)
        if result_parent:
            os.makedirs(result_parent, exist_ok=True)
        with open(result_path, "w", encoding="utf-8") as result_file:
            json.dump(results, result_file, indent=2, ensure_ascii=False)
        print("Evaluation summary:")
        print(json.dumps(results.get("summary", {}), indent=2))
        print(f"Detailed results: {result_path}")
    finally:
        builder.close()


if __name__ == "__main__":
    main()
