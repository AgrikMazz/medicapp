import os
import re
import time
import math
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
nltk.download('punkt_tab')
nltk.download('wordnet', quiet=True)
from rouge import Rouge
import json
import string
from collections import Counter
from tqdm import tqdm
from .chunker import Chunker
from .graph_builder import GraphBuilder
from .generator import GraphAugmentedGenerator
from .entity_handler import EntityHandler

class NodeRAGEvaluator:
    def __init__(self, chunker: Chunker, builder: GraphBuilder, generator: GraphAugmentedGenerator, entity_handler: EntityHandler, tokenizer=None):
        self.chunker = chunker
        self.builder = builder
        self.generator = generator
        self.entity_handler = entity_handler
        self.tokenizer = tokenizer
    
    def load_musique(self, path, max_samples=None):
        examples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if not item.get("answerable", True):
                    continue

                id = item["id"]
                question = item["question"]
                answer_main = item["answer"]
                aliases = item.get("answer_aliases", [])
                answers = list({a.strip() for a in [answer_main] + aliases if a.strip()})

                paragraphs = item.get("paragraphs", [])
                full_context = " ".join(p["paragraph_text"].strip() for p in paragraphs if p.get("paragraph_text"))

                examples.append({
                    "id": id,
                    "question": question,
                    "answers": answers,
                    "context": full_context
                })

                if max_samples and len(examples) >= max_samples:
                    break

        return examples

    def normalize_answer(self, s):
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            return ''.join(ch for ch in text if ch not in string.punctuation)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def compute_f1(self, prediction, ground_truth):
        pred_tokens = self.normalize_answer(prediction).split()
        gt_tokens = self.normalize_answer(ground_truth).split()
        common = Counter(pred_tokens) & Counter(gt_tokens)
        overlap = sum(common.values())
        if not overlap:
            return 0, 0, 0
        prec = overlap / len(pred_tokens)
        rec = overlap / len(gt_tokens)
        f1 = 2 * prec * rec / (prec + rec)
        return f1, prec, rec

    def compute_exact(self, prediction, ground_truth):
        return int(self.normalize_answer(prediction) == self.normalize_answer(ground_truth))

    def compute_substring_match(self, prediction, ground_truth):
        return int(self.normalize_answer(ground_truth) in self.normalize_answer(prediction))

    def get_metric_score(self, prediction, references):
        if not references:
            return {}

        prediction = re.sub(
            r"(?:\[[^\]]+\]|\u3010[^\u3011]*\u3011)",
            " ",
            prediction.strip(),
            flags=re.IGNORECASE,
        )
        if references and isinstance(references[0], list):
            references = references[0]
        references = [r.strip() for r in references if isinstance(r, str) and r.strip()]
        best_f1, best_prec, best_rec = 0, 0, 0
        best_em = 0
        best_acc = 0
        best_bleu1, best_bleu4 = 0, 0
        best_bleu1_smooth, best_bleu4_smooth = 0, 0
        best_meteor = 0
        best_rouge_l = 0

        smoothing = SmoothingFunction()
        rouge = Rouge()

        for ref in references:
            f1, prec, rec = self.compute_f1(prediction, ref)
            best_f1 = max(best_f1, f1)
            best_prec = max(best_prec, prec)
            best_rec = max(best_rec, rec)

            em = self.compute_exact(prediction, ref)
            best_em = max(best_em, em)

            acc = self.compute_substring_match(prediction, ref)
            best_acc = max(best_acc, acc)

            try:
                bleu1 = sentence_bleu([ref.split()], prediction.split(), weights=(1, 0, 0, 0))
                bleu4 = sentence_bleu([ref.split()], prediction.split(), weights=(0.25, 0.25, 0.25, 0.25))
                bleu1_smooth = sentence_bleu([ref.split()], prediction.split(), weights=(1, 0, 0, 0), smoothing_function=smoothing.method1)
                bleu4_smooth = sentence_bleu([ref.split()], prediction.split(), weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing.method1)
            except:
                bleu1 = bleu4 = bleu1_smooth = bleu4_smooth = 0

            best_bleu1 = max(best_bleu1, bleu1)
            best_bleu4 = max(best_bleu4, bleu4)
            best_bleu1_smooth = max(best_bleu1_smooth, bleu1_smooth)
            best_bleu4_smooth = max(best_bleu4_smooth, bleu4_smooth)

            try:
                best_meteor = max(best_meteor, meteor_score([ref.split()], prediction.split()))
            except:
                pass

            try:
                rouge_scores = rouge.get_scores(prediction, ref)
                best_rouge_l = max(best_rouge_l, rouge_scores[0]["rouge-l"]["f"])
            except:
                pass

        return {
            "f1": best_f1,
            "precision": best_prec,
            "recall": best_rec,
            "exact_match": best_em,
            "bleu_1": best_bleu1,
            "bleu_4": best_bleu4,
            "bleu_1_smooth": best_bleu1_smooth,
            "bleu_4_smooth": best_bleu4_smooth,
            "meteor": best_meteor,
            "rouge_l": best_rouge_l
        }

    def build_from_paragraphs(
        self,
        eval: bool = True,
        dataset_name: str = None,
        dataset_path: str = None,
        pdf_path: str = None,
        max_samples=100,
        page_numbers=None,
        artifact_dir=None,
        text_path=None,
    ):
        suffix = "-eval" if eval else ""
        chunks = []
        chunk_root, chunk_extension = os.path.splitext(self.chunker.store_file)
        if chunk_root.endswith("-eval"):
            chunk_root = chunk_root[:-len("-eval")]
        if artifact_dir:
            os.makedirs(artifact_dir, exist_ok=True)
            chunk_root = os.path.join(artifact_dir, os.path.basename(chunk_root))
            community_path = os.path.join(artifact_dir, f"community_map{suffix}.json")
            graph_path = os.path.join(artifact_dir, f"graph{suffix}.pkl")
            index_path = os.path.join(artifact_dir, f"graph{suffix}-index.bin")
            incomplete_path = os.path.join(artifact_dir, f"incomplete_graph_elements{suffix}.json")
            self.entity_handler.progress_file = os.path.join(artifact_dir, "progress.json")
        else:
            community_path = f"data/community_map{suffix}.json"
            graph_path = f"graph{suffix}.pkl"
            index_path = f"data/graph{suffix}-index.bin"
            incomplete_path = f"data/incomplete_graph_elements{suffix}.json"
        chunk_path = f"{chunk_root}{suffix}{chunk_extension or '.jsonl'}"
        self.chunker.store_file = chunk_path
        self.entity_handler.store_file = chunk_path
        self.entity_handler.community_file = community_path
        self.builder.jsonl_path = chunk_path

        if eval:
            dataset_name = dataset_name.lower()
            if dataset_name == "musique":
                examples = self.load_musique(dataset_path, max_samples)
            else:
                raise NotImplementedError(f"Dataset '{dataset_name}' not yet supported.")

            for example in examples:
                chunks.extend(self.chunker.segment_into_chunks(
                    text=example["context"], source_id=f"{dataset_name}:{example['id']}", persist=False
                ))
        elif pdf_path:
            source_id = os.path.splitext(os.path.basename(pdf_path))[0]
            pages = self.chunker.pdf_to_pages(path=pdf_path)
            if page_numbers is not None:
                selected_pages = {int(page) for page in page_numbers}
                pages = [page for page in pages if page["page"] in selected_pages]
                found_pages = {page["page"] for page in pages}
                missing_pages = selected_pages - found_pages
                if missing_pages:
                    raise ValueError(f"Requested PDF pages do not exist: {sorted(missing_pages)}")
            chunks = self.chunker.segment_into_chunks(text=pages, source_id=source_id, persist=False)
        elif text_path:
            if not os.path.isfile(text_path):
                raise FileNotFoundError(f"Text source does not exist: {text_path}")
            with open(text_path, "r", encoding="utf-8-sig") as source_file:
                source_text = source_file.read()
            if not source_text.strip():
                raise ValueError(f"Text source is empty: {text_path}")
            source_id = os.path.splitext(os.path.basename(text_path))[0]
            chunks = self.chunker.segment_into_chunks(
                text=source_text,
                source_id=source_id,
                persist=False,
            )
        else:
            raise ValueError("Provide pdf_path or text_path when eval=False.")

        self.chunker.write_chunks(chunks)
        self.builder.G.clear()

        self.entity_handler.process_documents(type="entity")
        self.entity_handler.retry_stored_unprocessed_documents(type="entity")
        self.entity_handler.process_documents(type="relationship")
        self.entity_handler.retry_stored_unprocessed_documents(type="relationship")
        self.entity_handler.process_documents(type="summary")
        self.entity_handler.retry_stored_unprocessed_documents(type="summary")

        self.builder.build_graph()
        self.builder.detect_and_save_communities(output_path=community_path)
        self.builder.save_incomplete_entries(incomplete_path)
        self.builder.save_pickle(graph_path)

        self.entity_handler.process_documents(type="HL_summary")
        self.entity_handler.retry_stored_unprocessed_documents(type="HL_summary")
        self.entity_handler.process_documents(type="title")
        self.entity_handler.retry_stored_unprocessed_documents(type="title")

        self.builder.add_HO_nodes_from_community_map(community_map_path=community_path)
        self.builder.finalize_graph_G3_from_community_map(community_map_path=community_path)
        self.builder.build_G4_text_attachment(chunk_file=chunk_path)
        self.builder.build_G5_semantic_hnsw_index(index_path=index_path)
        self.builder.save_pickle(graph_path)

        print(f"✅ build_from_paragraphs complete (G5 ready, eval={eval}).")

    def get_single_answer(
        self,
        query: str,
        hnsw_index_path="data/graph-index.bin",
        graph_path="graph.pkl",
        retrieval_method="hnsw_ppr_crossencoder",
        candidate_k=40,
        context_token_budget=4096,
        hnsw_candidate_k=20,
        graph_expand_k=20,
        ppr_alpha=0.5,
    ):
        self.builder.load_graph_pickle(graph_path)
        reranker = (
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
            if "crossencoder" in retrieval_method else None
        )
        retriever = self.builder.get_passage_retriever(reranker)
        ranked = retriever.rank(
            query,
            method=retrieval_method,
            candidate_k=candidate_k,
            hnsw_index_path=hnsw_index_path,
            hnsw_candidate_k=hnsw_candidate_k,
            graph_expand_k=graph_expand_k,
            hnsw_ppr_alpha=ppr_alpha,
        )
        passages = [
            retriever.passage_context_record(node_id, score)
            for node_id, score in ranked
        ]
        context = self.generator.textualize_ranked_passages(
            query, passages, token_budget=context_token_budget
        )
        evidence_graph = self.builder.G.subgraph(
            [node_id for node_id, _ in ranked]
        ).copy()
        return self.generator.generate_answer(
            query, evidence_graph, textual_context=context
        )

    def evaluate(self, dataset_name: str, dataset_path, hnsw_index_path, max_samples=100,
                dump_path="data/eval_dump.json", type="entry_points"):
        
        dataset_name = dataset_name.lower()
        
        if type == "entry_points":
            if dataset_name == "musique":
                examples = self.load_musique(dataset_path, max_samples)
            else:
                raise NotImplementedError(f"Dataset '{dataset_name}' not supported.")
        else:
            examples = None

        self.builder.load_graph_pickle("graph-eval.pkl")

        if os.path.exists(dump_path):
            with open(dump_path, "r", encoding="utf-8") as f:
                uid_to_item = json.load(f)
        else:
            uid_to_item = {}

        if type == "entry_points":
            for item in tqdm(examples, desc=f"🔍 Evaluating {dataset_name} ({type})"):
                query = item["question"]
                gold_answer = item["answers"]
                uid = item.get("id")
                aliases = list(gold_answer)

                if uid not in uid_to_item:
                    uid_to_item[uid] = {
                        "id": uid,
                        "query": query,
                        "answers": aliases
                    }

                try:
                    entry_nodes, p_weights = self.builder.get_entry_points(query, hnsw_index_path, top_k=10)
                    G_raw, _ = self.builder.run_shallow_ppr(entry_nodes, p_weights)
                    uid_to_item[uid]["G_raw"] = self.builder.graph_to_json(G_raw)
                except Exception as e:
                    print(f"❌ Entry point or PPR failed: {uid} → {e}")
                    continue

            with open(dump_path, "w", encoding="utf-8") as f:
                json.dump(uid_to_item, f, indent=2)
            return None

        for uid, sample in tqdm(uid_to_item.items(), desc=f"🔧 Running step: {type}"):
            query = sample["query"]

            if type == "align":
                try:
                    G_raw = self.builder.json_to_graph(sample["G_raw"])
                    G_aligned = self.builder.prune_subgraph(query, G_raw)
                    sample["G_aligned"] = self.builder.graph_to_json(G_aligned)
                except Exception as e:
                    print(f"⚠️ Alignment failed: {uid} → {e}")
                    continue

            elif type == "prompt":
                try:
                    G_aligned = self.builder.json_to_graph(sample["G_aligned"])
                    sample["prompt"] = self.generator.build_prompt(query, G_aligned)
                except Exception as e:
                    print(f"⚠️ Prompt generation failed: {uid} → {e}")
                    continue

        if type == "reasoning_chain":
            self.entity_handler.process_documents("reasoning_chain")

        if type == "answer":
            self.entity_handler.process_documents("answer")

        if type in {"align", "prompt"}:
            with open(dump_path, "w", encoding="utf-8") as f:
                json.dump(uid_to_item, f, indent=2)

        if type == "eval":
            predictions = []
            for uid, sample in uid_to_item.items():
                pred_answer = sample.get("answer", "")
                prompt = sample.get("prompt", "")
                gold_answer = sample["answers"]
                query = sample["query"]

                if self.tokenizer is not None:
                    total_tokens = len(self.tokenizer.encode(prompt)) + len(self.tokenizer.encode(pred_answer))
                else:
                    total_tokens = len((prompt + " " + pred_answer).split())
                score = self.get_metric_score(pred_answer, gold_answer)

                score.update({
                    "tokens": total_tokens,
                    "question": query,
                    "prediction": pred_answer,
                    "gold": gold_answer
                })

                predictions.append(score)

            print("✅ Final evaluation complete.")
            return self.aggregate(predictions)

        return None

    def evaluate_pdf_questions(
        self,
        questions_path,
        hnsw_index_path,
        graph_path="graph.pkl",
        max_samples=None,
        retrieval_only=False,
        generation_contexts=None,
        max_generation_retries=2,
    ):
        """Evaluate retrieval and answers against annotations for the indexed PDF.

        JSONL records should include ``question``, optional ``answers`` and
        ``evidence``/``evidence_pages``. Each evidence item may contain a page
        number or page_start/page_end. The PDF itself remains the indexed corpus.
        """
        self.builder.load_graph_pickle(graph_path)
        examples = []
        with open(questions_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
                if max_samples and len(examples) >= max_samples:
                    break

        prediction_logs = []
        for index, item in enumerate(tqdm(examples, desc="Evaluating PDF questions")):
            question = item["question"]
            entry_nodes, weights = self.builder.get_entry_points(question, hnsw_index_path, top_k=10)
            G_raw, _ = self.builder.run_shallow_ppr(entry_nodes, weights)
            G_final = self.builder.prune_subgraph(question, G_raw)

            gold_pages = set()
            for page in item.get("evidence_pages", []):
                try:
                    gold_pages.add(int(page))
                except (TypeError, ValueError):
                    continue
            for evidence in item.get("evidence", []):
                if isinstance(evidence, int):
                    gold_pages.add(evidence)
                    continue
                if not isinstance(evidence, dict):
                    continue
                if evidence.get("page") is not None:
                    try:
                        gold_pages.add(int(evidence["page"]))
                    except (TypeError, ValueError):
                        pass
                start = evidence.get("page_start")
                end = evidence.get("page_end", start)
                if start is not None:
                    gold_pages.update(range(int(start), int(end or start) + 1))

            retrieved_pages = set()
            for _, node in G_raw.nodes(data=True):
                if node.get("type") != "T":
                    continue
                retrieved_pages.update(node.get("source_pages") or [])
                start = node.get("page_start")
                end = node.get("page_end", start)
                if start is not None:
                    retrieved_pages.update(range(int(start), int(end or start) + 1))

            final_context_pages = set()
            for _, node in G_final.nodes(data=True):
                if node.get("type") != "T":
                    continue
                final_context_pages.update(node.get("source_pages") or [])
                start = node.get("page_start")
                end = node.get("page_end", start)
                if start is not None:
                    final_context_pages.update(range(int(start), int(end or start) + 1))

            score = {}
            answers = item.get("answers", item.get("answer", []))
            if isinstance(answers, str):
                answers = [answers]

            # Plain-text books can carry exact source-character evidence spans
            # instead of PDF page labels. Score both the retrieved candidate set
            # and the final context after graph pruning.
            gold_spans = []
            for evidence_span in item.get("evidence_spans", []):
                if not isinstance(evidence_span, dict):
                    continue
                start = evidence_span.get("char_start", evidence_span.get("start"))
                end = evidence_span.get("char_end", evidence_span.get("end"))
                try:
                    start, end = int(start), int(end)
                except (TypeError, ValueError):
                    continue
                if end > start:
                    gold_spans.append((start, end))

            def passage_intervals(graph):
                intervals = []
                for _, node in graph.nodes(data=True):
                    if node.get("type") != "T":
                        continue
                    try:
                        start = int(node.get("char_start"))
                        end = int(node.get("char_end"))
                    except (TypeError, ValueError):
                        continue
                    if end > start:
                        intervals.append((start, end))
                return sorted(intervals)

            def passage_records(graph, score_key):
                records = []
                for node_id, node in graph.nodes(data=True):
                    if node.get("type") != "T":
                        continue
                    try:
                        start = int(node.get("char_start"))
                        end = int(node.get("char_end"))
                    except (TypeError, ValueError):
                        continue
                    if end <= start:
                        continue
                    try:
                        rank_score = float(node.get(score_key, 0.0))
                    except (TypeError, ValueError):
                        rank_score = 0.0
                    records.append({"id": node_id, "start": start, "end": end, "score": rank_score})
                return sorted(records, key=lambda record: record["score"], reverse=True)

            def record_relevance(record):
                return sum(
                    record["start"] < span_end and record["end"] > span_start
                    for span_start, span_end in gold_spans
                )

            def span_recall_at(records, k):
                intervals = sorted((record["start"], record["end"]) for record in records[:k])
                hits = sum(covered(start, end, intervals) for start, end in gold_spans)
                return hits / len(gold_spans) if gold_spans else 0.0

            def rank_metrics(records, corpus_records, prefix):
                relevances = [record_relevance(record) for record in records]
                first_relevant_rank = next(
                    (rank for rank, relevance in enumerate(relevances, start=1) if relevance > 0),
                    None,
                )
                score[f"{prefix}_mrr"] = 1.0 / first_relevant_rank if first_relevant_rank else 0.0

                def dcg(values):
                    return sum(
                        (2 ** relevance - 1) / math.log2(rank + 1)
                        for rank, relevance in enumerate(values[:10], start=1)
                    )

                ideal_relevances = sorted(
                    (record_relevance(record) for record in corpus_records), reverse=True
                )
                ideal_dcg = dcg(ideal_relevances)
                score[f"{prefix}_ndcg@10"] = dcg(relevances) / ideal_dcg if ideal_dcg else 0.0
                for k in (1, 5, 10):
                    score[f"{prefix}_evidence_span_recall@{k}"] = span_recall_at(records, k)

            def covered(start, end, intervals):
                cursor = start
                for interval_start, interval_end in intervals:
                    if interval_end <= cursor:
                        continue
                    if interval_start > cursor:
                        return False
                    cursor = max(cursor, interval_end)
                    if cursor >= end:
                        return True
                return cursor >= end

            if gold_spans:
                raw_intervals = passage_intervals(G_raw)
                final_intervals = passage_intervals(G_final)
                raw_hits = sum(covered(start, end, raw_intervals) for start, end in gold_spans)
                final_hits = sum(covered(start, end, final_intervals) for start, end in gold_spans)
                score["retrieved_evidence_span_recall"] = raw_hits / len(gold_spans)
                score["final_context_evidence_span_recall"] = final_hits / len(gold_spans)
                score["all_evidence_spans_retrieved"] = float(raw_hits == len(gold_spans))
                score["all_evidence_spans_in_final_context"] = float(final_hits == len(gold_spans))
                score["gold_evidence_span_count"] = len(gold_spans)

                raw_records = passage_records(G_raw, "ppr_score")
                final_records = passage_records(G_final, "retrieval_score")
                corpus_records = passage_records(self.builder.G, "ppr_score")
                rank_metrics(raw_records, corpus_records, "retrieval")
                rank_metrics(final_records, corpus_records, "final_context")

                corpus_passage_count = len(corpus_records)
                bm25_scored = self.builder._lexical_candidates(
                    question,
                    top_k=corpus_passage_count,
                    node_types=("T",),
                    return_scores=True,
                )
                bm25_records = []
                for node_id, bm25_score in bm25_scored:
                    node = self.builder.G.nodes[node_id]
                    try:
                        start, end = int(node.get("char_start")), int(node.get("char_end"))
                    except (TypeError, ValueError):
                        continue
                    if end > start:
                        bm25_records.append({"id": node_id, "start": start, "end": end, "score": bm25_score})
                rank_metrics(bm25_records, corpus_records, "bm25")

            if gold_pages:
                hit_pages = gold_pages & retrieved_pages
                score["evidence_page_recall"] = len(hit_pages) / len(gold_pages)
                score["all_evidence_pages_retrieved"] = float(gold_pages.issubset(retrieved_pages))
                context_hits = gold_pages & final_context_pages
                score["final_context_page_recall"] = len(context_hits) / len(gold_pages)
                score["all_evidence_pages_in_final_context"] = float(gold_pages.issubset(final_context_pages))

            generation_context = None
            if generation_contexts is not None:
                generation_context = generation_contexts.get(question)
            if generation_context is None and hasattr(self.generator, "textualize_graph"):
                generation_context = self.generator.textualize_graph(G_final)

            if gold_spans and generation_context is not None:
                evidence_quotes = [
                    re.sub(r"\s+", " ", str(span.get("quote", "")).casefold()).strip()
                    for span in item.get("evidence_spans", [])
                    if isinstance(span, dict) and span.get("quote")
                ]
                if evidence_quotes:
                    normalized_context = re.sub(r"\s+", " ", str(generation_context).casefold())
                    serialized_hits = sum(quote in normalized_context for quote in evidence_quotes)
                    score["serialized_context_evidence_span_recall"] = serialized_hits / len(evidence_quotes)
                    score["all_evidence_spans_serialized"] = float(serialized_hits == len(evidence_quotes))

            prediction = ""
            generation_status = "not_run" if retrieval_only else "ok"
            generation_attempts = 0
            if not retrieval_only:
                raw_prediction = None
                for attempt in range(max_generation_retries + 1):
                    generation_attempts = attempt + 1
                    if generation_context is not None and isinstance(self.generator, GraphAugmentedGenerator):
                        raw_prediction = self.generator.generate_answer(
                            question,
                            G_final,
                            textual_context=generation_context,
                        )
                    else:
                        raw_prediction = self.generator.generate_answer(question, G_final)

                    if raw_prediction == "rate_limit_exceeded" and attempt < max_generation_retries:
                        groq = getattr(self.generator, "groq_handler", None)
                        delay = 10.0
                        if groq is not None and hasattr(groq, "_retry_delay"):
                            delay = groq._retry_delay(getattr(groq, "last_rate_limit_info", {}))
                        print(f"Generation rate-limited; retrying question after {delay:g}s ({attempt + 1}/{max_generation_retries}).")
                        time.sleep(delay)
                        continue
                    if raw_prediction is None and attempt < max_generation_retries:
                        print(f"Generation request failed; retrying question ({attempt + 1}/{max_generation_retries}).")
                        time.sleep(2.0)
                        continue
                    break

                if raw_prediction == "rate_limit_exceeded":
                    generation_status = "rate_limited"
                elif raw_prediction is None:
                    generation_status = "failed"
                else:
                    prediction = str(raw_prediction)

                if answers and generation_status == "ok":
                    score.update(self.get_metric_score(prediction, answers))

            if generation_context is not None:
                score["generation_context"] = generation_context
            score["retrieved_passage_node_ids"] = sorted(
                str(node_id)
                for node_id, data in G_raw.nodes(data=True)
                if data.get("type") == "T"
            )
            score["final_context_passage_node_ids"] = sorted(
                str(node_id)
                for node_id, data in G_final.nodes(data=True)
                if data.get("type") == "T"
            )
            if gold_spans:
                score["bm25_passage_node_ids"] = [str(record["id"]) for record in bm25_records]

            if gold_pages and generation_status == "ok" and not retrieval_only:
                cited_pages = set()
                citation_pattern = (
                    r"\[(?:p|pp)\.?\s*([0-9,\s\-–—]+)\]"
                    r"|【[^】]*?\b(?:p|pp)\.?\s*([0-9,\s\-–—]+)[^】]*】"
                )
                for citation_match in re.finditer(citation_pattern, prediction, flags=re.IGNORECASE):
                    citation = citation_match.group(1) or citation_match.group(2)
                    for part in citation.split(","):
                        part = part.strip()
                        range_match = re.fullmatch(r"(\d+)\s*[-–—]\s*(\d+)", part)
                        if range_match:
                            cited_pages.update(range(int(range_match.group(1)), int(range_match.group(2)) + 1))
                        elif part.isdigit():
                            cited_pages.add(int(part))
                score["citation_page_precision"] = (
                    len(cited_pages & gold_pages) / len(cited_pages) if cited_pages else 0.0
                )
                score["citation_page_recall"] = len(cited_pages & gold_pages) / len(gold_pages)

            if item.get("answerable") is False and generation_status == "ok" and not retrieval_only:
                score["abstained"] = float(bool(re.search(
                    r"\b(insufficient|not enough evidence|cannot determine|cannot answer|do not know|don't know|"
                    r"do not contain|does not contain|not in the provided passages|not available in the passages)\b",
                    prediction,
                    flags=re.IGNORECASE,
                )))

            score.update({
                "id": item.get("id", str(index)),
                "question": question,
                "question_category": item.get("question_category"),
                "question_source": item.get("question_source"),
                "prediction": prediction,
                "answers": answers,
                "gold_pages": sorted(gold_pages),
                "retrieved_pages": sorted(retrieved_pages),
                "final_context_pages": sorted(final_context_pages),
                "generation_status": generation_status,
                "generation_attempts": generation_attempts,
            })
            prediction_logs.append(score)

        return self.aggregate(prediction_logs)

    def aggregate(self, prediction_logs):
        summary = {}
        if not prediction_logs:
            return {"error": "No predictions"}

        keys = sorted({
            key
            for prediction in prediction_logs
            for key, value in prediction.items()
            if isinstance(value, (float, int)) and key != "tokens"
        })
        for k in keys:
            values = [p[k] for p in prediction_logs if k in p]
            summary[k] = sum(values) / len(values)

        token_values = [p["tokens"] for p in prediction_logs if "tokens" in p]
        if token_values:
            summary["avg_tokens"] = sum(token_values) / len(token_values)
        return {
            "summary": summary,
            "predictions": prediction_logs
        }

    def save_results(self, result_dict, path="results/noderag_eval.json"):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w") as f:
            json.dump(result_dict, f, indent=2)
        print(f"📁 Saved results to {path}")

    def save_sample(self, json_data, path="data/eval-dump.json"):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w") as f:
            json.dump(json_data, f, indent=2)
