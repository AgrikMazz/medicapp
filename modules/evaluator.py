import os
import re
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
nltk.download('punkt_tab')
from rouge import Rouge
import torch
import json
import string
from tqdm import tqdm
from transformers import GPT2TokenizerFast
from chunker import Chunker
from graph_builder import GraphBuilder
from generator import GraphAugmentedGenerator
from entity_handler import EntityHandler
from gnn_models import NodeAlignerGNN

pdf_path="docs\medbook1\medical_book.pdf"

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

class NodeRAGEvaluator:
    def __init__(self, chunker: Chunker, builder: GraphBuilder, generator: GraphAugmentedGenerator, entity_handler: EntityHandler, tokenizer: GPT2TokenizerFast):
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
        common = set(pred_tokens) & set(gt_tokens)
        if not common:
            return 0, 0, 0
        prec = len(common) / len(pred_tokens)
        rec = len(common) / len(gt_tokens)
        f1 = 2 * prec * rec / (prec + rec)
        return f1, prec, rec

    def compute_exact(self, prediction, ground_truth):
        return int(self.normalize_answer(prediction) == self.normalize_answer(ground_truth))

    def compute_substring_match(self, prediction, ground_truth):
        return int(self.normalize_answer(ground_truth) in self.normalize_answer(prediction))

    def get_metric_score(self, prediction, references):
        if not references:
            return {}

        prediction = prediction.strip()
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
                best_meteor = max(best_meteor, meteor_score([ref], prediction))
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
            "accuracy": best_acc,
            "bleu_1": best_bleu1,
            "bleu_4": best_bleu4,
            "bleu_1_smooth": best_bleu1_smooth,
            "bleu_4_smooth": best_bleu4_smooth,
            "meteor": best_meteor,
            "rouge_l": best_rouge_l
        }

    def build_from_paragraphs(self, eval: bool = True, dataset_name: str = None, dataset_path: str = None, pdf_path: str = None, max_samples=100):
        suffix = "-eval" if eval else ""

        if eval:
            dataset_name = dataset_name.lower()
            if dataset_name == "musique":
                examples = self.load_musique(dataset_path, max_samples)
            else:
                raise NotImplementedError(f"Dataset '{dataset_name}' not yet supported.")

            for example in examples:
                self.chunker.segment_into_chunks(text=example["context"])
        else:
            text = self.chunker.pdf_to_text(path=pdf_path)
            chunks = self.chunker.segment_into_chunks(text=text)

        self.entity_handler.process_documents(type="entity")
        self.entity_handler.retry_stored_unprocessed_documents(type="entity")
        self.entity_handler.process_documents(type="relationship")
        self.entity_handler.retry_stored_unprocessed_documents(type="relationship")
        self.entity_handler.process_documents(type="summary")
        self.entity_handler.retry_stored_unprocessed_documents(type="summary")

        self.builder.build_graph()
        self.builder.detect_and_save_communities(output_path=f"data/community_map{suffix}.json")
        self.builder.save_incomplete_entries(f"data/incomplete_graph_elements{suffix}.json")
        self.builder.save_pickle(f"graph{suffix}.pkl")

        self.entity_handler.process_documents(type="HL_summary")
        self.entity_handler.retry_stored_unprocessed_documents(type="HL_summary")
        self.entity_handler.process_documents(type="title")
        self.entity_handler.retry_stored_unprocessed_documents(type="title")

        self.builder.add_HO_nodes_from_community_map(community_map_path=f"data/community_map{suffix}.json")
        self.builder.finalize_graph_G3_from_community_map(community_map_path=f"data/community_map{suffix}.json")
        self.builder.build_G4_text_attachment(chunk_file=f"data/chunked_output{suffix}.jsonl")
        self.builder.build_G5_semantic_hnsw_index(index_path=f"data/graph{suffix}-index.bin")
        self.builder.save_pickle(f"graph{suffix}.pkl")

        print(f"✅ build_from_paragraphs complete (G5 ready, eval={eval}).")

    def get_single_answer(self, query: str, hnsw_index_path="data\graph-index.bin", graph_path="graph.pkl"):
        self.builder.load_graph_pickle(graph_path)

        entry_nodes, p_weights = self.builder.get_entry_points(query=query, hnsw_index_path=hnsw_index_path, top_k=10)
        G_raw, _ = self.builder.run_shallow_ppr(entry_nodes, p_weights)
        rc = self.builder.reasoning_chain(query, G_raw)
        print(rc)

        model = NodeAlignerGNN(in_dim=391, hidden_dim=128, out_dim=256)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        G_aligned, node_embeds, _ = self.builder.align_nodes_via_gnn(query, G_raw, rc, model, optimizer, device="cpu")

        #r_g, r_s, sim = self.builder.compute_graph_embedding(G_aligned, rc, node_embeds)
        pred_answer = self.generator.generate_answer(query, G_aligned)

        print(pred_answer)

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

        model = NodeAlignerGNN(in_dim=391, hidden_dim=128, out_dim=256)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        if type == "entry_points":
            for item in tqdm(examples, desc=f"🔍 Evaluating {dataset_name} ({type})"):
                query = item["question"]
                gold_answer = item["answers"]
                uid = item.get("id")
                aliases = item.get("aliases", [])
                aliases.append(gold_answer)

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
                    rc = sample.get("reasoning_chain", "")
                    G_aligned, node_embeds, _ = self.builder.align_nodes_via_gnn(
                        query, G_raw, rc, model, optimizer, device="cpu"
                    )
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

                total_tokens = len(self.tokenizer.encode(prompt)) + len(self.tokenizer.encode(pred_answer))
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

    def aggregate(self, prediction_logs):
        summary = {}
        if not prediction_logs:
            return {"error": "No predictions"}

        keys = [k for k in prediction_logs[0].keys() if isinstance(prediction_logs[0][k], (float, int)) and k != "tokens"]
        for k in keys:
            summary[k] = sum(p[k] for p in prediction_logs) / len(prediction_logs)

        summary["avg_tokens"] = sum(p["tokens"] for p in prediction_logs) / len(prediction_logs)
        return {
            "summary": summary,
            "predictions": prediction_logs
        }

    def save_results(self, result_dict, path="results/noderag_eval.json"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(result_dict, f, indent=2)
        print(f"📁 Saved results to {path}")

    def save_sample(self, json_data, path="data/eval-dump.json"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(json_data, f, indent=2)