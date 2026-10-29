import os
import time
import json
import jsonlines
from .groq_handler import GroqHandler
from .graph_builder import GraphBuilder
from .generator import GraphAugmentedGenerator

class EntityHandler:
    def __init__(self, groq_handler: GroqHandler, graph_builder: GraphBuilder, generator: GraphAugmentedGenerator, batch_size=8, wait_time=60,
                 progress_file="data/progress.json",
                 failed_file="data/permanent_failed.json",
                 store_file="data/chunked_output.jsonl",
                 community_file="data/community_map.json",
                 progress_file_name="ProgressCheckpoint",
                 eval_dump_file="data/eval_dump.json",
                 max_retries=3):
        self.groq_handler = groq_handler
        self.graph_builder = graph_builder
        self.generator = generator
        self.batch_size = batch_size
        self.wait_time = wait_time
        self.progress_file = progress_file
        self.failed_file = failed_file
        self.store_file = store_file
        self.community_file = community_file
        self.progress_file_name = progress_file_name
        self.eval_dump_file = eval_dump_file
        self.max_retries = max_retries
        self.processed_count = 0
        self.Entity_Unprocessed_elements = []

    def _checkpoint_paths(self, type, max_retries):
        directory = os.path.dirname(self.progress_file) or "."
        os.makedirs(directory, exist_ok=True)
        return [
            os.path.join(directory, f"{self.progress_file_name}-{type}-{index}.json")
            for index in range(1, max_retries + 1)
        ]

    def process_documents(self, type: str, start_index=0):
        self.load_progress(type)

        if type in {"reasoning_chain", "answer"}:
            data_file = self.eval_dump_file
            try:
                with open(data_file, "r", encoding="utf-8") as f:
                    json_data = json.load(f)

                if not json_data:
                    print("No data found in the eval_dump JSON.")
                    return

                doc_ids = list(json_data.keys())
                total_docs = len(doc_ids)

                for i in range(start_index, total_docs, self.batch_size):
                    batch_doc_ids = doc_ids[i : i + self.batch_size]
                    batch_processed = False

                    for doc_id in batch_doc_ids:
                        content = json_data[doc_id]
                        try:
                            query = content.get("query")
                            if not query:
                                print(f"⚠️ No query found in {doc_id}")
                                self.Entity_Unprocessed_elements.append({"doc_id": doc_id, "type": type})
                                continue

                            result_key = "reasoning_chain" if type == "reasoning_chain" else "answer"
                            if content.get(result_key):
                                continue

                            if type == "reasoning_chain":
                                G_raw = self.graph_builder.json_to_graph(content.get("G_raw", {}))
                                if not G_raw:
                                    print(f"⚠️ No G_raw graph found in {doc_id}")
                                    self.Entity_Unprocessed_elements.append({"doc_id": doc_id, "type": type})
                                    continue
                                result = self.graph_builder.reasoning_chain(query, G_raw)
                                key = "reasoning_chain"

                            elif type == "answer":
                                G_aligned = self.graph_builder.json_to_graph(content.get("G_aligned", {}))
                                if not G_aligned:
                                    print(f"⚠️ No G_aligned graph found in {doc_id}")
                                    self.Entity_Unprocessed_elements.append({"doc_id": doc_id, "type": type})
                                    continue
                                result = self.generator.generate_answer(query, G_aligned)
                                key = "answer"

                            if not result:
                                print(f"⚠️ No result for {type} in {doc_id}")
                                self.Entity_Unprocessed_elements.append({"doc_id": doc_id, "type": type})
                                continue

                            content[key] = result
                            self.processed_count += 1
                            batch_processed = True
                            print(f"✅ Processed {type} for {doc_id} ({self.processed_count}/{total_docs})")

                        except Exception as e:
                            print(f"❌ Error processing {type} for {doc_id}: {e}")
                            self.Entity_Unprocessed_elements.append({"doc_id": doc_id, "type": type})

                    if batch_processed:
                        with open(data_file, "w", encoding="utf-8") as f:
                            json.dump(json_data, f, indent=2, ensure_ascii=False)
                        print(f"✅ Batch {(i // self.batch_size) + 1} complete. Waiting {self.wait_time} seconds.")
                        time.sleep(self.wait_time)

            except Exception as e:
                print(f"⛔ Process interrupted: {e}")
                print(f"Documents processed before interruption: {self.processed_count}")
            finally:
                self.save_progress(type)
    
        if type in {"HL_summary", "title"}:
            data_file = self.community_file
            try:
                with open(data_file, "r", encoding="utf-8") as f:
                    json_data = json.load(f)

                if not json_data:
                    print("No data found in the JSON file.")
                    return

                doc_ids = list(json_data.keys())
                total_docs = len(doc_ids)
                
                for i in range(start_index, total_docs, self.batch_size):
                    batch_doc_ids = doc_ids[i : i + self.batch_size]
                    batch_processed = False

                    for doc_id in batch_doc_ids:
                        content = json_data[doc_id]
                        try:
                            result_key = "HL_summary" if type == "HL_summary" else "title"
                            if content.get(result_key):
                                continue
                            if type == "HL_summary":
                                context_lines = content.get("summary_texts", []) + content.get("relation_texts", [])
                                if not context_lines:
                                    print(f"⚠️ No context for HL_summary in {doc_id}")
                                    self.Entity_Unprocessed_elements.append({"doc_id": doc_id, "type": type})
                                    continue
                                result = self.groq_handler.generate_highlevel_summary(context_lines)
                                key = "HL_summary"

                            elif type == "title":
                                hl_summary = content.get("HL_summary")
                                if not hl_summary:
                                    print(f"⚠️ No HL_summary for title generation in {doc_id}")
                                    self.Entity_Unprocessed_elements.append({"doc_id": doc_id, "type": type})
                                    continue
                                result = self.groq_handler.generate_title(hl_summary)
                                key = "title"

                            if not result:
                                print(f"⚠️ No result for {type} in {doc_id}")
                                self.Entity_Unprocessed_elements.append({"doc_id": doc_id, "type": type})
                                continue

                            content[key] = result
                            self.processed_count += 1
                            batch_processed = True
                            print(f"✅ Processed {type} for community {doc_id} ({self.processed_count}/{total_docs})")

                        except Exception as e:
                            print(f"❌ Error processing {type} for {doc_id}: {e}")
                            self.Entity_Unprocessed_elements.append({"doc_id": doc_id, "type": type})

                    if batch_processed:
                        with open(data_file, "w", encoding="utf-8") as f:
                            json.dump(json_data, f, indent=2, ensure_ascii=False)
                        print(f"✅ Batch {(i // self.batch_size) + 1} complete. Waiting {self.wait_time} seconds.")
                        time.sleep(self.wait_time)

            except Exception as e:
                print(f"⛔ Process interrupted: {e}")
                print(f"Documents processed before interruption: {self.processed_count}")
            finally:
                self.save_progress(type)
            
        if type in {"entity", "summary", "relationship"}:
            try:
                with jsonlines.open(self.store_file, mode='r') as reader:
                    jsonl_data = list(reader)

                if not jsonl_data:
                    print("No data found in the JSONL file.")
                    return

                for i in range(start_index, len(jsonl_data), self.batch_size):
                    batch_chunks = jsonl_data[i : i + self.batch_size]

                    for chunk in batch_chunks:
                        try:
                            doc_id = chunk.get("id") or chunk.get("doc_id")
                            text_chunk = chunk.get("text")

                            if not text_chunk or not doc_id:
                                print(f"Skipping due to missing text or doc_id: {chunk}")
                                continue

                            result_key = {"entity": "entities", "summary": "summary", "relationship": "relations"}[type]
                            if result_key in chunk:
                                continue

                            if type == "entity":
                                result = self.groq_handler.extract_entities(text_chunk)
                                key = "entities"

                            elif type == "summary":
                                result = self.groq_handler.generate_summary(text_chunk)
                                key = "summary"

                            elif type == "relationship":
                                entity_list = chunk.get("entities")
                                if not entity_list:
                                    # Empty entity extraction is valid and cannot form a relation.
                                    chunk["relations"] = []
                                    with jsonlines.open(self.store_file, mode='w') as writer:
                                        writer.write_all(jsonl_data)
                                    continue
                                result = self.groq_handler.extract_relations(text_chunk, entity_list)
                                key = "relations"

                            else:
                                raise ValueError(f"Unsupported type: {type}")

                            extraction_can_be_empty = type in {"entity", "relationship"} and isinstance(result, list)
                            if result is None or (not result and not extraction_can_be_empty):
                                self.Entity_Unprocessed_elements.append({
                                    "doc_id": doc_id,
                                    "text_chunk": text_chunk,
                                    "type": type
                                })
                                print(f"⚠️ No result for document {doc_id}. Added to unprocessed.")
                                continue

                            chunk[key] = result
                            self.processed_count += 1
                            print(f"✅ Processed {type} for document {doc_id} ({self.processed_count} total).")

                            with jsonlines.open(self.store_file, mode='w') as writer:
                                writer.write_all(jsonl_data)

                        except Exception as e:
                            print(f"❌ Error processing document {chunk.get('doc_id')}: {e}")
                            continue

                    self.save_progress(type)
                    print(f"✅ Batch {(i // self.batch_size) + 1} complete. Waiting {self.wait_time} seconds.")
                    time.sleep(self.wait_time)

            except Exception as e:
                print(f"⛔ Process interrupted: {e}")
                print(f"Documents processed before interruption: {self.processed_count}")
            finally:
                self.save_progress(type)

    def retry_stored_unprocessed_documents(self, type, max_retries=3):
        self.load_progress(type)

        if type in {"reasoning_chain", "answer"}:
            data_file = self.eval_dump_file
            checkpoint_files = self._checkpoint_paths(type, max_retries)

            try:
                with open(data_file, "r", encoding="utf-8") as f:
                    json_data = json.load(f)

                if not json_data:
                    print("No data found in eval_dump.")
                    return

                for retry_idx, checkpoint_file in enumerate(checkpoint_files):
                    if not self.Entity_Unprocessed_elements:
                        print(f"✅ All documents processed successfully by retry {retry_idx}.")
                        return

                    print(f"\n🔄 Retry {retry_idx + 1}: Processing {len(self.Entity_Unprocessed_elements)} unprocessed elements...")
                    retried_elements = []

                    for item in self.Entity_Unprocessed_elements:
                        doc_id = item.get("doc_id")
                        if not doc_id:
                            continue

                        content = json_data.get(doc_id)
                        if not content:
                            print(f"⚠️ Document ID {doc_id} not found in eval_dump.")
                            continue

                        result_key = "HL_summary" if type == "HL_summary" else "title"
                        if content.get(result_key):
                            continue

                        try:
                            query = content.get("query")
                            if not query:
                                print(f"⚠️ No query in {doc_id}. Skipping.")
                                retried_elements.append(item)
                                continue

                            if type == "reasoning_chain":
                                G_raw = self.graph_builder.json_to_graph(content.get("G_raw", {}))
                                if not G_raw:
                                    print(f"⚠️ No G_raw for {doc_id}")
                                    retried_elements.append(item)
                                    continue
                                result = self.graph_builder.reasoning_chain(query, G_raw)
                                key = "reasoning_chain"

                            elif type == "answer":
                                G_aligned = self.graph_builder.json_to_graph(content.get("G_aligned", {}))
                                if not G_aligned:
                                    print(f"⚠️ No G_aligned for {doc_id}")
                                    retried_elements.append(item)
                                    continue
                                result = self.generator.generate_answer(query, G_aligned)
                                key = "answer"

                            if not result:
                                print(f"⚠️ No result for {type} in {doc_id}")
                                retried_elements.append(item)
                                continue

                            content[key] = result
                            self.processed_count += 1
                            print(f"✅ Retry {retry_idx + 1}: Successfully processed {type} for {doc_id}")

                            # Save after each success
                            with open(data_file, "w", encoding="utf-8") as f:
                                json.dump(json_data, f, indent=2, ensure_ascii=False)

                        except Exception as e:
                            print(f"❌ Error retrying {doc_id}: {e}")
                            retried_elements.append(item)

                    # Save retry checkpoint
                    with open(checkpoint_file, "w") as f:
                        json.dump({"Entity_Unprocessed_elements": retried_elements}, f, indent=4)
                    print(f"📁 Saved {len(retried_elements)} remaining to {checkpoint_file}")

                    self.Entity_Unprocessed_elements = retried_elements
                    self.save_progress(type)

            except Exception as e:
                print(f"⛔ Process interrupted during retry: {e}")
                print(f"Documents processed before interruption: {self.processed_count}")
            finally:
                self.save_progress(type)

        if type in {"HL_summary", "title"}:
            data_file = self.community_file
            checkpoint_files = self._checkpoint_paths(type, max_retries)

            try:
                with open(data_file, "r", encoding="utf-8") as f:
                    json_data = json.load(f)

                if not json_data:
                    print("No data found in the JSON file.")
                    return

                for retry_idx, checkpoint_file in enumerate(checkpoint_files):
                    if not self.Entity_Unprocessed_elements:
                        print(f"✅ All documents processed successfully by retry {retry_idx}.")
                        return

                    print(f"\n🔄 Retry {retry_idx + 1}: Processing {len(self.Entity_Unprocessed_elements)} unprocessed elements...")
                    retried_elements = []

                    for item in self.Entity_Unprocessed_elements:
                        doc_id = item["doc_id"]
                        if not doc_id:
                            continue

                        content = json_data.get(doc_id)
                        if not content:
                            print(f"⚠️ Document ID {doc_id} not found in JSON.")
                            continue

                        try:
                            if type == "HL_summary":
                                context_lines = content.get("summary_texts", []) + content.get("relation_texts", [])
                                if not context_lines:
                                    print(f"⚠️ No context for HL_summary in {doc_id}")
                                    retried_elements.append(item)
                                    continue
                                result = self.groq_handler.generate_highlevel_summary(context_lines)
                                key = "HL_summary"

                            elif type == "title":
                                hl_summary = content.get("HL_summary")
                                if not hl_summary:
                                    print(f"⚠️ No HL_summary for title generation in {doc_id}")
                                    retried_elements.append(item)
                                    continue
                                result = self.groq_handler.generate_title(hl_summary)
                                key = "title"

                            if not result:
                                print(f"⚠️ No result for {type} in {doc_id}")
                                retried_elements.append(item)
                                continue

                            content[key] = result
                            self.processed_count += 1
                            print(f"✅ Retry {retry_idx + 1}: Successfully processed {type} for {doc_id}.")

                            with open(data_file, "w", encoding="utf-8") as f:
                                json.dump(json_data, f, indent=2, ensure_ascii=False)

                        except Exception as e:
                            print(f"❌ Error retrying {doc_id}: {e}")
                            retried_elements.append(item)

                    with open(checkpoint_file, "w") as f:
                        json.dump({"Entity_Unprocessed_elements": retried_elements}, f, indent=4)
                    print(f"📁 Saved {len(retried_elements)} remaining to {checkpoint_file}")

                    self.Entity_Unprocessed_elements = retried_elements
                    self.save_progress(type)

            except Exception as e:
                print(f"⛔ Process interrupted: {e}")
                print(f"Documents processed before interruption: {self.processed_count}")
            finally:
                self.save_progress(type)

        elif type in {"entity", "summary", "relationship"}:
            store_file = self.store_file
            checkpoint_files = self._checkpoint_paths(type, max_retries)

            with jsonlines.open(store_file, mode='r') as reader:
                jsonl_data = list(reader)

            for retry_idx, checkpoint_file in enumerate(checkpoint_files):
                if not self.Entity_Unprocessed_elements:
                    print(f"✅ All documents processed successfully by retry {retry_idx}.")
                    return

                print(f"\n🔄 Retry {retry_idx + 1}: Processing {len(self.Entity_Unprocessed_elements)} unprocessed elements...")
                retried_elements = []

                for item in self.Entity_Unprocessed_elements:
                    doc_id = item.get("doc_id")
                    if not doc_id:
                        continue

                    chunk = next((c for c in jsonl_data if c.get("id") == doc_id or c.get("doc_id") == doc_id), None)

                    if not chunk:
                        print(f"⚠️ Document ID {doc_id} not found in JSONL.")
                        continue

                    result_key = {"entity": "entities", "summary": "summary", "relationship": "relations"}[type]
                    if result_key in chunk:
                        continue

                    try:
                        text_chunk = chunk.get("text")
                        if not text_chunk:
                            print(f"⚠️ No text for document {doc_id}.")
                            retried_elements.append(item)
                            continue

                        if type == "entity":
                            result = self.groq_handler.extract_entities(text_chunk)
                            key = "entities"
                        elif type == "summary":
                            result = self.groq_handler.generate_summary(text_chunk)
                            key = "summary"
                        elif type == "relationship":
                            entity_list = chunk.get("entities")
                            if not entity_list:
                                result = []
                            else:
                                result = self.groq_handler.extract_relations(text_chunk, entity_list)
                            key = "relations"
                        else:
                            raise ValueError(f"Unsupported type: {type}")

                        extraction_can_be_empty = type in {"entity", "relationship"} and isinstance(result, list)
                        if result is None or (not result and not extraction_can_be_empty):
                            retried_elements.append(item)
                            print(f"⚠️ Retry {retry_idx + 1}: Document {doc_id} still failed.")
                            continue

                        chunk[key] = result
                        self.processed_count += 1
                        print(f"✅ Retry {retry_idx + 1}: Successfully processed {doc_id}.")

                        with jsonlines.open(store_file, mode='w') as writer:
                            writer.write_all(jsonl_data)

                    except Exception as e:
                        print(f"❌ Error retrying {doc_id}: {e}")
                        retried_elements.append(item)

                with open(checkpoint_file, "w") as f:
                    json.dump({"Entity_Unprocessed_elements": retried_elements}, f, indent=4)
                print(f"📁 Saved {len(retried_elements)} remaining to {checkpoint_file}")

                self.Entity_Unprocessed_elements = retried_elements
                self.save_progress(type)

        print("✅ Retry process completed.")

    def save_progress(self, type):
        try:
            if not hasattr(self, "progress_data"):
                self.progress_data = {}

            if os.path.exists(self.progress_file):
                with open(self.progress_file, "r") as f:
                    self.progress_data = json.load(f)

            self.progress_data[type] = {
                "processed_count": self.processed_count,
                "Entity_Unprocessed_elements": self.Entity_Unprocessed_elements
            }

            with open(self.progress_file, "w") as f:
                json.dump(self.progress_data, f, indent=4)

            print(f"💾 Progress for `{type}` saved. {self.processed_count} processed, {len(self.Entity_Unprocessed_elements)} unprocessed.")

        except Exception as e:
            print(f"❌ Failed to save progress for `{type}`: {e}")

    def load_progress(self, type):
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, "r") as f:
                    data = json.load(f)

                self.progress_data = data
                type_data = data.get(type, {})
                self.processed_count = type_data.get("processed_count", 0)
                self.Entity_Unprocessed_elements = type_data.get("Entity_Unprocessed_elements", [])

                print(f"🔁 Resumed `{type}` progress: {self.processed_count} processed, {len(self.Entity_Unprocessed_elements)} unprocessed.")

            except Exception as e:
                print(f"❌ Failed to load progress for `{type}`: {e}")
        else:
            self.progress_data = {}
            self.processed_count = 0
            self.Entity_Unprocessed_elements = []

    def delete_progress(self):
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, "w") as f:
                    json.dump({}, f)
            except Exception as e:
                print(f"❌ Failed to flush json: {e}")
