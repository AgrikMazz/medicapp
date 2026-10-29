import re
import requests
import time
import json
import os
from dotenv import load_dotenv
load_dotenv()

GROQ_API_URL = os.getenv("GROQ_API_URL")
GROQ_MODEL = (os.getenv("GROQ_MODEL") or "openai/gpt-oss-20b").strip()
GROQ_REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "low")
GROQ_MAX_COMPLETION_TOKENS = int(os.getenv("GROQ_MAX_COMPLETION_TOKENS", "512"))
GROQ_MIN_REQUEST_INTERVAL = float(os.getenv("GROQ_MIN_REQUEST_INTERVAL_SECONDS", "2.2"))

class GroqHandler:
    def __init__(self):
        configured_keys = [
            ("GROQ_API_KEY", os.getenv("GROQ_API_KEY", "").strip()),
            ("GROQ_API_KEY_2", os.getenv("GROQ_API_KEY_2", "").strip()),
        ]
        models = [
            ("GROQ_MODEL", os.getenv("GROQ_MODEL", GROQ_MODEL).strip()),
            ("GROQ_MODEL_2", os.getenv("GROQ_MODEL_2", "").strip()),
            ("GROQ_MODEL_3", os.getenv("GROQ_MODEL_3", "").strip()),
        ]
        configured_keys = [(name, value) for name, value in configured_keys if value]
        models = [(name, value) for name, value in models if value]
        if not configured_keys:
            raise ValueError("Neither GROQ_API_KEY nor GROQ_API_KEY_2 is configured.")
        if not models:
            raise ValueError("No model is configured in GROQ_MODEL, GROQ_MODEL_2, or GROQ_MODEL_3.")

        # Key-major ordering is intentional: all configured models are tried with
        # key 1 before any model is tried with key 2.
        self.candidates = [
            {
                "key_name": key_name,
                "key": key,
                "model_env": model_name,
                "model": model,
            }
            for key_name, key in configured_keys
            for model_name, model in models
        ]
        self._active_candidate_index = 0
        self.last_http_status = None
        self.last_error_type = None
        self.last_rate_limit_info = {}
        self.last_response_metadata = {}
        self.last_model = None
        self.last_key_slot = None
        self.fallback_events = []
        self.total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.usage_by_model = {}
        self.successful_requests = 0
        self._last_request_at = 0.0
        if not GROQ_API_URL:
            raise ValueError("GROQ_API_URL is not configured.")
        base_url = GROQ_API_URL.rstrip("/")
        self.chat_completions_url = (
            base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"
        )

    def _call_groq(self, messages, is_json=True):
        self.last_http_status = None
        self.last_error_type = None
        self.last_rate_limit_info = {}

        # Once a candidate succeeds, keep using it for the rest of this run.
        # This avoids burning every request against a model/key already known to
        # be rate-limited, while preserving the configured fallback order.
        for candidate_index in range(self._active_candidate_index, len(self.candidates)):
            # A credential failure can jump directly over the other models for
            # that same key while the original range iterator is still active.
            if candidate_index < self._active_candidate_index:
                continue
            candidate = self.candidates[candidate_index]
            payload = {
                "model": candidate["model"],
                "messages": messages,
                "temperature": 0.2,
                "max_completion_tokens": GROQ_MAX_COMPLETION_TOKENS,
            }
            # reasoning_effort is a GPT-OSS parameter; other model families may
            # reject it as an invalid request field.
            if "gpt-oss" in candidate["model"].lower():
                payload["reasoning_effort"] = GROQ_REASONING_EFFORT
            if is_json:
                payload["response_format"] = {"type": "json_object"}

            candidate_failed = False
            for attempt in range(2):
                delay = GROQ_MIN_REQUEST_INTERVAL - (time.monotonic() - self._last_request_at)
                if delay > 0:
                    time.sleep(delay)
                self._last_request_at = time.monotonic()

                try:
                    response = requests.post(
                        url=self.chat_completions_url,
                        json=payload,
                        headers={
                            "Authorization": f"Bearer {candidate['key']}",
                            "Content-Type": "application/json",
                        },
                        timeout=(10, 120),
                    )
                    self.last_http_status = response.status_code

                    if response.status_code == 429:
                        self.last_rate_limit_info = {
                            header: response.headers.get(header)
                            for header in (
                                "retry-after",
                                "x-ratelimit-remaining-requests",
                                "x-ratelimit-remaining-tokens",
                                "x-ratelimit-reset-tokens",
                            )
                        }
                        self.last_error_type = "RateLimited"
                        candidate_failed = True
                        break

                    try:
                        response_data = response.json()
                    except (json.JSONDecodeError, ValueError):
                        self.last_error_type = "InvalidJSONResponse"
                        candidate_failed = True
                        break

                    if response.status_code >= 500:
                        self.last_error_type = f"HTTP{response.status_code}"
                        if attempt == 0:
                            time.sleep(1.0)
                            continue
                        candidate_failed = True
                        break

                    if response.status_code >= 400:
                        error_data = response_data.get("error", {})
                        if not isinstance(error_data, dict):
                            error_data = {"message": str(error_data)}
                        error_code = error_data.get("code", "")
                        error_message = str(error_data.get("message", ""))
                        self.last_error_type = str(error_code or f"HTTP{response.status_code}")
                        # A bad credential cannot work with the other models;
                        # skip directly to the next configured API key.
                        if response.status_code in (401, 403):
                            next_key_index = next(
                                (
                                    idx for idx in range(candidate_index + 1, len(self.candidates))
                                    if self.candidates[idx]["key_name"] != candidate["key_name"]
                                ),
                                len(self.candidates),
                            )
                            candidate_failed = True
                            self._active_candidate_index = next_key_index
                            break
                        # Retry this candidate without JSON mode if the provider
                        # reports that this model does not support response_format.
                        if is_json and "response_format" in error_message.lower():
                            payload.pop("response_format", None)
                            if attempt == 0:
                                continue
                        candidate_failed = True
                        break

                    usage = response_data.get("usage", {})
                    message = response_data.get("choices", [{}])[0].get("message", {})
                    content = message.get("content")
                    finish_reason = response_data.get("choices", [{}])[0].get("finish_reason")
                    self.last_response_metadata = {
                        "model": candidate["model"],
                        "key_slot": candidate["key_name"],
                        "finish_reason": finish_reason,
                        "content_characters": len(content) if isinstance(content, str) else 0,
                        "message_fields": sorted(message.keys()),
                        "usage": usage,
                    }
                    if not isinstance(content, str) or not content.strip():
                        self.last_error_type = "EmptyModelResponse"
                        candidate_failed = True
                        break

                    for token_key in self.total_usage:
                        amount = int(usage.get(token_key, 0) or 0)
                        self.total_usage[token_key] += amount
                    model_usage = self.usage_by_model.setdefault(candidate["model"], {
                        "requests": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    })
                    model_usage["requests"] += 1
                    for token_key in model_usage:
                        if token_key != "requests":
                            model_usage[token_key] += int(usage.get(token_key, 0) or 0)
                    self.successful_requests += 1
                    self.last_model = candidate["model"]
                    self.last_key_slot = candidate["key_name"]
                    self._active_candidate_index = candidate_index
                    return content

                except requests.exceptions.RequestException as e:
                    self.last_error_type = type(e).__name__
                    if attempt == 0:
                        time.sleep(1.0)
                        continue
                    candidate_failed = True
                    break

            if candidate_failed:
                self.fallback_events.append({
                    "key_slot": candidate["key_name"],
                    "model": candidate["model"],
                    "http_status": self.last_http_status,
                    "error_type": self.last_error_type,
                })
                print(
                    f"Groq candidate {candidate['key_name']} / {candidate['model']} "
                    f"failed ({self.last_error_type or 'request error'}); trying fallback."
                )
                if self._active_candidate_index >= len(self.candidates):
                    break
                self._active_candidate_index = max(self._active_candidate_index, candidate_index + 1)

        if self.last_error_type == "RateLimited":
            self._active_candidate_index = 0
            return "rate_limit_exceeded"
        # Allow the next logical request to retry the configured priority order
        # after an all-candidate outage instead of leaving the pointer exhausted.
        self._active_candidate_index = 0
        print("All configured Groq key/model combinations failed for this request.")
        return None

    def get_runtime_metadata(self):
        """Return safe run metadata; API key values are intentionally excluded."""
        return {
            "configured_models": [candidate["model"] for candidate in self.candidates],
            "configured_key_slots": list(dict.fromkeys(
                candidate["key_name"] for candidate in self.candidates
            )),
            "last_successful_model": self.last_model,
            "last_successful_key_slot": self.last_key_slot,
            "fallback_events": list(self.fallback_events),
            "usage_by_model": self.usage_by_model,
        }

    @staticmethod
    def _retry_delay(headers):
        retry_after = headers.get("retry-after")
        if retry_after:
            try:
                return max(float(retry_after), 1.0)
            except ValueError:
                pass

        reset = headers.get("x-ratelimit-reset-tokens", "")
        minutes = re.search(r"(\d+(?:\.\d+)?)m", reset)
        seconds = re.search(r"(\d+(?:\.\d+)?)s", reset)
        delay = (float(minutes.group(1)) * 60 if minutes else 0.0) + (
            float(seconds.group(1)) if seconds else 0.0
        )
        return max(delay, 10.0)

    def parse_entities(self, entities_string):
        try:
            data = json.loads(entities_string)
            entities = data.get("entities", []) if isinstance(data, dict) else data
            if isinstance(entities, dict):
                entities = list(entities.values())
            if not isinstance(entities, list):
                return []
            values = [
                item.get("name", "") if isinstance(item, dict) else str(item)
                for item in entities
            ]
            return list(dict.fromkeys(value.strip() for value in values if value and value.strip()))
        except (json.JSONDecodeError, TypeError):
            return list(dict.fromkeys(re.findall(r'"([^"\n]+)"', str(entities_string))))

    def extract_entities(self, text_chunk):
        messages = [{
                "role": "system",
                "content": """Extract the named entities and concepts stated in the passage. Do not invent entities. Return only a JSON object with this shape: {"entities": ["name", "name"]}. Include all relevant entities; do not impose an arbitrary count limit."""
            },{
                "role": "user",
                "content": f"Text: {text_chunk}"
            }]
        attempt=1
        while attempt <= 3:
            result = self._call_groq(messages)
            if result and result != "rate_limit_exceeded":
                break
            if result == "rate_limit_exceeded":
                retry_delay = self._retry_delay(self.last_rate_limit_info)
                print(f"Rate limit exceeded. Retrying after {retry_delay:g}s...")
                time.sleep(retry_delay)
            else:
                print(f"Trial number {attempt} failed.")
                time.sleep(2)
            attempt += 1
        if result and result != "rate_limit_exceeded":
            entities_list = self.parse_entities(result)
            print(f"Extracted {len(entities_list)} entities.")

            return entities_list
        return None

    def extract_relations(self, text_chunk, entity_list):
        messages = [{
                "role": "system",
                "content": "Extract at most 6 important relationships directly stated in the passage. Use exact entity names from the supplied list. Do not infer facts. Return only compact JSON: {\"relations\":[{\"Source\":\"name\",\"Relationship\":\"short relation\",\"Destination\":\"name\"}]}. Return an empty relations array if none are stated."
            },{
                "role": "user",
                "content": f"""
                    <Text_Chunk>{text_chunk}</Text_Chunk>
                    <Entity_List>{entity_list}</Entity_List>
                """
            }]
        try:
            result = self._call_groq(messages)
            if result is None:
                print(f"No triplets generated.")
                return
            
            data = json.loads(result)
            relations = data.get("relations", []) if isinstance(data, dict) else []
            if isinstance(relations, dict):
                relations = [relations]
            normalized = []
            for relation in relations:
                if not isinstance(relation, dict):
                    continue
                item = {
                    "Source": str(relation.get("Source", "")).strip(),
                    "Relationship": str(relation.get("Relationship", "")).strip(),
                    "Destination": str(relation.get("Destination", "")).strip(),
                }
                if all(item.values()):
                    normalized.append(item)
            return normalized

        except Exception as e:
            print("Error processing triplets:", e)
    
    def generate_summary(self, text_chunk):
        messages = [{
                "role": "system",
                "content": f"Paraphrase the given text_chunk into a concise, independent event‐level summary capturing the core semantic unit. Maximum word limit is 50 and should not be crossed. Output ONLY the summary and no other text."
            },{
                "role": "user",
                "content": f"<Text_Chunk>{text_chunk}</Text_Chunk>"
            }]
        return self._call_groq(messages, is_json=False)
    
    def generate_attributes(self, entity_name, context):
        messages = [{
                "role": "system",
                "content": f"You are an attribute based summarisation agent."
            },{
                "role": "user",
                "content": f"""
                    Using the following context paragraphs that involve {entity_name}, 
                    produce a concise attribute summary describing {entity_name}'s salient properties 
                    (e.g., roles, attributes, dates, relationships).

                    Context:
                    {context}

                    Summary (30 to 60 tokens):
                """
            }]
        return self._call_groq(messages, is_json=False)
    
    def generate_summary(self, text_chunk):
        messages = [{
                "role": "system",
                "content": f"Paraphrase the given text_chunk into a concise, independent event‐level summary capturing the core semantic unit. Maximum word limit is 50 and should not be crossed. Output ONLY the summary and no other text."
            },{
                "role": "user",
                "content": f"<Text_Chunk>{text_chunk}</Text_Chunk>"
            }]
        return self._call_groq(messages, is_json=False)
    
    def generate_highlevel_summary(self, context_list):
        context = "\n".join(context_list)
        messages = [{
            "role": "system",
            "content": "Using the following context paragraphs from a topical cluster, produce a high-level summary (100–150 tokens) describing the core theme, key insights, and salient points. Output ONLY the summary."
        }, {
            "role": "user",
            "content": f"<Community_Context>{context}</Community_Context>"
        }]
        return self._call_groq(messages, is_json=False)

    def generate_title(self, highlevel_summary):
        messages = [{
            "role": "system",
            "content": "Extract a concise 5–8 word keyword-style title from the following summary. Output ONLY the title."
        }, {
            "role": "user",
            "content": f"<Summary>{highlevel_summary}</Summary>"
        }]
        return self._call_groq(messages, is_json=False)

    def generate_reasoning_chain(self, prompt):
        messages = [{
            "role": "system",
            "content": "You are an advanced reasoning agent who thinks each step deeply, and returns a step-by-step reasoning chain capturing the question and the context provided."
        }, {
            "role": "user",
            "content": f"{prompt}"
        }]
        return self._call_groq(messages, is_json=False)
    
    def generate_answer(self, prompt):
        messages = [{
            "role": "system",
            "content": "You are a helpful AI agent. Your job is to answer the question asked by user correctly based on context provided, and NOT anywhere else. If you do not have context ot do not know the answer, say that you do not know the answer."
        }, {
            "role": "user",
            "content": f"{prompt}"
        }]
        return self._call_groq(messages, is_json=False)
