import re
import requests
import time
import json
import os
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = os.getenv("GROQ_API_URL")

class GroqHandler:
    def __init__(self):
        self.headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

    def _call_groq(self, messages, is_json=True):
        payload = {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "messages": messages,
            "temperature": 0.2
        }
        if is_json:
            payload["response_format"] = {
                "type": "json_object"
            }
        try:
            response = requests.post(
                url=GROQ_API_URL,
                json=payload,
                headers=self.headers
            )
            print(response.json())
            #response.raise_for_status()

            if response.status_code==200:
                res=response.json()
                content = res["choices"][0]["message"]["content"]
                if content:
                    processed_content=content
                    #print(content)
                    return processed_content

            if response.status_code==400:
                    error_data = response.json()
                    if (
                        "error" in error_data
                        and "failed_generation" in error_data["error"]
                    ):
                        print("Handling failed generation from 400 error.")
                        failed_gen = error_data["error"]["failed_generation"]
                        processed_failed_gen = failed_gen
                        #print(processed_failed_gen)
                        return processed_failed_gen
                    elif (
                        "error" in error_data
                        and "code" in error_data["error"]
                        and error_data["error"]["code"] == "rate_limit_exceeded"
                    ):
                        print("Rate limit exceeded. Retrying after 10s...")
                        return error_data["error"]["code"]
                    else:
                        print("400 error encountered but no 'failed_generation' found.")
                        return None

        except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError, IndexError) as e:
            return None

    def parse_entities(self, entities_string):
        pattern = r'"([a-zA-Z][^"]*[a-zA-Z])"'
        return re.findall(pattern, entities_string)

    def extract_entities(self, text_chunk):
        messages = [{
                "role": "system",
                "content": f"""You are a Named Entity Extractor Agent. Below is a text chunk, your job is to extract ALL single and multi-word named entities and output a SINGLE-LEVEL json_object containing those extracted entities. DO NOT have anything else in output, just the json object. Maximum entities to extract is 12 and this should NOT be extended under any circumstances. Response formats:
                    <Response>
                        {
                            "Entity1",
                            "Entity2",
                            "Entity3",
                            "Entity4"
                        }
                    </Response>
                """
            },{
                "role": "user",
                "content": f"Text: {text_chunk}"
            }]
        attempt=1
        while attempt <= 3:
            result = self._call_groq(messages)
            attempt += 1
            if result != None and result != "rate_limit_exceeded":
                break
            if result == None and result == "rate_limit_exceeded":
                print("Rate limit exceeded. Retrying after 10s...")
                time.sleep(10)
                continue
            print(f"Trial number {attempt} failed.")
            time.sleep(2)
        print(f"Result--> {result}")

        if result:
            entities_list = self.parse_entities(result)
            print(f"Parsed entities: {entities_list}")

            return entities_list
        return None

    def extract_relations(self, text_chunk, entity_list):
        messages = [{
                "role": "system",
                "content": f"You are a relationship extraction agent. Referencing the given text chunk, generate only contextually relevant Source-Relationship-Destination triplets STRICTLY in the format: {{'Source': 'S', 'Relationship': 'R', 'Destination': 'D'}} where Source, Relationship, Destination are key-value pairs, S and D are STRICTLY part of the given entity list and NOT anywhere else, and R is relevant text that connects S and D, referencing ONLY the given text chunk. Conditions: 1. MAXIMUM SIZE LIMIT of json output is 12 and SHOULD NOT EXCEED, 2. Output should only consist of a JSON object."
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
            
            print(result)
            input_string = str(result)

            kv_pattern = re.compile(r"""['\"]?Source['\"]?:\s*['\"']?([^,'\"}]+)['\"']?\s*,\s*['\"]?Relationship['\"]?:\s*['\"']?([^,'\"}]+)['\"']?\s*,\s*['\"]?Destination['\"]?:\s*['\"']?([^,'\"}]+)['\"']?""")
            set_pattern = re.compile(r"""[{(]['\"]([^'\"]+)['\"]\s*,\s*['\"]([^'\"]+)['\"][})]""")

            # KV format
            triplets_kv = re.findall(kv_pattern, input_string)
            if triplets_kv:
                print("Extracted triplets in key-value format:")
                triplets = []
                for triplet in triplets_kv:
                    triplets.append({"Source": triplet[0], "Relationship": triplet[1], "Destination": triplet[2]})
                    print({"Source": triplet[0], "Relationship": triplet[1], "Destination": triplet[2]})
                return triplets

            # Set format
            triplets_set = re.findall(set_pattern, input_string)
            if triplets_set:
                print("Extracted triplets in set format:")
                triplets = []
                for triplet in triplets_set:
                    triplets.append({"Source": triplet[0], "Relationship": triplet[1], "Destination": triplet[2]})
                    print({"Source": triplet[0], "Relationship": triplet[1], "Destination": triplet[2]})
                return triplets_set

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