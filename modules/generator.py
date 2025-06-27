
import jsonlines
from groq_handler import GroqHandler
pdf_path="docs\medbook1\medical_book.pdf"

class GraphAugmentedGenerator:
    def __init__(self, groq_handler: GroqHandler, graph_token="<GRAPH>", max_tokens=256):
        self.groq_handler = groq_handler
        self.graph_token = graph_token
        self.max_tokens = max_tokens

    def textualize_graph(self, G_aligned):
        lines = []
        for node in G_aligned.nodes():
            typ = G_aligned.nodes[node].get("type", "")
            txt = G_aligned.nodes[node].get("text", "").strip().replace("\n", " ")
            lines.append(f"[{typ}: {node}] {txt}")

        for u, v, data in G_aligned.edges(data=True):
            label = data.get("label", "")
            lines.append(f"[EDGE: {u} → {v}] {label}")

        return "\n".join(lines)

    def build_prompt(self, query, G_aligned):
        tg = self.textualize_graph(G_aligned)
        print(f"Context: {tg}")
        prompt = f"""Below is a user query and a pruned subgraph context. Use both to generate an accurate, comprehensive answer.

### Query:
{query}

### Subgraph (aligned):
{tg}

Answer:"""

        return prompt

    def generate_answer(self, query, G_aligned):
        prompt = self.build_prompt(query, G_aligned)
        print("📨 Sending prompt to Groq LLM...")
        answer = self.groq_handler.generate_answer(prompt)
        return answer

    def save_graph_embedding(self, r_hat_g, path="data/graph_embeddings.jsonl", metadata=None):
        item = {"embedding": r_hat_g.detach().cpu().tolist()}
        if metadata:
            item.update(metadata)

        with jsonlines.open(path, mode='a') as writer:
            writer.write(item)

        print(f"💾 Saved graph embedding to {path}")
