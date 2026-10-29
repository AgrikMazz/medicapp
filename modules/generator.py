
import jsonlines
import os
from functools import lru_cache
from .groq_handler import GroqHandler


@lru_cache(maxsize=4)
def _load_context_tokenizer(model_name):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name)


class GraphAugmentedGenerator:
    def __init__(
        self,
        groq_handler: GroqHandler,
        max_context_chars=24000,
        context_token_budget=4096,
        context_tokenizer=None,
    ):
        self.groq_handler = groq_handler
        self.max_context_chars = max_context_chars
        self.max_context_tokens = int(context_token_budget)
        self.context_tokenizer = context_tokenizer

    def textualize_ranked_passages(self, query, ranked_passages, token_budget=None):
        """Pack query-ranked original T passages without chopping source text."""
        from .retrieval import TokenBudgetPacker

        tokenizer = self.context_tokenizer
        if tokenizer is None:
            model_name = (
                os.getenv("RAG_CONTEXT_TOKENIZER")
                or os.getenv("GROQ_MODEL")
                or "openai/gpt-oss-20b"
            ).strip()
            tokenizer = _load_context_tokenizer(model_name)
            self.context_tokenizer = tokenizer
        packer = TokenBudgetPacker(tokenizer)
        packed = packer.pack(
            query,
            ranked_passages,
            token_budget=self.max_context_tokens if token_budget is None else token_budget,
        )
        return packed["context"]

    def textualize_graph(self, G_aligned):
        lines = []
        truncated_note = "[Some retrieved graph context was omitted to fit the configured context limit.]"
        content_budget = max(0, self.max_context_chars - len(truncated_note) - 1)
        node_order = sorted(
            G_aligned.nodes(data=True),
            key=lambda item: (
                item[1].get("type") != "T",
                -float(item[1].get("retrieval_score", item[1].get("ppr_score", 0.0))),
            ),
        )
        included_nodes = set()
        used_chars = 0
        truncated = False

        for node, data in node_order:
            typ = data.get("type", "")
            txt = str(data.get("text", "")).strip().replace("\n", " ")
            page_start = data.get("page_start")
            page_end = data.get("page_end")
            pages = data.get("source_pages") or []
            if typ in {"S", "H", "O"}:
                citation = "summary; cite supporting T page"
            elif page_start is not None:
                citation = f"p. {page_start}" if page_end in (None, page_start) else f"pp. {page_start}-{page_end}"
            elif pages:
                citation = "pp. " + ", ".join(str(page) for page in pages)
            elif typ == "T":
                char_start = data.get("char_start")
                char_end = data.get("char_end")
                citation = f"chars {char_start}-{char_end}" if char_start is not None and char_end is not None else "source text"
            else:
                citation = "no page metadata"
            source_id = data.get("source_chunk_id", node)
            prefix = f"[{typ} | source={source_id} | {citation}] "
            remaining = content_budget - used_chars
            if remaining <= len(prefix):
                truncated = True
                break
            if len(prefix) + len(txt) > remaining:
                txt = txt[:max(0, remaining - len(prefix) - 1)].rstrip() + "…"
                truncated = True
            lines.append(prefix + txt)
            included_nodes.add(node)
            used_chars += len(lines[-1]) + 1
            if truncated:
                break

        for u, v, data in G_aligned.edges(data=True):
            if u not in included_nodes or v not in included_nodes:
                truncated = True
                continue
            label = data.get("label", "")
            edge_line = f"[EDGE: {u} → {v}] {label}"
            if used_chars + len(edge_line) + 1 > content_budget:
                truncated = True
                break
            lines.append(edge_line)
            used_chars += len(edge_line) + 1

        if truncated:
            lines.append(truncated_note)

        return "\n".join(lines)

    def build_prompt(self, query, G_aligned, textual_context=None):
        tg = textual_context if textual_context is not None else self.textualize_graph(G_aligned)
        print(f"Prepared {len(tg)} context characters (limit: {self.max_context_chars}).")
        has_page_metadata = any(
            data.get("type") == "T" and (data.get("page_start") is not None or data.get("source_pages"))
            for _, data in G_aligned.nodes(data=True)
        ) if G_aligned is not None else False
        citation_instruction = (
            "For each factual claim, cite the page number of its supporting T passage using square-bracket syntax, "
            "such as [p. 12] or [pp. 10-11]. Never cite node IDs or summaries."
            if has_page_metadata else
            "For each factual claim, cite the exact source chunk ID from a T passage tag using [source=<ID>]. "
            "This corpus has no page metadata: do not invent page numbers. Never cite node IDs for non-T nodes or summaries."
        )
        prompt = f"""Answer in at most 90 words. For a multipart question, briefly answer every part. Use only the source context below. Treat summaries as navigation aids and prefer original text (T nodes) for factual support. {citation_instruction} When the question asks for conditions or criteria, include every relevant condition stated in the source. If the evidence is missing, ambiguous, or insufficient, say that the provided passages do not contain the answer instead of guessing.

### Query:
{query}

### Retrieved source evidence:
{tg}

Answer:"""

        return prompt

    def generate_answer(self, query, G_aligned, textual_context=None):
        prompt = self.build_prompt(query, G_aligned, textual_context=textual_context)
        print("Sending prompt to Groq LLM...")
        answer = self.groq_handler.generate_answer(prompt)
        return answer

    def save_graph_embedding(self, r_hat_g, path="data/graph_embeddings.jsonl", metadata=None):
        item = {"embedding": r_hat_g.detach().cpu().tolist()}
        if metadata:
            item.update(metadata)

        with jsonlines.open(path, mode='a') as writer:
            writer.write(item)

        print(f"💾 Saved graph embedding to {path}")
