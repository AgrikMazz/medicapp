"""Passage-level retrieval and query-conditioned context packing.

The original graph path returns a heterogeneous subgraph. This module keeps
the original T passages as the retrieval units, uses the graph as an optional
expansion channel, and provides comparable BM25, exact-dense, hybrid, and
cross-encoder rankings.
"""

from __future__ import annotations

import math
import re
import contextlib
import io
from collections import Counter
from dataclasses import dataclass

import numpy as np


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "their", "this",
    "to", "was", "were", "what", "when", "which", "who", "why", "with",
}


def _tokens(text):
    return [
        token for token in re.findall(r"[\w]+", str(text).casefold())
        if token not in STOPWORDS
    ]


def reciprocal_rank_fusion(rankings, rrf_k=60, allowed_ids=None, tie_order=None):
    """Fuse ranked passage ID lists without comparing uncalibrated scores."""
    scores = Counter()
    allowed = set(allowed_ids) if allowed_ids is not None else None
    for ranking in rankings:
        for rank, row in enumerate(ranking, start=1):
            passage_id = row[0] if isinstance(row, (tuple, list)) else row
            if allowed is not None and passage_id not in allowed:
                continue
            scores[passage_id] += 1.0 / (rrf_k + rank)
    stable_order = tie_order or {}
    return sorted(
        ((passage_id, float(score)) for passage_id, score in scores.items()),
        key=lambda row: (-row[1], stable_order.get(row[0], math.inf), row[0]),
    )


@dataclass(frozen=True)
class Passage:
    node_id: str
    source_id: str
    text: str
    char_start: int
    char_end: int
    order: int


class PassageRetriever:
    """Cache a book's T passages and provide passage-level retrieval methods."""

    def __init__(self, graph_builder, embedding_model=None, cross_encoder=None):
        self.builder = graph_builder
        self.graph = graph_builder.G
        self.embedding_model = embedding_model or graph_builder._embedder()
        self.cross_encoder = cross_encoder
        self._ppr_passage_cache = {}

        rows = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") != "T":
                continue
            try:
                start = int(data.get("char_start"))
                end = int(data.get("char_end"))
            except (TypeError, ValueError):
                continue
            if end <= start:
                continue
            rows.append(Passage(
                node_id=str(node_id),
                source_id=str(data.get("source_chunk_id", node_id)),
                text=str(data.get("text", "")),
                char_start=start,
                char_end=end,
                order=0,
            ))
        rows.sort(key=lambda passage: (passage.char_start, passage.char_end, passage.node_id))
        self.passages = [
            Passage(
                passage.node_id, passage.source_id, passage.text,
                passage.char_start, passage.char_end, order,
            )
            for order, passage in enumerate(rows)
        ]
        if not self.passages:
            raise ValueError("The loaded graph contains no T passages with source offsets.")

        self.by_id = {passage.node_id: passage for passage in self.passages}
        self.order = {passage.node_id: passage.order for passage in self.passages}
        self.document_tokens = [_tokens(passage.text) for passage in self.passages]
        self.document_frequencies = Counter()
        for values in self.document_tokens:
            self.document_frequencies.update(set(values))
        self.average_length = sum(map(len, self.document_tokens)) / len(self.document_tokens)
        self.idf = {
            term: math.log(
                1.0 + (len(self.passages) - frequency + 0.5) / (frequency + 0.5)
            )
            for term, frequency in self.document_frequencies.items()
        }
        self.term_frequencies = [Counter(values) for values in self.document_tokens]

        self.embeddings = self.embedding_model.encode(
            [passage.text for passage in self.passages],
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        self._cross_scores = {}

    def bm25(self, query, k1=1.5, b=0.75):
        query_terms = set(_tokens(query))
        scored = []
        for index, frequencies in enumerate(self.term_frequencies):
            length = len(self.document_tokens[index])
            score = 0.0
            for term in query_terms:
                tf = frequencies.get(term, 0)
                if not tf:
                    continue
                normalization = tf + k1 * (
                    1.0 - b + b * length / max(self.average_length, 1.0)
                )
                score += self.idf[term] * tf * (k1 + 1.0) / normalization
            if score > 0:
                scored.append((self.passages[index].node_id, float(score)))
        scored.sort(key=lambda row: (-row[1], self.order[row[0]]))
        return scored

    def dense(self, query):
        query_embedding = self.embedding_model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        similarities = self.embeddings @ query_embedding
        scored = [
            (passage.node_id, float(similarities[index]))
            for index, passage in enumerate(self.passages)
        ]
        scored.sort(key=lambda row: (-row[1], self.order[row[0]]))
        return scored

    def cross_scores(self, query, candidates=None):
        if self.cross_encoder is None:
            raise RuntimeError("Cross-encoder is not loaded.")
        if candidates is None:
            candidate_ids = [passage.node_id for passage in self.passages]
        else:
            candidate_ids = [
                row[0] if isinstance(row, (tuple, list)) else row
                for row in candidates
            ]
            candidate_ids = list(dict.fromkeys(candidate_ids))
        score_cache = self._cross_scores.setdefault(query, {})
        missing_ids = [node_id for node_id in candidate_ids if node_id not in score_cache]
        if missing_ids:
            pairs = [[query, self.by_id[node_id].text] for node_id in missing_ids]
            scores = self.cross_encoder.predict(
                pairs, batch_size=16, show_progress_bar=False
            )
            score_cache.update({
                node_id: float(score)
                for node_id, score in zip(missing_ids, np.asarray(scores).reshape(-1))
            })
        return score_cache

    def cross_rerank(self, query, candidates):
        candidate_ids = [row[0] if isinstance(row, (tuple, list)) else row for row in candidates]
        candidate_ids = list(dict.fromkeys(candidate_ids))
        scores = self.cross_scores(query, candidate_ids)
        return sorted(
            ((passage_id, scores[passage_id]) for passage_id in candidate_ids),
            key=lambda row: (-row[1], self.order[row[0]]),
        )

    def rrf(self, bm25_rows, dense_rows, rrf_k=60):
        return reciprocal_rank_fusion(
            (bm25_rows, dense_rows), rrf_k=rrf_k, tie_order=self.order
        )

    def graph_expand(
        self,
        bm25_rows,
        dense_rows,
        seed_k=20,
        expand_k=20,
        alpha=0.35,
        max_iter=20,
        rrf_k=60,
    ):
        """Expand hybrid passage seeds over the book graph and return T passages.

        Seed weights come from fused retrieval ranks. PPR is one additional
        ranked list; it does not assign fixed quotas to heterogeneous node
        types. Only original T passages are returned as evidence candidates.
        """
        fused_seeds = self.rrf(bm25_rows, dense_rows, rrf_k=rrf_k)[:max(1, seed_k)]
        if not fused_seeds:
            return []
        seed_nodes = {node_id for node_id, _ in fused_seeds}
        seed_weights = {node_id: score for node_id, score in fused_seeds}
        cache_key = (
            tuple(sorted((node, round(float(weight), 14)) for node, weight in seed_weights.items())),
            round(float(alpha), 8),
            int(max_iter),
        )
        ppr_rows = self._ppr_passage_cache.get(cache_key)
        if ppr_rows is None:
            ppr = self.builder.personalized_pagerank_scores(
                seed_nodes,
                seed_weights,
                alpha=alpha,
                max_iter=max_iter,
            )
            ppr_rows = [
                (passage.node_id, ppr.get(passage.node_id, 0.0))
                for passage in self.passages
                if ppr.get(passage.node_id, 0.0) > 0
            ]
            ppr_rows.sort(key=lambda row: (-row[1], self.order[row[0]]))
            self._ppr_passage_cache[cache_key] = ppr_rows
        selected_ppr = ppr_rows[:max(1, expand_k)]
        candidate_ids = seed_nodes | {node_id for node_id, _ in selected_ppr}
        return reciprocal_rank_fusion(
            (bm25_rows, dense_rows, selected_ppr),
            rrf_k=rrf_k,
            allowed_ids=candidate_ids,
            tie_order=self.order,
        )

    def hnsw_graph_expand(
        self,
        query,
        hnsw_index_path="data/hnsw_index.bin",
        hnsw_candidate_k=20,
        expand_k=20,
        alpha=0.5,
        max_iter=10,
    ):
        """Retrieve T passages from saved HNSW/lexical entry points via PPR."""
        with contextlib.redirect_stdout(io.StringIO()):
            entry_nodes, personalization = self.builder.get_entry_points(
                query,
                hnsw_index_path=hnsw_index_path,
                top_k=max(1, int(hnsw_candidate_k)),
            )
        ppr = self.builder.personalized_pagerank_scores(
            entry_nodes,
            personalization,
            alpha=alpha,
            max_iter=max_iter,
        )
        ranked = [
            (passage.node_id, float(ppr.get(passage.node_id, 0.0)))
            for passage in self.passages
            if ppr.get(passage.node_id, 0.0) > 0
        ]
        ranked.sort(key=lambda row: (-row[1], self.order[row[0]]))
        selected = {node_id for node_id, _ in ranked[:max(1, int(expand_k))]}
        selected.update(node_id for node_id in entry_nodes if node_id in self.by_id)
        return [row for row in ranked if row[0] in selected]

    def attach_cross_scores(self, query, candidates):
        """Sort a bounded candidate list using the loaded cross-encoder."""
        return self.cross_rerank(query, candidates)

    def rank(
        self,
        query,
        method="graph_crossencoder",
        candidate_k=40,
        graph_seed_k=20,
        graph_expand_k=20,
        ppr_alpha=0.35,
        ppr_max_iter=20,
        rrf_k=60,
        hnsw_index_path="data/hnsw_index.bin",
        hnsw_candidate_k=20,
        hnsw_ppr_alpha=0.5,
        hnsw_ppr_max_iter=10,
    ):
        """Return one ranked list of original T passages for an online query."""
        lexical = self.bm25(query)
        if method == "bm25":
            return lexical
        if method in {"hnsw_ppr", "hnsw_ppr_crossencoder"}:
            candidates = self.hnsw_graph_expand(
                query,
                hnsw_index_path=hnsw_index_path,
                hnsw_candidate_k=hnsw_candidate_k,
                expand_k=graph_expand_k,
                alpha=hnsw_ppr_alpha,
                max_iter=hnsw_ppr_max_iter,
            )
            if method == "hnsw_ppr":
                return candidates
            if self.cross_encoder is None:
                from sentence_transformers import CrossEncoder
                self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            return self.cross_rerank(query, candidates[:max(1, candidate_k)])

        dense = self.dense(query)
        if method == "dense_exact":
            return dense
        hybrid = self.rrf(lexical, dense, rrf_k=rrf_k)
        if method == "hybrid_rrf":
            return hybrid
        if method in {"graph_ppr", "graph_crossencoder"}:
            candidates = self.graph_expand(
                lexical,
                dense,
                seed_k=graph_seed_k,
                expand_k=graph_expand_k,
                alpha=ppr_alpha,
                max_iter=ppr_max_iter,
                rrf_k=rrf_k,
            )
        elif method in {"hybrid_crossencoder", "hybrid_crossencoder_all"}:
            candidates = hybrid
        else:
            raise ValueError(f"Unknown passage retrieval method: {method}")

        if method in {"hybrid_crossencoder", "graph_crossencoder", "hybrid_crossencoder_all"}:
            rerank_count = len(candidates) if method == "hybrid_crossencoder_all" else candidate_k
            if self.cross_encoder is None:
                from sentence_transformers import CrossEncoder
                self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            return self.cross_rerank(query, candidates[:max(1, rerank_count)])
        return candidates

    def passage_context_record(self, node_id, score=0.0):
        passage = self.by_id[node_id]
        data = self.graph.nodes[node_id]
        return {
            "node_id": passage.node_id,
            "source_id": passage.source_id,
            "text": passage.text,
            "char_start": passage.char_start,
            "char_end": passage.char_end,
            "score": float(score),
            "page_start": data.get("page_start"),
            "page_end": data.get("page_end"),
            "source_pages": data.get("source_pages") or [],
        }


class TokenBudgetPacker:
    """Pack ranked, intact source passages under a tokenizer-measured budget."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def pack(self, query, ranked_passages, token_budget=4096):
        """Pack passages in query-conditioned rank order without truncating them."""
        del query  # Ranking is already conditioned on the live query.
        lines = []
        selected_ids = []
        used_tokens = 0
        for passage in ranked_passages:
            if not isinstance(passage, dict):
                raise TypeError("Passage rank records must be dictionaries for packing.")
            node_id = passage["node_id"]
            node_id = str(node_id)
            if node_id in selected_ids:
                continue
            source_id = str(passage.get("source_id", node_id))
            pages = passage.get("source_pages") or []
            page_start = passage.get("page_start")
            page_end = passage.get("page_end", page_start)
            if page_start is not None:
                citation = f"p. {page_start}" if page_end in (None, page_start) else f"pp. {page_start}-{page_end}"
            elif pages:
                citation = "pp. " + ", ".join(map(str, pages))
            else:
                citation = f"chars {passage['char_start']}-{passage['char_end']}"
            prefix = f"[T | source={source_id} | {citation}] "
            text = re.sub(r"\s+", " ", str(passage.get("text", ""))).strip()
            line = prefix + text
            candidate_context = "\n".join(lines + [line])
            candidate_tokens = self._count(candidate_context)
            if candidate_tokens > token_budget:
                continue
            lines.append(line)
            selected_ids.append(node_id)
            used_tokens = candidate_tokens
        return {
            "context": "\n".join(lines),
            "passage_ids": selected_ids,
            "token_count": used_tokens,
            "token_budget": int(token_budget),
        }
