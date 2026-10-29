import os
import re
import hashlib
from collections import defaultdict
from neo4j import GraphDatabase
import json
import jsonlines
import networkx as nx
import pickle
import math
from collections import Counter
import igraph as ig
import leidenalg
from sentence_transformers import SentenceTransformer
import numpy as np
import hnswlib
from scipy.sparse import coo_matrix, diags
from .groq_handler import GroqHandler

class GraphBuilder:
    def __init__(self, groq_handler: GroqHandler, jsonl_path, neo4j_uri, neo4j_user, neo4j_password, neo4j_database):
        self.groq_handler = groq_handler
        self.jsonl_path = jsonl_path
        self.neo4j_database = neo4j_database
        self.neo4j_driver = None
        if neo4j_uri and neo4j_user and neo4j_password:
            self.neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.G = nx.MultiDiGraph()
        self.incomplete_graph_elements = []
        self._embedding_model = None
        self._passage_retriever = None
        self._ppr_transition_cache = None

    def _invalidate_retrieval_caches(self):
        self._passage_retriever = None
        self._ppr_transition_cache = None

    def _embedder(self):
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer("all-MiniLM-L12-v2")
        return self._embedding_model

    def get_passage_retriever(self, cross_encoder_name=None):
        """Return a cached passage retriever for the currently loaded graph."""
        from .retrieval import PassageRetriever

        if self._passage_retriever is None:
            self._passage_retriever = PassageRetriever(self)
        if cross_encoder_name and self._passage_retriever.cross_encoder is None:
            from sentence_transformers import CrossEncoder
            self._passage_retriever.cross_encoder = CrossEncoder(cross_encoder_name)
        return self._passage_retriever

    def _neo4j_session(self):
        if self.neo4j_driver is None:
            raise RuntimeError("Neo4j is not configured. Set NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD.")
        if self.neo4j_database:
            return self.neo4j_driver.session(database=self.neo4j_database)
        return self.neo4j_driver.session()

    def normalize(self, name):
        return re.sub(r"[^\w]+", "_", str(name).casefold(), flags=re.UNICODE).strip("_")

    def build_graph(self, line_start=0, line_end=None):
        self._invalidate_retrieval_caches()
        with jsonlines.open(self.jsonl_path, mode='r') as reader:
            for i, chunk in enumerate(reader):
                if i < line_start:
                    continue
                if line_end is not None and i >= line_end:
                    break

                doc_id = chunk.get("id") or chunk.get("doc_id")
                if not doc_id:
                    continue

                text = chunk.get("text", "")
                summary = chunk.get("summary", "")
                entities = chunk.get("entities", [])
                relations = chunk.get("relations", [])
                missing_fields = []
                if not summary:
                    missing_fields.append("summary")
                for field in ("entities", "relations"):
                    if field not in chunk or chunk[field] is None:
                        missing_fields.append(field)
                if missing_fields:
                    self.incomplete_graph_elements.append({"doc_id": doc_id, "missing_fields": missing_fields})
                if not text:
                    self.incomplete_graph_elements.append({"doc_id": doc_id, "missing_fields": ["text"]})
                    continue

                source_attrs = {
                    "source_chunk_id": str(doc_id),
                    "source_doc_id": chunk.get("source_doc_id", chunk.get("doc_id", "")),
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                    "source_pages": chunk.get("source_pages", []),
                    "char_start": chunk.get("char_start"),
                    "char_end": chunk.get("char_end"),
                }
                Ti_id = f"T_{doc_id}"
                Si_id = f"S_{doc_id}"

                if Ti_id not in self.G:
                    self.G.add_node(Ti_id, type="T", text=text, **source_attrs)
                if summary and Si_id not in self.G:
                    self.G.add_node(Si_id, type="S", text=summary, **source_attrs)
                    self.G.add_edge(Ti_id, Si_id, label="derived_from", **source_attrs)

                for ent in entities:
                    ent_text = ent.get("name", "") if isinstance(ent, dict) else str(ent)
                    if not ent_text.strip():
                        continue
                    ent_id = f"N_{self.normalize(ent_text)}"
                    if ent_id not in self.G:
                        self.G.add_node(ent_id, type="N", text=ent_text, source_chunk_ids=[])
                    self.G.nodes[ent_id].setdefault("source_chunk_ids", [])
                    if str(doc_id) not in self.G.nodes[ent_id]["source_chunk_ids"]:
                        self.G.nodes[ent_id]["source_chunk_ids"].append(str(doc_id))
                    if summary and Si_id in self.G:
                        self.G.add_edge(Si_id, ent_id, label="mentions", **source_attrs)
                    self.G.add_edge(Ti_id, ent_id, label="mentions_entity", **source_attrs)

                for rel in relations:
                    if not isinstance(rel, dict):
                        continue
                    subject = str(rel.get("Source", "")).strip()
                    predicate = str(rel.get("Relationship", "")).strip()
                    obj = str(rel.get("Destination", "")).strip()
                    if not subject or not predicate or not obj:
                        continue
                    subj_id = f"N_{self.normalize(subject)}"
                    obj_id = f"N_{self.normalize(obj)}"
                    for node_id, entity_text in ((subj_id, subject), (obj_id, obj)):
                        if node_id not in self.G:
                            self.G.add_node(node_id, type="N", text=entity_text, source_chunk_ids=[str(doc_id)])
                    relation_key = hashlib.sha256(
                        f"{doc_id}:{self.normalize(subject)}:{self.normalize(predicate)}:{self.normalize(obj)}".encode("utf-8")
                    ).hexdigest()[:20]
                    rel_id = f"R_{relation_key}"
                    self.G.add_node(rel_id, type="R", text=predicate, **source_attrs)
                    self.G.add_edge(rel_id, subj_id, label="subject", **source_attrs)
                    self.G.add_edge(rel_id, obj_id, label="object", **source_attrs)
                    self.G.add_edge(Ti_id, rel_id, label="asserts", **source_attrs)
                    if Si_id in self.G:
                        self.G.add_edge(Si_id, rel_id, label="summarizes_relation", **source_attrs)

        print(
            f"Graph built: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges, "
            f"{len(self.incomplete_graph_elements)} incomplete chunks."
        )

    def to_simple_graph(self, G: nx.MultiDiGraph):
        simple = nx.Graph()
        simple.add_nodes_from(G.nodes(data=True))
        print(f"Edges--> {G.edges()}, Nodes--> {G.nodes()}")
        for u, v in G.edges():
            simple.add_edge(u, v)
        return simple

    def detect_and_save_communities(self, output_path="data/community_map.json"):
        print("📊 [Step 1] Running Leiden community detection on S/N/R node subgraph...")

        previous_context = {}
        if os.path.exists(output_path):
            try:
                with open(output_path, "r", encoding="utf-8") as previous_file:
                    previous_context = json.load(previous_file)
            except (OSError, json.JSONDecodeError):
                previous_context = {}

        sub_nodes = [n for n, d in self.G.nodes(data=True) if d.get("type") in {"S", "N", "R"}]
        subgraph = self.G.subgraph(sub_nodes)
        id_map = {node: idx for idx, node in enumerate(subgraph.nodes())}
        rev_map = {v: k for k, v in id_map.items()}
        edges = [(id_map[u], id_map[v]) for u, v in subgraph.edges()]

        community_map = {}
        if edges:
            undirected_edges = sorted({tuple(sorted(edge)) for edge in edges if edge[0] != edge[1]})
            ig_graph = ig.Graph(n=len(id_map), edges=undirected_edges, directed=False)
            partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition)
            for community_id, members in enumerate(partition):
                for idx in members:
                    node_id = rev_map[idx]
                    community_map[node_id] = community_id
        else:
            for community_id, node_id in enumerate(sub_nodes):
                community_map[node_id] = community_id

        print(f"🔍 Found {len(set(community_map.values()))} communities.")
        for node_id, comm_id in community_map.items():
            self.G.nodes[node_id]["community"] = comm_id

        context_map = defaultdict(lambda: {
            "summary_texts": [], "relation_texts": [], "source_pages": [], "source_chunk_ids": []
        })

        for node_id, comm_id in community_map.items():
            node = self.G.nodes.get(node_id)
            if not node:
                continue

            node_type = node.get("type")
            community_context = context_map[str(comm_id)]
            community_context["source_pages"].extend(node.get("source_pages", []))
            if node.get("source_chunk_id") is not None:
                community_context["source_chunk_ids"].append(str(node["source_chunk_id"]))

            if node_type == "S":
                text = node.get("text", "")
                if text.strip():
                    community_context["summary_texts"].append(text)

            elif node_type == "R":
                predicate = node.get("text", "")
                subject_id = None
                object_id = None

                for _, target, data in self.G.out_edges(node_id, data=True):
                    if data.get("label") in {"subj", "subject"}:
                        subject_id = target
                    elif data.get("label") in {"obj", "object"}:
                        object_id = target

                if subject_id and object_id:
                    subject_text = self.G.nodes.get(subject_id, {}).get("text", "")
                    object_text = self.G.nodes.get(object_id, {}).get("text", "")
                    if subject_text and object_text:
                        triple = f"{subject_text} {predicate} {object_text}"
                        community_context["relation_texts"].append(triple)

        for comm_id, context in context_map.items():
            previous = previous_context.get(comm_id, {})
            if (
                previous.get("summary_texts") == context["summary_texts"]
                and previous.get("relation_texts") == context["relation_texts"]
            ):
                for key in ("HL_summary", "title"):
                    if previous.get(key):
                        context[key] = previous[key]

        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(context_map, f, indent=2, ensure_ascii=False)

        print(f"✅ Saved enriched context for {len(context_map)} communities to {output_path}")

    def add_HO_nodes_from_community_map(self, neo4j=False, community_map_path="data/community_map.json"):
        self._invalidate_retrieval_caches()
        if not os.path.exists(community_map_path):
            print("❌ Missing community_map.json")
            return

        with open(community_map_path, "r", encoding="utf-8") as f:
            community_data = json.load(f)

        for comm_id, data in community_data.items():
            hl_summary = data.get("HL_summary")
            title = data.get("title")

            if not hl_summary or not title:
                print(f"⚠️ Skipping community {comm_id}: missing HL_summary or title.")
                continue

            H_id = f"H_{comm_id}"
            O_id = f"O_{comm_id}"
            member_s_nodes = [
                node_id for node_id, attrs in self.G.nodes(data=True)
                if attrs.get("type") == "S" and str(attrs.get("community")) == str(comm_id)
            ]
            community_pages = sorted({
                page
                for node_id in member_s_nodes
                for page in self.G.nodes[node_id].get("source_pages", [])
                if page is not None
            })
            community_chunks = sorted({
                str(self.G.nodes[node_id].get("source_chunk_id"))
                for node_id in member_s_nodes
                if self.G.nodes[node_id].get("source_chunk_id") is not None
            } | {str(chunk_id) for chunk_id in data.get("source_chunk_ids", [])})
            community_pages = sorted(set(community_pages) | set(data.get("source_pages", [])))
            highlevel_attrs = {
                "community": int(comm_id),
                "source_pages": community_pages,
                "source_chunk_ids": community_chunks,
            }

            if H_id not in self.G:
                self.G.add_node(H_id, type="H", text=hl_summary, **highlevel_attrs)
            else:
                self.G.nodes[H_id].update(type="H", text=hl_summary, **highlevel_attrs)
            if O_id not in self.G:
                self.G.add_node(O_id, type="O", text=title, **highlevel_attrs)
            else:
                self.G.nodes[O_id].update(type="O", text=title, **highlevel_attrs)
            self.G.add_edge(O_id, H_id, label="overview_of")

            for node_id in member_s_nodes:
                self.G.add_edge(
                    H_id,
                    node_id,
                    label="summarizes",
                    source_chunk_id=self.G.nodes[node_id].get("source_chunk_id"),
                    source_pages=self.G.nodes[node_id].get("source_pages", []),
                )

            if neo4j:
                with self._neo4j_session() as session:
                    session.run("MERGE (n:Node {id: $id}) SET n.type = 'H', n.text = $text", id=H_id, text=hl_summary)
                    session.run("MERGE (n:Node {id: $id}) SET n.type = 'O', n.text = $text", id=O_id, text=title)

                    session.run("""
                        MATCH (a:Node {id: $src}), (b:Node {id: $dst})
                        MERGE (a)-[:REL {label: 'overview_of'}]->(b)
                    """, src=O_id, dst=H_id)

                    for node_id in member_s_nodes:
                        session.run("""
                            MATCH (a:Node {id: $src}), (b:Node {id: $dst})
                            MERGE (a)-[:REL {label: 'summarizes'}]->(b)
                        """, src=H_id, dst=node_id)

            print(f"✅ Added H & O for community {comm_id}")

    def finalize_graph_G3_from_community_map(self, neo4j=False, community_map_path="data/community_map.json"):
        self._invalidate_retrieval_caches()
        embed_nodes = [n for n, d in self.G.nodes(data=True) if d.get("type") in {"S", "H"}]
        if len(embed_nodes) < 2:
            return
        texts = [self.G.nodes[n]["text"] for n in embed_nodes]

        print(f"🔍 Embedding {len(embed_nodes)} nodes...")
        embeddings = self._embedder().encode(texts, normalize_embeddings=True)
        index = hnswlib.Index(space="cosine", dim=embeddings.shape[1])
        index.init_index(max_elements=len(embeddings), ef_construction=100, M=16)
        index.add_items(embeddings, np.arange(len(embeddings)))
        index.set_ef(min(100, len(embeddings)))
        neighbor_ids, distances = index.knn_query(embeddings, k=min(11, len(embeddings)))

        for i, neighbors in enumerate(neighbor_ids):
            src = embed_nodes[i]
            src_community = self.G.nodes[src].get("community")
            if src_community is None:
                continue
            for neighbor_pos, j in enumerate(neighbors):
                if i == j:
                    continue
                dst = embed_nodes[int(j)]
                if self.G.nodes[dst].get("community") != src_community:
                    continue
                similarity = 1.0 - float(distances[i][neighbor_pos])
                if similarity < 0.45:
                    continue
                self.G.add_edge(src, dst, label="semantically_related", weight=similarity)

        print("✅ Semantic edges added. G3 construction complete.")

    def build_G4_text_attachment(self, neo4j=False, chunk_file="data/chunked_output.jsonl"):
        self._invalidate_retrieval_caches()
        print("📎 Reattaching original text chunks as T nodes...")

        with jsonlines.open(chunk_file, mode='r') as reader:
            chunks = list(reader)

        for chunk in chunks:
            tid = chunk.get("id")
            text = chunk.get("text")
            entities = chunk.get("entities", [])
            sid = f"S_{tid}"

            if not text or not tid:
                continue

            tid = f"T_{tid}"
            if tid not in self.G:
                self.G.add_node(tid, type="T", text=text)
            source_attrs = {
                "source_chunk_id": str(chunk.get("id")),
                "source_doc_id": chunk.get("doc_id", ""),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "source_pages": chunk.get("source_pages", []),
                "char_start": chunk.get("char_start"),
                "char_end": chunk.get("char_end"),
            }
            self.G.nodes[tid].update(type="T", text=text, **source_attrs)
            if sid in self.G:
                community = self.G.nodes[sid].get("community")
                if community is not None:
                    self.G.nodes[tid]["community"] = community
                self.G.add_edge(tid, sid, label="derived_from", **source_attrs)

            for entity in entities:
                entity_text = entity.get("name", "") if isinstance(entity, dict) else str(entity)
                nid = f"N_{self.normalize(entity_text)}"
                if nid in self.G:
                    self.G.add_edge(tid, nid, label="mentions_entity", **source_attrs)

            if neo4j:
                with self._neo4j_session() as session:
                    session.run("MERGE (n:Node {id: $id}) SET n.type = 'T', n.text = $text", id=tid, text=text)
                    session.run("""
                        MATCH (a:Node {id: $src}), (b:Node {id: $dst})
                        MERGE (a)-[:REL {label: 'derived_from'}]->(b)
                    """, src=tid, dst=sid)

                    for entity in entities:
                        nid = f"N_{entity['normalized']}" if isinstance(entity, dict) and 'normalized' in entity else f"N_{entity}"
                        if nid in self.G:
                            session.run("""
                                MATCH (a:Node {id: $src}), (b:Node {id: $dst})
                                MERGE (a)-[:REL {label: 'mentions_entity'}]->(b)
                            """, src=tid, dst=nid)

        print("✅ G4 built: all T nodes attached and connected.")

    def build_G5_semantic_hnsw_index(self, neo4j=False, dim=384, ef=200, M=16, index_path="data/hnsw_index.bin"):
        self._invalidate_retrieval_caches()
        print("🧠 Starting G5 semantic index construction...")

        embed_nodes = [n for n, d in self.G.nodes(data=True) if d.get("type") in {"T", "S", "H"}]
        if not embed_nodes:
            raise ValueError("Cannot build the semantic index: the graph contains no T, S, or H nodes.")
        texts = [self.G.nodes[n]["text"] for n in embed_nodes]

        print(f"📚 Embedding {len(embed_nodes)} nodes...")
        embeddings = self._embedder().encode(texts, normalize_embeddings=True)

        dim = int(embeddings.shape[1])
        p = hnswlib.Index(space='cosine', dim=dim)
        p.init_index(max_elements=len(embeddings), ef_construction=ef, M=M)
        p.add_items(embeddings, list(range(len(embeddings))))
        p.set_ef(min(ef, len(embeddings)))

        print("🔗 Performing HNSW neighbor search and edge enrichment...")

        labels, distances = p.knn_query(embeddings, k=min(11, len(embeddings)))

        for i, neighbors in enumerate(labels):
            src = embed_nodes[i]
            c1 = self.G.nodes[src].get("community")
            if c1 is None:
                continue
            for neighbor_pos, j in enumerate(neighbors):
                dst = embed_nodes[int(j)]
                if src == dst:
                    continue
                c2 = self.G.nodes[dst].get("community")
                similarity = 1.0 - float(distances[i][neighbor_pos])
                if c1 != c2 or similarity < 0.45:
                    continue

                self.G.add_edge(src, dst, label="semantic_hnsw", weight=similarity)

        parent = os.path.dirname(index_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        p.save_index(index_path)
        index_fingerprint = hashlib.sha256(json.dumps(
            [(node_id, self.G.nodes[node_id].get("text", "")) for node_id in embed_nodes],
            ensure_ascii=False,
        ).encode("utf-8")).hexdigest()
        with open(index_path + ".nodes.json", "w", encoding="utf-8") as f:
            json.dump({"nodes": embed_nodes, "fingerprint": index_fingerprint}, f, ensure_ascii=False)
        print(f"💾 HNSW index saved to: {index_path}")
        print("✅ G5 completed: HNSW semantic links embedded.")

    def get_entry_points(self, query, hnsw_index_path="data/hnsw_index.bin", top_k=30):
        print("🎯 Identifying entry points for query...")

        query_tokens = set(query.lower().split())
        matched_nodes = set()

        for node_id, data in self.G.nodes(data=True):
            if data.get("type") not in {"N", "O"}:
                continue
            label = data.get("text", "").lower()
            label_tokens = set(label.split())
            jaccard = len(query_tokens & label_tokens) / len(query_tokens | label_tokens | {"."})
            if jaccard > 0.3:
                matched_nodes.add(node_id)

        print(f"🔎 Found {len(matched_nodes)} exact-match nodes from N/O.")

        q_emb = self._embedder().encode([query], normalize_embeddings=True)[0]

        mapping_path = hnsw_index_path + ".nodes.json"
        if os.path.exists(mapping_path):
            with open(mapping_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            embed_nodes = mapping.get("nodes", mapping) if isinstance(mapping, dict) else mapping
        else:
            embed_nodes = [n for n, d in self.G.nodes(data=True) if d.get("type") in {"T", "S", "H"}]
            mapping = None
        if any(node_id not in self.G for node_id in embed_nodes):
            raise ValueError("HNSW node mapping references nodes missing from the loaded graph.")
        expected_fingerprint = hashlib.sha256(json.dumps(
            [(node_id, self.G.nodes[node_id].get("text", "")) for node_id in embed_nodes],
            ensure_ascii=False,
        ).encode("utf-8")).hexdigest()
        if isinstance(mapping, dict) and mapping.get("fingerprint") != expected_fingerprint:
            raise ValueError("HNSW node mapping is stale for the loaded graph. Rebuild the semantic index.")
        embed_id_to_node = {i: node_id for i, node_id in enumerate(embed_nodes)}

        dim = len(q_emb)
        index = hnswlib.Index(space="cosine", dim=dim)
        index.load_index(hnsw_index_path)
        if index.get_current_count() != len(embed_nodes):
            raise ValueError("HNSW index and node mapping do not match. Rebuild the index for this graph.")
        index.set_ef(min(200, index.get_current_count()))

        labels, distances = index.knn_query(q_emb, k=min(top_k, index.get_current_count()))
        try:
            sim_nodes = {embed_id_to_node[i] for i in labels[0]}
        except KeyError as e:
            print(f"❌ Missing index ID in embed_id_to_node: {e}")
            print("Available keys:", list(embed_id_to_node.keys())[:10])
            raise

        print(f"📐 Retrieved {len(sim_nodes)} semantic neighbors from HNSW.")

        lexical_nodes = self._lexical_candidates(query, top_k=min(8, top_k))
        print(f"🔤 Retrieved {len(lexical_nodes)} lexical passage candidates.")

        entry_nodes = matched_nodes.union(sim_nodes).union(lexical_nodes)
        if not entry_nodes:
            print("❌ No entry points found.")
            return set(), {}

        p_i = {node: 1 / len(entry_nodes) for node in entry_nodes}

        print(f"✅ Entry set V_entry contains {len(entry_nodes)} nodes.")
        return entry_nodes, p_i

    def _lexical_candidates(self, query, top_k=5, node_types=("T", "S", "H"), return_scores=False):
        """Retrieve BM25-style lexical candidates, optionally preserving rank scores."""
        stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from",
            "how", "in", "is", "it", "of", "on", "or", "that", "the", "their", "this",
            "to", "was", "were", "what", "when", "which", "who", "why", "with",
        }

        def terms(text):
            return [term for term in re.findall(r"[\w]+", str(text).casefold()) if term not in stopwords]

        query_terms = set(terms(query))
        if not query_terms:
            return [] if return_scores else set()

        documents = []
        for node, attrs in self.G.nodes(data=True):
            if attrs.get("type") in set(node_types):
                token_list = terms(attrs.get("text", ""))
                if token_list:
                    documents.append((node, token_list))
        if not documents:
            return [] if return_scores else set()

        document_frequency = Counter()
        for _, token_list in documents:
            document_frequency.update(set(token_list) & query_terms)
        average_length = sum(len(tokens) for _, tokens in documents) / len(documents)
        scores = []
        k1, b = 1.5, 0.75
        for node, token_list in documents:
            frequencies = Counter(token_list)
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if frequency == 0:
                    continue
                doc_frequency = document_frequency[term]
                inverse_frequency = math.log(1.0 + (len(documents) - doc_frequency + 0.5) / (doc_frequency + 0.5))
                norm = frequency + k1 * (1.0 - b + b * len(token_list) / max(average_length, 1.0))
                score += inverse_frequency * frequency * (k1 + 1.0) / norm
            if score > 0:
                scores.append((node, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        if return_scores:
            return scores[:top_k]
        return {node for node, _ in scores[:top_k]}

    def personalized_pagerank_scores(
        self, entry_nodes, personalization_weights, alpha=0.5, max_iter=10, tol=1e-8
    ):
        """Return PPR scores for every node using a symmetrized retrieval walk.

        The stored evidence graph is directed, but retrieval needs to walk from a
        passage to an entity and back to other passages that mention it. The
        transition graph is therefore intentionally symmetrized here; this is a
        retrieval projection, not a claim that the source relations are
        themselves symmetric.
        """
        node_list = list(self.G.nodes())
        if not node_list or not entry_nodes:
            return {}
        if self._ppr_transition_cache is None:
            node_index = {node: i for i, node in enumerate(node_list)}
            rows, cols, values = [], [], []
            for u, v, data in self.G.edges(data=True):
                if u not in node_index or v not in node_index or u == v:
                    continue
                try:
                    weight = max(float(data.get("weight", 1.0)), 0.0)
                except (TypeError, ValueError):
                    weight = 1.0
                if weight == 0:
                    continue
                i, j = node_index[u], node_index[v]
                rows.extend((j, i))
                cols.extend((i, j))
                values.extend((weight, weight))

            count = len(node_list)
            adjacency = coo_matrix(
                (values, (rows, cols)), shape=(count, count), dtype=np.float64
            ).tocsr()
            degree = np.asarray(adjacency.sum(axis=0)).ravel()
            inverse_degree = np.zeros_like(degree)
            non_dangling = degree > 0
            inverse_degree[non_dangling] = 1.0 / degree[non_dangling]
            transition = adjacency @ diags(inverse_degree)
            self._ppr_transition_cache = (node_list, node_index, transition, ~non_dangling)
        node_list, node_index, transition, dangling = self._ppr_transition_cache

        count = len(node_list)
        personalization = np.zeros(count, dtype=np.float64)
        for node, value in personalization_weights.items():
            if node in node_index:
                personalization[node_index[node]] = max(float(value), 0.0)
        total = personalization.sum()
        if total <= 0:
            valid_entries = [node for node in entry_nodes if node in node_index]
            if not valid_entries:
                return {}
            personalization[[node_index[node] for node in valid_entries]] = 1.0 / len(valid_entries)
        else:
            personalization /= total

        scores = personalization.copy()
        for _ in range(max_iter):
            dangling_mass = scores[dangling].sum()
            updated = alpha * personalization + (1.0 - alpha) * (
                transition @ scores + dangling_mass * personalization
            )
            if np.abs(updated - scores).sum() < tol:
                scores = updated
                break
            scores = updated

        return {node_id: float(scores[index]) for node_id, index in node_index.items()}

    def run_shallow_ppr(self, entry_nodes, personalization_weights, m=5, alpha=0.5, max_iter=10, tol=1e-8):
        """Run shallow PPR and retain entry nodes plus top ``m`` of each type."""
        scores_by_node = self.personalized_pagerank_scores(
            entry_nodes, personalization_weights, alpha=alpha, max_iter=max_iter, tol=tol
        )
        if not scores_by_node:
            return self.G.subgraph([]).copy(), set()

        scores_by_type = defaultdict(list)
        for node_id, score in scores_by_node.items():
            node_type = self.G.nodes[node_id].get("type")
            if score > 0 and node_type in {"T", "S", "H", "R", "N", "O"}:
                scores_by_type[node_type].append((node_id, score))

        selected = set(entry_nodes)
        for scored in scores_by_type.values():
            selected.update(node_id for node_id, _ in sorted(scored, key=lambda item: item[1], reverse=True)[:m])
        selected = {node for node in selected if node in self.G}

        G_raw = self.G.subgraph(selected).copy()
        for node_id in G_raw.nodes:
            G_raw.nodes[node_id]["ppr_score"] = scores_by_node[node_id]
        print(f"📎 Retrieved subgraph: {len(G_raw.nodes)} nodes, {len(G_raw.edges)} edges.")
        return G_raw, selected

    def persist_to_neo4j(self):
        with self._neo4j_session() as session:
            for node_id, data in self.G.nodes(data=True):
                session.run(
                    """
                    MERGE (n:Node {id: $id})
                    SET n.type = $type, n.text = $text
                    """,
                    id=node_id, type=data.get("type"), text=data.get("text", "")
                )

            for src, dst, edge_data in self.G.edges(data=True):
                session.run(
                    """
                    MATCH (a:Node {id: $src}), (b:Node {id: $dst})
                    MERGE (a)-[r:REL {label: $label}]->(b)
                    """,
                    src=src, dst=dst, label=edge_data.get("label", "")
                )

    def save_incomplete_entries(self, path="data/incomplete_graph_elements.json"):
        if self.incomplete_graph_elements:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "w") as f:
                json.dump(self.incomplete_graph_elements, f, indent=4)
            print(f"🛑 {len(self.incomplete_graph_elements)} incomplete chunks saved to {path}")
        else:
            print("✅ No incomplete graph elements to save.")

    def graph_to_json(self, G: nx.Graph) -> dict:
        return {
            "nodes": [
                {"id": node, **data}
                for node, data in G.nodes(data=True)
            ],
            "edges": [
                {"source": u, "target": v, **data}
                for u, v, data in G.edges(data=True)
            ]
        }

    def json_to_graph(self, graph_json: dict) -> nx.MultiDiGraph:
        G = nx.MultiDiGraph()
        for node in graph_json["nodes"]:
            node_id = node["id"]
            attrs = {k: v for k, v in node.items() if k != "id"}
            G.add_node(node_id, **attrs)

        for edge in graph_json["edges"]:
            u = edge["source"]
            v = edge["target"]
            attrs = {k: v for k, v in edge.items() if k not in {"source", "target"}}
            G.add_edge(u, v, **attrs)

        return G

    def save_pickle(self, filename="graph.pkl"):
        parent = os.path.dirname(filename)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(self.G, f)

    def load_graph_pickle(self, path="data/G.pkl"):
        import pickle
        with open(path, "rb") as f:
            self.G = pickle.load(f)
        self._invalidate_retrieval_caches()

    def reasoning_chain(self, query, G_raw):
        context_texts = []
        for node_id in G_raw.nodes:
            node_type = self.G.nodes[node_id]["type"]
            if node_type in {"T", "S", "H", "R"}:
                context_texts.append(self.G.nodes[node_id].get("text", ""))
        context="\n".join(context_texts)

        llm_prompt = f"""
            Given the question: {query}, and the following context from the candidate subgraph nodes:
            {context},
            produce a step-by-step reasoning chain that connects the question to the correct answer, referencing intermediate entities/nodes.
        """

        return self.groq_handler.generate_reasoning_chain(llm_prompt)

    def prune_subgraph(self, query, G_raw, top_n=12):
        """Deterministically rerank retrieved nodes and retain their one-hop evidence."""
        if G_raw.number_of_nodes() == 0:
            return G_raw.copy()

        node_ids = list(G_raw.nodes())
        texts = [str(G_raw.nodes[node].get("text", "")) for node in node_ids]
        embeddings = self._embedder().encode([query] + texts, normalize_embeddings=True)
        query_embedding = embeddings[0]
        semantic_scores = embeddings[1:] @ query_embedding
        max_ppr = max((float(G_raw.nodes[node].get("ppr_score", 0.0)) for node in node_ids), default=0.0)

        scored = []
        for node_id, semantic_score in zip(node_ids, semantic_scores):
            ppr_score = float(G_raw.nodes[node_id].get("ppr_score", 0.0))
            normalized_ppr = ppr_score / max_ppr if max_ppr > 0 else 0.0
            score = 0.75 * ((float(semantic_score) + 1.0) / 2.0) + 0.25 * normalized_ppr
            scored.append((node_id, score))
        scored.sort(key=lambda item: item[1], reverse=True)

        seed_nodes = {node_id for node_id, _ in scored[:max(1, top_n)]}
        keep_nodes = set(seed_nodes)
        for node_id in seed_nodes:
            keep_nodes.update(G_raw.neighbors(node_id))
            if G_raw.is_directed():
                keep_nodes.update(G_raw.predecessors(node_id))

        G_aligned = G_raw.subgraph(keep_nodes).copy()
        score_map = dict(scored)
        for node_id in G_aligned.nodes:
            G_aligned.nodes[node_id]["retrieval_score"] = score_map.get(node_id, 0.0)
        return G_aligned


    def close(self):
        if self.neo4j_driver is not None:
            self.neo4j_driver.close()
