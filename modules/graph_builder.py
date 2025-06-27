import os
import uuid
import re
from collections import defaultdict
from neo4j import GraphDatabase
import torch
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
import json
import jsonlines
import networkx as nx
import pickle
import igraph as ig
import leidenalg
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
import hnswlib
from groq_handler import GroqHandler

pdf_path="docs\medbook1\medical_book.pdf"

class GraphBuilder:
    def __init__(self, groq_handler: GroqHandler, jsonl_path, neo4j_uri, neo4j_user, neo4j_password, neo4j_database):
        self.groq_handler = groq_handler
        self.jsonl_path = jsonl_path
        self.neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password), database=neo4j_database)
        self.G = nx.MultiDiGraph()
        self.incomplete_graph_elements = []

    def normalize(self, name):
        return name.lower().replace(" ", "_").replace("-", "_")

    def build_graph(self, line_start=0, line_end=None):
        with jsonlines.open(self.jsonl_path, mode='r') as reader:
            for i, chunk in enumerate(reader):
                if i < line_start:
                    continue
                if line_end is not None and i >= line_end:
                    break

                doc_id = chunk.get("id") or chunk.get("doc_id")
                if not doc_id:
                    continue

                missing_fields = []

                text = chunk.get("text", "")
                summary = chunk.get("summary", "")
                entities = chunk.get("entities", [])
                relations = chunk.get("relations", [])

                if not text:
                    missing_fields.append("text")
                if not summary:
                    missing_fields.append("summary")
                if not entities:
                    missing_fields.append("entities")
                if not relations:
                    missing_fields.append("relations")

                if missing_fields:
                    self.incomplete_graph_elements.append({
                        "doc_id": doc_id,
                        "missing_fields": missing_fields
                    })
                    print(f"⚠️ Skipping doc_id={doc_id} due to missing fields: {missing_fields}")
                    continue

                # Begin adding graph nodes and edges
                Ti_id = f"T_{doc_id}"
                Si_id = f"S_{doc_id}"

                
                if Ti_id not in self.G:
                    self.G.add_node(Ti_id, type="T", text=text)
                if Si_id not in self.G:
                    self.G.add_node(Si_id, type="S", text=summary)
                self.G.add_edge(Ti_id, Si_id, label="derived_from")

                for ent in entities:
                    ent_id = f"N_{self.normalize(ent)}"
                    if ent_id not in self.G:
                        self.G.add_node(ent_id, type="N", text=ent)
                    self.G.add_edge(Si_id, ent_id, label="mentions")

                for rel in relations:
                    subj_id = f"N_{self.normalize(rel['Source'])}"
                    obj_id = f"N_{self.normalize(rel['Destination'])}"
                    rel_id = f"R_{uuid.uuid4().hex[:8]}"

                    if rel_id not in self.G:
                        self.G.add_node(rel_id, type="R", text=rel['Relationship'])
                    self.G.add_edge(rel_id, subj_id, label="subj")
                    self.G.add_edge(rel_id, obj_id, label="obj")
        
        print(f"Edges--> {self.G.edges()}, Nodes--> {self.G.nodes()}")

    def to_simple_graph(self, G: nx.MultiDiGraph):
        simple = nx.Graph()
        simple.add_nodes_from(G.nodes(data=True))
        print(f"Edges--> {G.edges()}, Nodes--> {G.nodes()}")
        for u, v in G.edges():
            simple.add_edge(u, v)
        return simple
    
    def detect_and_save_communities(self, output_path="data/community_map.json"):
        print("📊 [Step 1] Running Leiden community detection on S/N/R node subgraph...")

        sub_nodes = [n for n, d in self.G.nodes(data=True) if d.get("type") in {"S", "N", "R"}]
        subgraph = self.G.subgraph(sub_nodes)
        id_map = {node: idx for idx, node in enumerate(subgraph.nodes())}
        rev_map = {v: k for k, v in id_map.items()}
        edges = [(id_map[u], id_map[v]) for u, v in subgraph.edges()]

        ig_graph = ig.Graph(n=len(id_map), edges=edges)
        partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition)

        print(f"🔍 Found {len(partition)} communities.")

        community_map = {}
        for community_id, members in enumerate(partition):
            for idx in members:
                node_id = rev_map[idx]
                community_map[node_id] = community_id
                self.G.nodes[node_id]["community"] = community_id  # update networkx for consistency

        context_map = defaultdict(lambda: {"summary_texts": [], "relation_texts": []})

        for node_id, comm_id in community_map.items():
            node = self.G.nodes.get(node_id)
            if not node:
                continue

            node_type = node.get("type")

            if node_type == "S":
                text = node.get("text", "")
                if text.strip():
                    context_map[str(comm_id)]["summary_texts"].append(text)

            elif node_type == "R":
                predicate = node.get("text", "")
                subject_id = None
                object_id = None

                for _, target, data in self.G.out_edges(node_id, data=True):
                    if data.get("label") == "subj":
                        subject_id = target
                    elif data.get("label") == "obj":
                        object_id = target

                if subject_id and object_id:
                    subject_text = self.G.nodes.get(subject_id, {}).get("text", "")
                    object_text = self.G.nodes.get(object_id, {}).get("text", "")
                    if subject_text and object_text:
                        triple = f"{subject_text} {predicate} {object_text}"
                        context_map[str(comm_id)]["relation_texts"].append(triple)

                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(context_map, f, indent=2, ensure_ascii=False)

                print(f"✅ Saved enriched context for {len(context_map)} communities to {output_path}")

    def add_HO_nodes_from_community_map(self, neo4j=False, community_map_path="data/community_map.json"):
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

            if H_id not in self.G:
                self.G.add_node(H_id, type="H", text=hl_summary)
            if O_id not in self.G:
                self.G.add_node(O_id, type="O", text=title)
            self.G.add_edge(O_id, H_id, label="overview_of")

            for node_id, d in self.G.nodes(data=True):
                if d.get("type") == "S" and str(self.G.nodes[node_id].get("community")) == str(comm_id):
                    self.G.add_edge(H_id, node_id, label="summarizes")

            if neo4j:
                with self.neo4j_driver.session() as session:
                    session.run("MERGE (n:Node {id: $id}) SET n.type = 'H', n.text = $text", id=H_id, text=hl_summary)
                    session.run("MERGE (n:Node {id: $id}) SET n.type = 'O', n.text = $text", id=O_id, text=title)

                    session.run("""
                        MATCH (a:Node {id: $src}), (b:Node {id: $dst})
                        MERGE (a)-[:REL {label: 'overview_of'}]->(b)
                    """, src=O_id, dst=H_id)

                    for node_id, d in self.G.nodes(data=True):
                        if d.get("type") == "S" and str(self.G.nodes[node_id].get("community")) == str(comm_id):
                            session.run("""
                                MATCH (a:Node {id: $src}), (b:Node {id: $dst})
                                MERGE (a)-[:REL {label: 'summarizes'}]->(b)
                            """, src=H_id, dst=node_id)

            print(f"✅ Added H & O for community {comm_id}")

    def finalize_graph_G3_from_community_map(self, neo4j=False, community_map_path="data/community_map.json"):
        embed_nodes = [n for n, d in self.G.nodes(data=True) if d.get("type") in {"S", "H"}]
        texts = [self.G.nodes[n]["text"] for n in embed_nodes]

        print(f"🔍 Embedding {len(embed_nodes)} nodes...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(texts)

        k = int(np.ceil(np.sqrt(len(embed_nodes))))
        print(f"📈 Clustering into {k} clusters with KMeans...")
        kmeans = KMeans(n_clusters=k, random_state=42).fit(embeddings)
        labels = kmeans.labels_

        print("🔗 Adding semantically_related edges...")
        for i in range(len(embed_nodes)):
            for j in range(i + 1, len(embed_nodes)):
                node_i = embed_nodes[i]
                node_j = embed_nodes[j]
                if labels[i] == labels[j]:
                    c_i = self.G.nodes[node_i].get("community")
                    c_j = self.G.nodes[node_j].get("community")
                    if c_i == c_j and c_i is not None:
                        self.G.add_edge(node_i, node_j, label="semantically_related")
                        if neo4j:
                            with self.neo4j_driver.session() as session:
                                session.run("""
                                    MATCH (a:Node {id: $src}), (b:Node {id: $dst})
                                    MERGE (a)-[:REL {label: 'semantically_related'}]->(b)
                                """, src=node_i, dst=node_j)

        print("✅ Semantic edges added. G3 construction complete.")
    
    def build_G4_text_attachment(self, neo4j=False, chunk_file="data/chunked_output.jsonl"):
        print("📎 Reattaching original text chunks as T nodes...")

        with jsonlines.open(chunk_file, mode='r') as reader:
            chunks = list(reader)

        for chunk in chunks:
            tid = chunk.get("id")
            text = chunk.get("text")
            entities = chunk.get("entities", [])
            sid = f"S_{tid}"

            if not text or not tid or sid not in self.G:
                continue

            tid = f"T_{tid}"
            if tid not in self.G:
                self.G.add_node(tid, type="T", text=text)
            self.G.add_edge(tid, sid, label="derived_from")

            for entity in entities:
                nid = f"N_{entity['normalized']}" if isinstance(entity, dict) and 'normalized' in entity else f"N_{entity}"
                if nid in self.G:
                    self.G.add_edge(tid, nid, label="mentions_entity")

            if neo4j:
                with self.neo4j_driver.session() as session:
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
        print("🧠 Starting G5 semantic index construction...")

        embed_nodes = [n for n, d in self.G.nodes(data=True) if d.get("type") in {"T", "S", "H"}]
        texts = [self.G.nodes[n]["text"] for n in embed_nodes]

        print(f"📚 Embedding {len(embed_nodes)} nodes...")
        model = SentenceTransformer("all-MiniLM-L12-v2")
        embeddings = model.encode(texts)

        p = hnswlib.Index(space='cosine', dim=dim)
        p.init_index(max_elements=len(embeddings), ef_construction=ef, M=M)
        p.add_items(embeddings, list(range(len(embeddings))))
        p.set_ef(ef)

        print("🔗 Performing HNSW neighbor search and edge enrichment...")

        labels, distances = p.knn_query(embeddings, k=10)

        for i, neighbors in enumerate(labels):
            src = embed_nodes[i]
            c1 = self.G.nodes[src].get("community")
            for j in neighbors:
                dst = embed_nodes[j]
                if src == dst:
                    continue
                c2 = self.G.nodes[dst].get("community")
                if c1 != c2:
                    continue

                if self.G.has_edge(src, dst):
                    if isinstance(self.G, (nx.MultiGraph, nx.MultiDiGraph)):
                        for key in self.G[src][dst]:
                            self.G[src][dst][key]['weight'] = self.G[src][dst][key].get('weight', 1) + 1
                    else:
                        self.G[src][dst]['weight'] = self.G[src][dst].get('weight', 1) + 1
                else:
                    self.G.add_edge(src, dst, label="semantic_hnsw", weight=1)
                    if neo4j:
                        with self.neo4j_driver.session() as session:
                            session.run("""
                                MATCH (a:Node {id: $src}), (b:Node {id: $dst})
                                MERGE (a)-[:REL {label: 'semantic_hnsw', weight: 1}]->(b)
                            """, src=src, dst=dst)

        p.save_index(index_path)
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

        model = SentenceTransformer("all-MiniLM-L12-v2")
        q_emb = model.encode([query])[0]

        embed_nodes = [n for n, d in self.G.nodes(data=True) if d.get("type") in {"T", "S", "H"}]
        embed_id_to_node = {i: node_id for i, node_id in enumerate(embed_nodes)}

        dim = len(q_emb)
        index = hnswlib.Index(space="cosine", dim=dim)
        index.load_index(hnsw_index_path)
        index.set_ef(200)

        labels, distances = index.knn_query(q_emb, k=top_k)
        try:
            sim_nodes = {embed_id_to_node[i] for i in labels[0]}
        except KeyError as e:
            print(f"❌ Missing index ID in embed_id_to_node: {e}")
            print("Available keys:", list(embed_id_to_node.keys())[:10])
            raise

        print(f"📐 Retrieved {len(sim_nodes)} semantic neighbors from HNSW.")

        entry_nodes = matched_nodes.union(sim_nodes)
        if not entry_nodes:
            print("❌ No entry points found.")
            return set(), {}

        p_i = {node: 1 / len(entry_nodes) for node in entry_nodes}

        print(f"✅ Entry set V_entry contains {len(entry_nodes)} nodes.")
        return entry_nodes, p_i
    
    def run_shallow_ppr(self, entry_nodes, personalization_weights, m=5, alpha=0.5):
        print("🔄 Starting Shallow Personalized PageRank (t=2)...")

        node_list = list(self.G.nodes())
        node_index = {node: i for i, node in enumerate(node_list)}
        index_node = {i: node for node, i in node_index.items()}
        N = len(node_list)

        A = np.zeros((N, N))
        for u, v, data in self.G.edges(data=True):
            i, j = node_index[u], node_index[v]
            weight = data.get("weight", 1)
            A[j, i] += weight

        col_sums = A.sum(axis=0)
        P = A / (col_sums + 1e-8)

        pi_0 = np.zeros(N)
        for node, value in personalization_weights.items():
            if node in node_index:
                pi_0[node_index[node]] = value

        pi_1 = alpha * pi_0 + (1 - alpha) * P @ pi_0
        pi_2 = alpha * pi_0 + (1 - alpha) * P @ pi_1

        scores_by_type = defaultdict(list)
        for i, score in enumerate(pi_2):
            node_id = index_node[i]
            node_type = self.G.nodes[node_id].get("type")
            if node_type in {"T", "S", "H", "R"}:
                scores_by_type[node_type].append((node_id, score))

        V_cross = set()
        for node_type, scored in scores_by_type.items():
            top_nodes = sorted(scored, key=lambda x: -x[1])[:m]
            for node_id, _ in top_nodes:
                V_cross.add(node_id)

        print(f"📌 Selected {len(V_cross)} V_cross nodes (top {m} per type)")

        V_raw = set(entry_nodes) | V_cross
        V_raw = {v for v in V_raw if self.G.nodes[v].get("type") in {"T", "S", "H", "R"}}

        G_raw = self.G.subgraph(V_raw).copy()
        print(f"📎 Induced G_raw with {len(G_raw.nodes)} nodes and {len(G_raw.edges)} edges.")

        return G_raw, V_raw

    def persist_to_neo4j(self):
        with self.neo4j_driver.session() as session:
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
            os.makedirs(os.path.dirname(path), exist_ok=True)
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

    def json_to_graph(self, graph_json: dict) -> nx.DiGraph:
        G = nx.DiGraph()
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
        with open(filename, "wb") as f:
            pickle.dump(self.G, f)

    def load_graph_pickle(self, path="data/G.pkl"):
        import pickle
        with open(path, "rb") as f:
            self.G = pickle.load(f)
    
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
    
    def align_nodes_via_gnn(self, query, G_raw, reasoning_chain, model, optimizer, device, top_n=5):
        type_map = {'T':0, 'S':1, 'H':2, 'R':3, 'N':4, 'O':5}
        model.eval()

        node_texts = []
        node_types = []

        for node_id in G_raw.nodes:
            text = self.G.nodes[node_id].get("text", "")
            typ = self.G.nodes[node_id].get("type")
            node_texts.append(text)
            type_vec = [0] * 7
            type_vec[type_map[typ]] = 1
            node_types.append(type_vec)

        encoder = SentenceTransformer("all-MiniLM-L12-v2")
        text_embeds = encoder.encode(node_texts)
        x_input = torch.tensor([list(t) + type_vec for t, type_vec in zip(text_embeds, node_types)], dtype=torch.float32)

        required_edge_keys = {"label"}
        G_cleaned = nx.Graph()
        required_node_keys = {"type", "text"}
        for n, d in G_raw.nodes(data=True):
            G_cleaned.add_node(n)
            for key in required_node_keys:
                G_cleaned.nodes[n][key] = d.get(key, "" if key == "text" else "Unknown")

        for u, v, edata in G_raw.edges(data=True):
            norm_edata = {}
            for key in required_edge_keys:
                norm_edata[key] = edata.get(key, "")
            G_cleaned.add_edge(u, v, **norm_edata)
        pyg_data = from_networkx(G_cleaned)
        pyg_data.x = x_input
        edge_index = pyg_data.edge_index
        batch = torch.zeros(x_input.shape[0], dtype=torch.long)

        # 2. Query Embedding
        q_emb = encoder.encode([query])
        q_tensor = torch.tensor(q_emb[0], dtype=torch.float32)

        # 3. Extract ground-truth reasoning hits
        reasoning_text = reasoning_chain.lower()
        hit_counts = {}
        for i, text in enumerate(node_texts):
            count = len(re.findall(re.escape(text.lower()), reasoning_text))
            hit_counts[i] = count

        p_reason = torch.tensor(
            [(hit_counts.get(i, 0) + 1e-6) for i in range(len(node_texts))],
            dtype=torch.float32
        )
        p_reason = p_reason / p_reason.sum()

        # 4. Forward and loss
        node_embeds, p_pred = model(x_input, edge_index, batch, q_tensor)

        kl_loss = F.kl_div(p_pred.log(), p_reason, reduction='batchmean')

        optimizer.zero_grad()
        kl_loss.backward()
        optimizer.step()

        # 5. Prune via Top-K p_pred
        topk = torch.topk(p_pred, top_n)
        keep_nodes = set(topk.indices.tolist())
        for i in topk.indices.tolist():
            for j in G_raw.neighbors(list(G_raw.nodes)[i]):
                keep_nodes.add(list(G_raw.nodes).index(j))

        G_aligned = G_raw.subgraph([list(G_raw.nodes)[i] for i in keep_nodes]).copy()
        print(f"🧠 G_aligned contains {len(G_aligned)} nodes.")

        return G_aligned, node_embeds, p_pred

    def compute_graph_embedding(self, G_aligned, reasoning_chain, node_embeds, model_dim=256):
        node_emb_matrix = node_embeds.detach()
        r_g = node_emb_matrix.mean(dim=0)

        encoder = SentenceTransformer("all-MiniLM-L12-v2")
        r_s = encoder.encode([reasoning_chain])[0]
        r_s = torch.tensor(r_s, dtype=torch.float32)

        mlp_g = torch.nn.Sequential(
            torch.nn.Linear(256, 12288),
            torch.nn.ReLU(),
            torch.nn.Linear(12288, 12288)
        )

        mlp_s = torch.nn.Sequential(
            torch.nn.Linear(384, 12288),
            torch.nn.ReLU(),
            torch.nn.Linear(12288, 12288)
        )

        r_hat_g = mlp_g(r_g)
        r_hat_s = mlp_s(r_s)

        cos = torch.nn.CosineSimilarity(dim=0)
        sim_score = cos(r_hat_g, r_hat_s).item()
        print(f"🔗 Alignment score: {sim_score:.4f}")

        return r_hat_g, r_hat_s, sim_score


    def close(self):
        self.neo4j_driver.close()
