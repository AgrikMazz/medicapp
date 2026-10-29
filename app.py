import streamlit as st
import os
from modules.chunker import Chunker
from modules.groq_handler import GroqHandler
from modules.graph_builder import GraphBuilder
from modules.generator import GraphAugmentedGenerator
from modules.entity_handler import EntityHandler
from modules.evaluator import NodeRAGEvaluator
from dotenv import load_dotenv
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")

chunker = Chunker()
groq = GroqHandler()
generator = GraphAugmentedGenerator(
    groq,
    max_context_chars=int(os.getenv("ANSWER_CONTEXT_CHAR_LIMIT", "24000")),
    context_token_budget=int(os.getenv("ANSWER_CONTEXT_TOKEN_BUDGET", "4096")),
)
builder = GraphBuilder(
    groq_handler=groq,
    jsonl_path="data/chunked_output.jsonl",
    neo4j_uri=NEO4J_URI,
    neo4j_user=NEO4J_USER,
    neo4j_password=NEO4J_PASSWORD,
    neo4j_database=NEO4J_DATABASE
)
entity_handler = EntityHandler(groq, builder, generator)
evaluator = NodeRAGEvaluator(chunker, builder, generator, entity_handler)

st.title("📘 Graph-RAG Explorer")

option = st.sidebar.selectbox("Choose an operation", [
    "Upload and Chunk PDF",
    "Run Entity/Relation Extraction",
    "Build Graph & Detect Communities",
    "Generate HO Nodes & Semantic Edges",
    "Run QA Query",
])

if option == "Upload and Chunk PDF":
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())
        pages = chunker.pdf_to_pages("temp.pdf")
        chunks = chunker.segment_into_chunks(pages, source_id=uploaded_file.name)
        st.success(f"Processed and saved {len(chunks)} chunks.")

elif option == "Run Entity/Relation Extraction":
    st.write("Running Named Entity and Relationship Extraction...")
    with st.spinner("Processing..."):
        entity_handler.process_documents(type="entity")
        entity_handler.retry_stored_unprocessed_documents(type="entity")
        entity_handler.process_documents(type="summary")
        entity_handler.retry_stored_unprocessed_documents(type="summary")
        entity_handler.process_documents(type="relationship")
        entity_handler.retry_stored_unprocessed_documents(type="relationship")
    st.success("Extraction complete.")

elif option == "Build Graph & Detect Communities":
    with st.spinner("Building graph..."):
        builder.G.clear()
        builder.build_graph()
        builder.detect_and_save_communities()
        builder.save_pickle("graph.pkl")
        st.success("Graph and communities built!")

elif option == "Generate HO Nodes & Semantic Edges":
    with st.spinner("Enhancing graph..."):
        builder.load_graph_pickle("graph.pkl")
        entity_handler.process_documents(type="HL_summary")
        entity_handler.retry_stored_unprocessed_documents(type="HL_summary")
        entity_handler.process_documents(type="title")
        entity_handler.retry_stored_unprocessed_documents(type="title")
        builder.add_HO_nodes_from_community_map()
        builder.finalize_graph_G3_from_community_map()
        builder.build_G4_text_attachment()
        builder.build_G5_semantic_hnsw_index(
            index_path=os.getenv("HNSW_INDEX_PATH", "data/hnsw_index.bin")
        )
        builder.save_pickle()
        st.success("Graph enriched and saved.")

elif option == "Run QA Query":
    query = st.text_input("Enter your question:")
    if query:
        builder.load_graph_pickle("graph.pkl")
        retriever = builder.get_passage_retriever(
            os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        )
        retrieval_method = os.getenv("RETRIEVAL_METHOD", "hnsw_ppr_crossencoder")
        ranked = retriever.rank(
            query,
            method=retrieval_method,
            candidate_k=int(os.getenv("RETRIEVAL_RERANK_CANDIDATES", "40")),
            graph_seed_k=int(os.getenv("GRAPH_RETRIEVAL_SEED_K", "20")),
            graph_expand_k=int(os.getenv("GRAPH_RETRIEVAL_EXPAND_K", "20")),
            ppr_alpha=float(os.getenv("GRAPH_RETRIEVAL_PPR_ALPHA", "0.5")),
            hnsw_index_path=os.getenv("HNSW_INDEX_PATH", "data/hnsw_index.bin"),
            hnsw_candidate_k=int(os.getenv("RETRIEVAL_HNSW_CANDIDATES", "20")),
            hnsw_ppr_alpha=float(os.getenv("GRAPH_RETRIEVAL_PPR_ALPHA", "0.5")),
        )
        packed_passages = [
            retriever.passage_context_record(node_id, score)
            for node_id, score in ranked
        ]
        text_context = generator.textualize_ranked_passages(query, packed_passages)
        G_aligned = builder.G.subgraph(
            [node_id for node_id, _ in ranked]
        ).copy()
        answer = generator.generate_answer(query, G_aligned, textual_context=text_context)
        st.write("### Answer")
        st.write(answer)
        with st.expander("Retrieved source passages"):
            st.text(text_context)

