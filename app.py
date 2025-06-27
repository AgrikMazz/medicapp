import streamlit as st
import os
from modules.chunker import Chunker
from modules.groq_handler import GroqHandler
from modules.graph_builder import GraphBuilder
from modules.generator import GraphAugmentedGenerator
from modules.entity_handler import EntityHandler
from modules.evaluator import NodeRAGEvaluator
from transformers import GPT2TokenizerFast
from dotenv import load_dotenv
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")

chunker = Chunker()
groq = GroqHandler()
generator = GraphAugmentedGenerator(groq)
builder = GraphBuilder(
    groq_handler=groq,
    jsonl_path="data/chunked_output.jsonl",
    neo4j_uri=NEO4J_URI,
    neo4j_user=NEO4J_USER,
    neo4j_password=NEO4J_PASSWORD,
    neo4j_database=NEO4J_DATABASE
)
entity_handler = EntityHandler(groq, builder, generator)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
evaluator = NodeRAGEvaluator(chunker, builder, generator, entity_handler, tokenizer)

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
        text = chunker.pdf_to_text("temp.pdf")
        chunks = chunker.segment_into_chunks(text)
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
        builder.build_graph()
        builder.detect_and_save_communities()
        st.success("Graph and communities built!")

elif option == "Generate HO Nodes & Semantic Edges":
    with st.spinner("Enhancing graph..."):
        builder.add_HO_nodes_from_community_map()
        builder.finalize_graph_G3_from_community_map()
        builder.build_G4_text_attachment()
        builder.build_G5_semantic_hnsw_index()
        builder.save_pickle()
        st.success("Graph enriched and saved.")

elif option == "Run QA Query":
    query = st.text_input("Enter your question:")
    if query:
        builder.load_graph_pickle("graph.pkl")
        entry_nodes, p_weights = builder.get_entry_points(query)
        G_raw, _ = builder.run_shallow_ppr(entry_nodes, p_weights)
        rc = builder.reasoning_chain(query, G_raw)
        model = ...  # Load NodeAlignerGNN
        optimizer = ...
        G_aligned, _, _ = builder.align_nodes_via_gnn(query, G_raw, rc, model, optimizer, device="cpu")
        answer = generator.generate_answer(query, G_aligned)
        st.write("### Answer")
        st.write(answer)

