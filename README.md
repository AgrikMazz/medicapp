# 📚 Graph-Based RAG from PDF

This project implements a **graph-based retrieval-augmented generation (RAG)** framework that transforms any **PDF document into a semantic graph**, enabling intelligent, multi-hop **question answering** over its contents using LLMs.

---

## 🚀 Features

✅ Upload any PDF  
✅ Automatically extract and chunk text  
✅ Construct an entity or concept-level graph  
✅ Enable multi-hop QA using graph traversal  
✅ Flexible LLM backend (OpenAI, local, etc.)  
✅ Easily extendable pipeline  

---

## 🧠 How It Works

1. **PDF Ingestion**  
   - Extracts clean text from PDF files using `pdfminer` or `PyMuPDF`.

2. **Chunking & Embedding**  
   - Splits long text into meaningful passages (recursive/semantic splitting).
   - Converts chunks into vector embeddings.

3. **Graph Construction**  
   - Uses entity extraction, co-reference resolution, and sentence linking.
   - Nodes = concepts or sections  
   - Edges = relations (e.g. entity co-occurrence, citations, textual similarity)

4. **Graph Indexing**  
   - Stores embeddings and graph metadata for retrieval.

5. **QA Pipeline**  
   - User inputs a question.
   - Relevant subgraphs are retrieved.
   - Prompts are dynamically composed using retrieved chunks.
   - An LLM generates the final answer.
