# 
import os
import pandas as pd
from datetime import datetime
from pathlib import Path

# LangChain / Vector RAG Imports
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

# Your Custom Logic Imports
from custom_llm import TheCustomLLM
from graph_db import build_optimized_graph # Updated name from our last fix

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
MARKDOWN_PATH = "./output/Report.md"
PDF_PATH = "./uploads/Report.pdf"
CHROMA_PDF_DIR = "./chroma_db_pdf_naive"
CHROMA_MD_DIR = "./chroma_db_markdown_struct"

NEO4J_CONF = {
    "username": "neo4j", 
    "password": "password", 
    "url": "bolt://localhost:7687"
}

# Initialize shared resources
llm = TheCustomLLM()
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ==========================================
# 2. VECTOR DB BUILDERS
# ==========================================
def ensure_vector_dbs():
    """Builds Chroma DBs only if they don't exist."""
    
    # PDF Baseline
    if not os.path.exists(CHROMA_PDF_DIR):
        print("📦 Building Baseline PDF RAG...")
        loader = PyPDFLoader(PDF_PATH)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = splitter.split_documents(pages)
        Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_PDF_DIR)
        print(f"✅ PDF RAG saved ({len(docs)} chunks)")

    # Markdown Optimized
    if not os.path.exists(CHROMA_MD_DIR):
        print("📦 Building Structured Markdown RAG...")
        with open(MARKDOWN_PATH, "r", encoding="utf-8") as f:
            md_text = f.read()
        md_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        md_docs = md_splitter.split_text(md_text)
        Chroma.from_documents(md_docs, embeddings, persist_directory=CHROMA_MD_DIR)
        print(f"✅ Markdown RAG saved ({len(md_docs)} chunks)")

# ==========================================
# 3. EVALUATION ENGINE
# ==========================================
def run_battle_thee(query, index_graphrag):
    """Executes the 3-way RAG Battle."""
    
    # Load retrievers
    db_pdf = Chroma(persist_directory=CHROMA_PDF_DIR, embedding_function=embeddings)
    db_md = Chroma(persist_directory=CHROMA_MD_DIR, embedding_function=embeddings)
    
    # Retrieve Context
    context_pdf = "\n".join([d.page_content for d in db_pdf.similarity_search(query, k=3)])
    context_md = "\n".join([d.page_content for d in db_md.similarity_search(query, k=3)])
    
    # Generate Answers
    def get_ans(ctx, q):
        p = f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer briefly based ONLY on context:"
        return llm.complete(p).text # Use our fixed .complete method

    ans_pdf = get_ans(context_pdf, query)
    ans_md = get_ans(context_md, query)
    
    # GraphRAG Answer
    query_engine = index_graphrag.as_query_engine(response_mode="tree_summarize")
    ans_graph = str(query_engine.query(query))

    # Judge Phase
    judge_prompt = f"""
    Compare these 3 answers for the query: {query}
    A (Naive): {ans_pdf}
    B (Markdown): {ans_md}
    C (Graph): {ans_graph}
    Rank them and explain why in JSON format: {{"winner": "", "reasoning": ""}}
    """
    judgment = llm.complete(judge_prompt).text

    return {
        "query": query, 
        "pdf": ans_pdf, 
        "markdown": ans_md, 
        "graph": ans_graph, 
        "judgment": judgment
    }

def run_battle_two(query):
    """Executes the 2-way RAG Battle."""
    
    # Load retrievers
    db_pdf = Chroma(persist_directory=CHROMA_PDF_DIR, embedding_function=embeddings)
    db_md = Chroma(persist_directory=CHROMA_MD_DIR, embedding_function=embeddings)
    
    # Retrieve Context
    context_pdf = "\n".join([d.page_content for d in db_pdf.similarity_search(query, k=3)])
    context_md = "\n".join([d.page_content for d in db_md.similarity_search(query, k=3)])
    
    # Generate Answers
    def get_ans(ctx, q):
        p = f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer briefly based ONLY on context:"
        return llm.complete(p).text # Use our fixed .complete method

    ans_pdf = get_ans(context_pdf, query)
    ans_md = get_ans(context_md, query)
    
    # Judge Phase
    judge_prompt = f"""
    Compare these 3 answers for the query: {query}
    A (Naive): {ans_pdf}
    B (Markdown): {ans_md}
    Rank them and explain why in JSON format: {{"winner": "", "reasoning": ""}}
    """
    judgment = llm.complete(judge_prompt).text

    return {
        "query": query, 
        "pdf": ans_pdf, 
        "markdown": ans_md, 
        "judgment": judgment
    }

# ==========================================
# 4. MAIN EXECUTION FLOW
# ==========================================
if __name__ == "__main__":
    # Step 1: Prepare Vector DBs
    ensure_vector_dbs()

    # Step 2: Prepare GraphRAG (Neo4j)
    # print("🚀 Initializing GraphRAG Index...")
    # index_graphrag = build_optimized_graph(MARKDOWN_PATH, NEO4J_CONF)

    # Step 3: Run the Evaluation
    test_queries = [
        "What are the core technical risks mentioned?",
        "Summarize the overall conclusion of the report.",
        "What specific technologies are driving the 2026 growth?"
    ]

    print(f"\n⚔️ Starting Battle for {len(test_queries)} queries...")
    results = []
    for q in test_queries:
        print(f"Processing: {q}")
        results.append(run_battle_two(q))

    # Step 4: Export
    df = pd.DataFrame(results)
    csv_name = f"battle_results_{datetime.now().strftime('%m%d_%H%M')}.csv"
    df.to_csv(csv_name, index=False)
    print(f"\n📊 Done! Results in {csv_name}")