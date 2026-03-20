import os
import time
import torch
import json
from typing import Any, Dict, Optional, List, Literal
from pathlib import Path
import requests

from llama_index.core import Document, Settings, PropertyGraphIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor

from custom_llm import TheCustomLLM
# ==========================================
# 2. DATA PRE-PROCESSOR (The 28k Row Fix)
# ==========================================
def clean_and_load_markdown(file_path: str) -> List[Document]:
    """Cleans messy markdown rows to save LLM tokens."""
    path = Path(file_path)
    if not path.exists(): raise FileNotFoundError(path)

    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Filter: Keep lines with actual words, skip empty table rows like "| | |"
    cleaned_lines = [l for l in lines if len(l.strip()) > 10 and any(c.isalpha() for c in l)]
    
    # Re-group into larger logical chunks (e.g., every 50 lines) to reduce LLM calls
    chunk_size = 50
    grouped_chunks = ["".join(cleaned_lines[i:i+chunk_size]) for i in range(0, len(cleaned_lines), chunk_size)]
    
    return [Document(text=chunk, metadata={"source": path.name}) for chunk in grouped_chunks]

# ==========================================
# 3. CORE GRAPHRAG BUILDER
# ==========================================
def build_optimized_graph(markdown_path: str, neo4j_config: Dict):
    llm = TheCustomLLM()
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")
    
    Settings.llm = llm
    Settings.embed_model = embed_model

    # Connect to Neo4j
    graph_store = Neo4jPropertyGraphStore(
        username=neo4j_config["username"],
        password=neo4j_config["password"],
        url=neo4j_config["url"]
    )

    # SCHEMA EXTRACTION: Forces Mistral to only look for specific things
    # This prevents the LLM from hallucinating useless relationships
    entities = Literal["COMPANY", "FINANCIAL_METRIC", "TECHNOLOGY", "MARKET_REGION"]
    relations = Literal["REPORTS", "DEVELOPED", "OPERATES_IN", "INCREASED_BY"]
    
    kg_extractor = SchemaLLMPathExtractor(
        llm=llm,
        possible_entities=entities,
        possible_relations=relations,
        strict=False # Allow some flexibility
    )

    # Check for existing data
    try:
        index = PropertyGraphIndex.from_existing(property_graph_store=graph_store)
        print("✅ Found existing graph. Skipping build.")
    except:
        print("🔨 Building graph from cleaned data...")
        documents = clean_and_load_markdown(markdown_path)
        index = PropertyGraphIndex.from_documents(
            documents,
            property_graph_store=graph_store,
            kg_extractors=[kg_extractor],
            show_progress=True
        )
    
    return index

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    NEO4J_CONF = {"username": "neo4j", "password": "password", "url": "bolt://localhost:7687"}
    MD_FILE = "./output/Report.md"

    index = build_optimized_graph(MD_FILE, NEO4J_CONF)
    
    query_engine = index.as_query_engine(include_text=True, response_mode="tree_summarize")
    print(f"\n🔍 Query Result: {query_engine.query('What was the revenue growth for 3nm technology?')}")