# AI Knowledge Navigator

An end-to-end AI system that transforms static PDFs into a structured knowledge graph and enables intelligent question answering using GraphRAG and LLM routing.

The platform extracts knowledge from documents, builds a Neo4j knowledge graph, and dynamically routes user queries to the most suitable reasoning pipeline to generate precise and context-aware answers.

---

## 🚀 Features

- 📄 **PDF Knowledge Extraction**
  - Convert unstructured PDF documents into structured JSON knowledge.

- 🧠 **Knowledge Graph Construction**
  - Automatically transform extracted knowledge into a **Neo4j graph database**.

- 🔍 **GraphRAG Retrieval**
  - Retrieve relevant information through graph traversal instead of traditional vector search.

- 🤖 **LLM Router**
  - Classifies user queries and routes them to the appropriate reasoning pipeline.

- 💬 **Natural Language Q&A**
  - Users can ask questions in natural language and receive context-aware answers.

---

## 🏗 System Architecture

<p align="center">
  <img src="docs/architecture.png" width="900">
</p>

### Pipeline Overview

1. **PDF Ingestion**
   - PDF documents are parsed and converted into structured JSON.

2. **Knowledge Extraction**
   - Important entities and relationships are extracted using LLM-based parsing.

3. **Knowledge Graph Construction**
   - Extracted knowledge is converted into nodes and relationships inside **Neo4j**.

4. **LLM Router**
   - Incoming user queries are classified to determine the best retrieval strategy.

5. **GraphRAG Querying**
   - The system retrieves relevant graph context and feeds it into the LLM.

6. **Answer Generation**
   - The LLM generates responses using retrieved graph knowledge.

---

### ✅ Core System (Completed)

- [x] PDF ingestion pipeline
- [x] PDF → Markdown knowledge extraction
- [x] Knowledge graph construction with **Neo4j**
- [x] GraphRAG-based retrieval
- [x] Fine-tuned **LLM Router** for query classification
- [x] Intelligent routing between query pipelines
- [x] Natural language question-answering API
- [x] **LLM Router evaluation framework**
      
### 🔧 System Improvements (In Progress)

- [ ] **GraphRAG retrieval evaluation**
- [ ] Async ingestion pipeline for large document processing
- [ ] Logging and system observability

---

## 🎯 LLM Router Fine-Tuning & Optimization

To minimize system latency and slash API costs, I fine-tuned a **Llama-3-8B-Instruct** model using **LoRA (Low-Rank Adaptation)**. This model serves as the "intelligent gatekeeper," dynamically routing queries to the most efficient reasoning pipeline.

### **Training & Performance**
* **Base Model**: Llama-3-8B-Instruct.
* **Method**: LoRA Fine-tuning (balancing parameter updates with limited compute).
* **Final Training Loss**: **0.09**.
* **Core Objective**: Binary classification. The router identifies "Easy" tasks for the **EFFICIENT (Local)** pipeline and "Complex" tasks for the **CLOUD** pipeline.

### **Pushing the Limits: Hardware Challenges**
A key part of this project was engineering a solution within strict hardware bottlenecks. Training and running an 8B model on an **NVIDIA RTX 3060 Ti (8GB VRAM)** frequently triggered **Out-of-Memory (OOM)** errors during initial tests.

**How I resolved the constraints:**
* **4-bit Quantization**: Integrated `BitsAndBytes` to compress the model into 4-bit NormalFloat (NF4), reducing the memory footprint by roughly **75%**.
* **Precision Capping**: I strictly limited `max_new_tokens` (10 for routing, 250 for local inference) and tuned training steps to stay within the 8GB VRAM safety zone.
* **Resource Management**: Leveraged SSD-backed swap to handle peak loads during large-scale evaluation, preventing system crashes during long benchmark runs.

### **Real-World Impact (Evaluation Results)**
I built a custom benchmarking framework using the `routellm/gpt4_dataset` to validate the router's performance across 500 samples:

| Metric | Result |
| :--- | :--- |
| **Easy Task Routing Accuracy** | **95.0%** |
| **Overall System Accuracy** | **96.6%** |
| **Net Cloud Token Reduction** | **~38%** |

> **Developer Note:**
> Achieving **95% accuracy on easy tasks** means the vast majority of daily interactions are handled locally and instantly. This doesn't just save money—it drastically improves the user experience by providing near-zero latency responses for common queries.

### 📊 Evaluation (Planned)

To ensure reliability and answer quality, a structured evaluation pipeline will be introduced.

The evaluation framework will focus on measuring both **retrieval quality** and **LLM response quality**.

**Planned metrics include:**

- **Retrieval Accuracy**  
  Measures whether the GraphRAG pipeline retrieves the correct graph context.

- **Answer Relevance**  
  Evaluates how well the generated answer addresses the user's query.

- **Faithfulness (Hallucination Detection)**  
  Ensures the generated response is grounded in the retrieved knowledge graph.

Future work will include automated evaluation scripts and benchmark datasets to continuously assess system performance.

## 🧩 Tech Stack

| Component | Technology |
|--------|--------|
Backend API | FastAPI |
Knowledge Graph | Neo4j |
Document Processing | Python |
LLM Integration | OpenAI API |
Retrieval | GraphRAG |
Orchestration | Python |

## 🚧 Development Roadmap

This project is actively under development.  
The roadmap below highlights the current progress and planned improvements toward a production-ready AI knowledge system.

---
