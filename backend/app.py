"""
Complete API: Document Upload + GraphDB Query System
Two main endpoints: /upload (file to GraphDB) and /query (search GraphDB)
"""

import os
import uuid
import shutil
import time
import asyncio
import traceback
import psutil
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn
import logging
import torch

# Document processing imports
from miner_pdf_to import read_fn, do_parse

# GraphRAG imports
from llama_index.core import Settings, PropertyGraphIndex, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

UPLOAD_DIR = Path("./uploads")
OUTPUT_DIR = Path("./output")
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".jp2"}

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# GLOBAL STATE
# ============================================================================

app_state = {
    "timing_stats": {
        "total_requests": 0,
        "documents_processed": 0,
        "documents_indexed": 0,
        "queries_processed": 0,
    },
    "start_time": datetime.now(datetime.UTC) if hasattr(datetime, 'UTC') else datetime.utcnow(),
    "processing_jobs": {},
    "embed_model": None,
    "custom_llm": None,
    "graph_store": None,
    "query_engine": None,
}

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class Neo4jConfig(BaseModel):
    username: str = "neo4j"
    password: str = "password"
    url: str = "bolt://localhost:7687"
    database: str = "neo4j"

class UploadResponse(BaseModel):
    success: bool
    job_id: str
    filename: str
    message: str
    status: str

class ProcessingStatus(BaseModel):
    job_id: str
    status: str
    message: str
    markdown_path: Optional[str] = None
    graph_indexed: bool = False
    processing_time: Optional[float] = None

class GraphQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000, description="User query to search the knowledge graph")
    max_results: Optional[int] = Field(5, ge=1, le=20, description="Maximum number of results")
    include_sources: Optional[bool] = Field(True, description="Include source documents")
    
    @validator('query')
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()

class GraphQueryResponse(BaseModel):
    success: bool
    query: str
    answer: str
    sources: Optional[List[str]] = None
    metadata: Dict[str, Any]
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    uptime_seconds: float
    documents_processed: int
    documents_indexed: int
    queries_processed: int
    graph_connected: bool
    version: str

# ============================================================================
# GLOBAL CONFIG
# ============================================================================

neo4j_config = Neo4jConfig()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_embed_model():
    """Lazy load embedding model"""
    if app_state["embed_model"] is None:
        logger.info("Loading embedding model...")
        app_state["embed_model"] = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-mpnet-base-v2",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        Settings.embed_model = app_state["embed_model"]
        logger.info("Embedding model loaded successfully")
    return app_state["embed_model"]

def get_custom_llm():
    """Lazy load custom LLM"""
    if app_state.get("custom_llm") is None:
        logger.info("Loading custom LLM...")
        from custom_llm import TheCustomLLM  # Import your custom LLM
        app_state["custom_llm"] = TheCustomLLM()
        Settings.llm = app_state["custom_llm"]
        logger.info("Custom LLM loaded successfully")
    return app_state["custom_llm"]

def get_graph_store():
    """Get or create Neo4j graph store connection"""
    if app_state["graph_store"] is None:
        try:
            logger.info("Connecting to Neo4j...")
            app_state["graph_store"] = Neo4jPropertyGraphStore(
                username=neo4j_config.username,
                password=neo4j_config.password,
                url=neo4j_config.url,
                database=neo4j_config.database
            )
            logger.info("Connected to Neo4j successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise HTTPException(status_code=503, detail=f"Neo4j connection failed: {str(e)}")
    return app_state["graph_store"]

def validate_file_extension(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def get_system_info() -> Dict[str, Any]:
    try:
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
        }
    except Exception as e:
        return {"error": str(e)}

# ============================================================================
# DOCUMENT PROCESSING FUNCTIONS
# ============================================================================

async def process_file_to_markdown(
    file_path: Path,
    job_id: str,
    start_page: int = 0,
    end_page: Optional[int] = None
) -> str:
    """Convert file to markdown using MinerU"""
    try:
        app_state["processing_jobs"][job_id]["status"] = "processing"
        app_state["processing_jobs"][job_id]["message"] = "Converting to markdown..."
        
        logger.info(f"Converting {file_path.name} to markdown...")
        
        # Read file bytes
        pdf_bytes = read_fn(file_path)
        file_stem = file_path.stem
        output_path = OUTPUT_DIR / file_stem
        
        # Parse the document
        do_parse(
            output_dir=str(output_path),
            pdf_file_names=[file_stem],
            pdf_bytes_list=[pdf_bytes],
            p_lang_list=["en"],
            backend="pipeline",
            parse_method="auto",
            formula_enable=True,
            table_enable=True,
            f_dump_md=True,
            f_dump_middle_json=False,
            f_dump_model_output=False,
            f_dump_orig_pdf=False,
            f_dump_content_list=False,
            start_page_id=start_page,
            end_page_id=end_page
        )
        
        # Find the generated markdown file
        md_file = output_path / f"{file_stem}.md"
        if not md_file.exists():
            raise FileNotFoundError(f"Markdown file not generated: {md_file}")
        
        app_state["processing_jobs"][job_id]["markdown_path"] = str(md_file)
        app_state["timing_stats"]["documents_processed"] += 1
        
        logger.info(f"Markdown conversion completed: {md_file}")
        return str(md_file)
        
    except Exception as e:
        app_state["processing_jobs"][job_id]["status"] = "failed"
        app_state["processing_jobs"][job_id]["message"] = f"Conversion failed: {str(e)}"
        logger.error(f"Markdown conversion failed: {e}")
        raise

async def index_markdown_to_graph(markdown_path: str, job_id: str):
    """Index markdown content to Neo4j graph database"""
    try:
        app_state["processing_jobs"][job_id]["status"] = "indexing"
        app_state["processing_jobs"][job_id]["message"] = "Indexing to graph database..."
        
        logger.info(f"Indexing {markdown_path} to Neo4j...")
        
        # Load embedding model AND custom LLM
        get_embed_model()
        get_custom_llm()
        
        # Get graph store
        graph_store = get_graph_store()
        
        # Read markdown content
        md_path = Path(markdown_path)
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if not content.strip():
            raise ValueError("Markdown file is empty")
        
        # Create document
        document = Document(
            text=content,
            metadata={
                "filename": md_path.name,
                "job_id": job_id,
                "source_path": str(md_path),
                "indexed_at": (datetime.now(datetime.UTC) if hasattr(datetime, 'UTC') else datetime.utcnow()).isoformat()
            }
        )
        
        # Build or update graph index using async method
        logger.info("Building PropertyGraph index...")
        
        # Use asyncio.create_task or run in executor
        loop = asyncio.get_event_loop()
        index = await loop.run_in_executor(
            None,
            lambda: PropertyGraphIndex.from_documents(
                [document],
                property_graph_store=graph_store,
                llm=app_state["custom_llm"],
                embed_model=app_state["embed_model"],
                embed_kg_nodes=True,
                show_progress=True
            )
        )
        
        # Update query engine with latest index
        app_state["query_engine"] = index.as_query_engine(
            include_text=True,
            response_mode="tree_summarize",
            llm=app_state["custom_llm"]
        )
        
        app_state["processing_jobs"][job_id]["status"] = "completed"
        app_state["processing_jobs"][job_id]["message"] = "Successfully indexed to graph database"
        app_state["processing_jobs"][job_id]["graph_indexed"] = True
        app_state["timing_stats"]["documents_indexed"] += 1
        
        logger.info(f"Successfully indexed {md_path.name} to Neo4j")
        return True
        
    except Exception as e:
        app_state["processing_jobs"][job_id]["status"] = "failed"
        app_state["processing_jobs"][job_id]["message"] = f"Graph indexing failed: {str(e)}"
        app_state["processing_jobs"][job_id]["graph_indexed"] = False
        logger.error(f"Graph indexing failed: {e}")
        traceback.print_exc()
        raise
# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Document Processing & Knowledge Graph API",
    description="""
    Complete document processing and knowledge graph query system.
    
    ## Workflow:
    1. Upload PDF/image files via `/upload`
    2. System converts to markdown and indexes to Neo4j
    3. Query the knowledge graph via `/query_graph`
    
    ## Features:
    - Automatic document conversion (MinerU)
    - Neo4j graph database storage
    - Intelligent query processing with GraphRAG
    - Background processing for uploads
    """,
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", tags=["System"])
async def root():
    return {
        "message": "Document Processing & Knowledge Graph API",
        "version": "2.0.0",
        "endpoints": {
            "health": "GET /health",
            "upload": "POST /upload",
            "query_graph": "POST /query_graph",
            "status": "GET /status/{job_id}",
            "config": "GET/POST /config/neo4j"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """System health check"""
    current_time = datetime.now(datetime.UTC) if hasattr(datetime, 'UTC') else datetime.utcnow()
    uptime = (current_time - app_state["start_time"]).total_seconds()
    stats = app_state["timing_stats"]
    
    # Check Neo4j connection
    graph_connected = False
    try:
        if app_state["graph_store"] is not None:
            graph_connected = True
    except:
        pass
    
    return HealthResponse(
        status="healthy",
        timestamp=current_time.isoformat(),
        uptime_seconds=round(uptime, 2),
        documents_processed=stats["documents_processed"],
        documents_indexed=stats["documents_indexed"],
        queries_processed=stats["queries_processed"],
        graph_connected=graph_connected,
        version="2.0.0"
    )

@app.post("/upload", response_model=UploadResponse, tags=["Document Processing"])
async def upload_and_index(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    start_page: int = Form(0),
    end_page: Optional[int] = Form(None),
):
    """
    Upload a file, convert to markdown, and index to Neo4j GraphDB
    
    Args:
        file: PDF or image file
        start_page: Starting page number (default: 0)
        end_page: Ending page number (default: None = all pages)
    
    Returns:
        Job ID for tracking processing status
    """
    
    # Validate file extension
    if not validate_file_extension(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    file_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")
    
    # Create processing job
    app_state["processing_jobs"][job_id] = {
        "status": "pending",
        "message": "Job created",
        "filename": file.filename,
        "markdown_path": None,
        "graph_indexed": False,
        "created_at": time.time()
    }
    
    # Background processing pipeline
    async def process_pipeline():
        start_time = time.time()
        try:
            # Step 1: Convert to markdown
            md_path = await process_file_to_markdown(file_path, job_id, start_page, end_page)
            
            # Step 2: Index to Neo4j
            await index_markdown_to_graph(md_path, job_id)
            
            processing_time = time.time() - start_time
            app_state["processing_jobs"][job_id]["processing_time"] = processing_time
            
            logger.info(f"Job {job_id} completed in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Processing pipeline failed for job {job_id}: {e}")
            traceback.print_exc()
    
    background_tasks.add_task(process_pipeline)
    
    return UploadResponse(
        success=True,
        job_id=job_id,
        filename=file.filename,
        message="File uploaded. Processing started in background.",
        status="pending"
    )

@app.get("/status/{job_id}", response_model=ProcessingStatus, tags=["Document Processing"])
async def get_job_status(job_id: str):
    """Get processing status for a job"""
    job = app_state["processing_jobs"].get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return ProcessingStatus(
        job_id=job_id,
        status=job["status"],
        message=job["message"],
        markdown_path=job.get("markdown_path"),
        graph_indexed=job.get("graph_indexed", False),
        processing_time=job.get("processing_time")
    )

@app.post("/query_graph", response_model=GraphQueryResponse, tags=["Knowledge Graph Query"])
async def query_knowledge_graph(request: GraphQueryRequest):
    """
    Query the knowledge graph with natural language
    
    Args:
        request: Query request with user query and options
    
    Returns:
        Answer from the knowledge graph with sources and metadata
    """
    
    start_time = time.time()
    current_time = datetime.now(datetime.UTC) if hasattr(datetime, 'UTC') else datetime.utcnow()
    
    # Check if query engine is initialized
    if app_state["query_engine"] is None:
        raise HTTPException(
            status_code=503,
            detail="No documents indexed yet. Please upload documents first via /upload"
        )
    
    try:
        logger.info(f"Processing graph query: {request.query[:100]}...")
        
        # Query the graph
        response = app_state["query_engine"].query(request.query)
        
        # Extract answer
        answer = str(response)
        
        # Extract sources if requested
        sources = None
        if request.include_sources:
            sources = []
            if hasattr(response, 'source_nodes'):
                for node in response.source_nodes[:request.max_results]:
                    source_text = node.text[:200] + "..." if len(node.text) > 200 else node.text
                    sources.append(source_text)
        
        processing_time = time.time() - start_time
        app_state["timing_stats"]["queries_processed"] += 1
        
        logger.info(f"Query processed in {processing_time:.2f}s")
        
        return GraphQueryResponse(
            success=True,
            query=request.query,
            answer=answer,
            sources=sources,
            metadata={
                "processing_time": round(processing_time, 3),
                "timestamp": current_time.isoformat(),
                "documents_in_graph": app_state["timing_stats"]["documents_indexed"]
            },
            error=None
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Query failed: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        
        return GraphQueryResponse(
            success=False,
            query=request.query,
            answer="",
            sources=None,
            metadata={
                "processing_time": round(processing_time, 3),
                "timestamp": current_time.isoformat()
            },
            error=error_msg
        )

@app.get("/config/neo4j", tags=["Configuration"])
async def get_neo4j_config():
    """Get current Neo4j configuration (password hidden)"""
    return {
        "username": neo4j_config.username,
        "password": "***" if neo4j_config.password else None,
        "url": neo4j_config.url,
        "database": neo4j_config.database,
        "connected": app_state["graph_store"] is not None
    }

@app.post("/config/neo4j", tags=["Configuration"])
async def update_neo4j_config(config: Neo4jConfig):
    """Update Neo4j configuration (requires restart of connections)"""
    global neo4j_config
    neo4j_config = config
    
    # Reset connections to force reconnection with new config
    app_state["graph_store"] = None
    app_state["query_engine"] = None
    
    logger.info("Neo4j configuration updated. Connections will be re-established on next use.")
    
    return {
        "message": "Neo4j configuration updated successfully",
        "config": {
            "username": config.username,
            "url": config.url,
            "database": config.database
        }
    }

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Starting Document Processing & Knowledge Graph API")
    print("=" * 60)
    print(f"📁 Upload Directory: {UPLOAD_DIR.absolute()}")
    print(f"📄 Output Directory: {OUTPUT_DIR.absolute()}")
    print(f"🔗 Neo4j URL: {neo4j_config.url}")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )