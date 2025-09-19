from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
import time
import uuid
import logging
from datetime import datetime
import traceback
import asyncio
from contextlib import asynccontextmanager

from run_graph import agent_runner
from agents.state_management import AgentState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state for managing resources
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("🚀 Starting Knowledge Navigator API...")
    app_state["start_time"] = datetime.utcnow()
    app_state["request_count"] = 0
    
    # Initialize agent runner if needed
    try:
        # You can add any initialization logic here
        logger.info("✅ Application startup complete")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down Knowledge Navigator API...")
    # Cleanup resources if needed
    logger.info("✅ Shutdown complete")

app = FastAPI(
    title="Knowledge Navigator API",
    description="""
    🧠 **Knowledge Navigator API** - Intelligent Query Processing System
    
    This API provides intelligent query processing using a multi-agent workflow:
    
    ## Features
    - **Smart Query Classification**: Automatically categorizes queries as simple or complex
    - **Adaptive Processing**: Routes queries to appropriate processing agents
    - **Document Retrieval**: Fetches relevant documents for context
    - **Structured Responses**: Returns comprehensive, well-formatted results
    - **Error Handling**: Robust error management with detailed feedback
    - **Performance Monitoring**: Built-in request tracking and timing
    
    ## Workflow
    1. **Preprocessing**: Query analysis and classification
    2. **Routing**: Smart routing to local or cloud-based agents
    3. **Processing**: Context-aware response generation
    4. **Response**: Structured output with metadata
    """,
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Request/Response Models
class QueryRequest(BaseModel):
    """Request model for query processing"""
    
    user_query: str = Field(
        ..., 
        description="The user's query to process",
        min_length=1,
        max_length=10000,
        example="What is machine learning and how does it work?"
    )
    session_id: Optional[str] = Field(
        None, 
        description="Optional session identifier for tracking",
        example="session_123"
    )
    max_response_length: Optional[int] = Field(
        500,
        description="Maximum length of the response",
        ge=50,
        le=2000
    )
    include_sources: Optional[bool] = Field(
        True,
        description="Whether to include source documents in response"
    )
    
    @validator('user_query')
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError('Query cannot be empty or whitespace only')
        return v.strip()


class ProcessingMetadata(BaseModel):
    """Metadata about the processing"""
    
    session_id: Optional[str]
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: str = Field(..., description="ISO timestamp of the response")
    request_id: str = Field(..., description="Unique request identifier")
    routing_type: str = Field(..., description="Processing route taken")
    agent_used: str = Field(..., description="Which agent processed the request")


class ProcessingData(BaseModel):
    """Data structure for successful processing results"""
    
    user_query: str
    processed_query: Optional[str] = Field(None, description="Preprocessed version of the query")
    classification: int = Field(..., description="Query complexity classification (1=simple, 2=complex)")
    retrieved_docs: List[str] = Field(default_factory=list, description="Retrieved document snippets")
    final_answer: str = Field(..., description="Generated response to the query")
    confidence_score: Optional[float] = Field(None, description="Confidence in the response (0-1)")
    token_usage: Optional[Dict[str, int]] = Field(None, description="Token usage statistics")


class QueryResponse(BaseModel):
    """Response model for query processing"""
    
    success: bool = Field(..., description="Whether the request was successful")
    data: Optional[ProcessingData] = Field(None, description="Processing results")
    error: Optional[str] = Field(None, description="Error message if any")
    metadata: ProcessingMetadata = Field(..., description="Processing metadata")


class HealthResponse(BaseModel):
    """Health check response model"""
    
    status: str
    timestamp: str
    uptime_seconds: float
    total_requests: int
    version: str
    system_info: Dict[str, Any]


# Utility functions
def generate_request_id() -> str:
    """Generate a unique request ID"""
    return f"req_{uuid.uuid4().hex[:12]}"


def calculate_confidence_score(state: AgentState) -> float:
    """Calculate confidence score based on processing results"""
    score = 0.8  # Base confidence
    
    # Adjust based on error presence
    if any(key.endswith('_error') for key in state.keys()):
        score -= 0.2
    
    # Adjust based on retrieved documents
    retrieved_docs = state.get('retrieved_docs', [])
    if len(retrieved_docs) > 0:
        score += 0.1
    
    # Adjust based on response length
    final_answer = state.get('final_answer', '')
    if isinstance(final_answer, str) and len(final_answer) > 50:
        score += 0.05
    
    return min(1.0, max(0.1, score))


async def process_query_async(initial_state: AgentState) -> Dict[str, Any]:
    """Asynchronously process query using the agent workflow"""
    try:
        # Run in executor to prevent blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, agent_runner.invoke, initial_state)
        return result
    except Exception as e:
        logger.error(f"Agent processing failed: {e}")
        raise


# API Endpoints
@app.post("/query", response_model=QueryResponse, tags=["Query Processing"])
async def process_query(
    request: QueryRequest, 
    background_tasks: BackgroundTasks,
    http_request: Request
):
    """
    🧠 **Process User Query**
    
    Process user queries using the intelligent multi-agent workflow.
    
    The system automatically:
    - Analyzes and classifies your query
    - Routes to the most appropriate processing agent
    - Retrieves relevant context documents
    - Generates comprehensive, accurate responses
    
    **Example Usage:**
    ```json
    {
        "user_query": "Explain quantum computing in simple terms",
        "session_id": "my_session_123",
        "max_response_length": 300,
        "include_sources": true
    }
    ```
    """
    start_time = time.time()
    request_id = generate_request_id()
    timestamp = datetime.utcnow().isoformat()
    
    # Track request
    app_state["request_count"] = app_state.get("request_count", 0) + 1
    
    logger.info(f"Processing request {request_id}: {request.user_query[:100]}...")
    
    try:
        # Prepare initial state
        initial_state: AgentState = {
            "user_query": request.user_query,
            "max_response_length": request.max_response_length,
            "include_sources": request.include_sources,
            "request_id": request_id,
        }
        
        # Process query asynchronously
        result = await process_query_async(initial_state)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Determine which agent was used
        agent_used = "unknown"
        if result.get("response_source") == "local_model":
            agent_used = "local_agent"
        elif result.get("response_source") == "openrouter_fallback":
            agent_used = "openrouter_agent"
        elif "current_node" in result:
            agent_used = result["current_node"]
        
        # Calculate confidence score
        confidence_score = calculate_confidence_score(result)
        
        # Structure the response data
        processing_data = ProcessingData(
            user_query=result.get("user_query", request.user_query),
            processed_query=result.get("processed_query"),
            classification=int(result.get("routing_type", "1")),
            retrieved_docs=result.get("retrieved_docs", []) if request.include_sources else [],
            final_answer=result.get("final_answer", "No response generated"),
            confidence_score=confidence_score,
            token_usage=result.get("token_usage")
        )
        
        # Create metadata
        metadata = ProcessingMetadata(
            session_id=request.session_id,
            processing_time=round(processing_time, 3),
            timestamp=timestamp,
            request_id=request_id,
            routing_type=result.get("routing_type", "1"),
            agent_used=agent_used
        )
        
        # Log successful processing in background
        background_tasks.add_task(
            log_request_success, 
            request_id, 
            processing_time, 
            agent_used
        )
        
        logger.info(f"✅ Request {request_id} completed in {processing_time:.3f}s")
        
        return QueryResponse(
            success=True,
            data=processing_data,
            error=None,
            metadata=metadata
        )
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        processing_time = time.time() - start_time
        
        # Log error details
        error_details = f"Request {request_id} failed: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_details)
        
        # Create error metadata
        metadata = ProcessingMetadata(
            session_id=request.session_id,
            processing_time=round(processing_time, 3),
            timestamp=timestamp,
            request_id=request_id,
            routing_type="error",
            agent_used="none"
        )
        
        # Log failed processing in background
        background_tasks.add_task(
            log_request_error, 
            request_id, 
            str(e), 
            processing_time
        )
        
        return QueryResponse(
            success=False,
            data=None,
            error=f"Processing failed: {str(e)}",
            metadata=metadata
        )


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    ❤️ **Health Check**
    
    Get system health status and basic metrics.
    """
    import psutil
    import sys
    
    current_time = datetime.utcnow()
    start_time = app_state.get("start_time", current_time)
    uptime = (current_time - start_time).total_seconds()
    
    return HealthResponse(
        status="healthy",
        timestamp=current_time.isoformat(),
        uptime_seconds=round(uptime, 1),
        total_requests=app_state.get("request_count", 0),
        version="2.0.0",
        system_info={
            "python_version": sys.version.split()[0],
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "available_memory_gb": round(psutil.virtual_memory().available / (1024**3), 2)
        }
    )


@app.get("/", tags=["System"])
async def root():
    """
    🏠 **API Information**
    
    Get basic API information and available endpoints.
    """
    return {
        "message": "🧠 Knowledge Navigator API",
        "description": "Intelligent query processing with multi-agent workflow",
        "version": "2.0.0",
        "documentation": {
            "interactive_docs": "/docs",
            "redoc": "/redoc"
        },
        "endpoints": {
            "process_query": "POST /query - Main query processing endpoint",
            "health_check": "GET /health - System health and metrics",
            "api_info": "GET / - This endpoint"
        },
        "status": "🟢 Online"
    }


@app.get("/stats", tags=["System"])
async def get_stats():
    """
    📊 **System Statistics**
    
    Get detailed system statistics and performance metrics.
    """
    return {
        "requests": {
            "total": app_state.get("request_count", 0),
            "uptime_hours": round((datetime.utcnow() - app_state.get("start_time", datetime.utcnow())).total_seconds() / 3600, 2)
        },
        "system": {
            "start_time": app_state.get("start_time", datetime.utcnow()).isoformat(),
            "current_time": datetime.utcnow().isoformat()
        }
    }


# Background task functions
async def log_request_success(request_id: str, processing_time: float, agent_used: str):
    """Log successful request processing"""
    logger.info(f"SUCCESS: {request_id} | {processing_time:.3f}s | {agent_used}")


async def log_request_error(request_id: str, error: str, processing_time: float):
    """Log failed request processing"""
    logger.error(f"ERROR: {request_id} | {processing_time:.3f}s | {error}")


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url.path)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url.path)
        }
    )


# Run the application
if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting Knowledge Navigator API server...")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True,
        reload_excludes=["*.log", "*.db", "__pycache__/*"]
    )