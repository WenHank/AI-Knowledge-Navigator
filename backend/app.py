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
import psutil
import os

import uvicorn

from run_graph import agent_runner
from llmrouter_agents.state_management import AgentState
from miner_pdf_to import do_parse
from graph_db import create_graphrag_from_markdown_folder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state for managing resources
app_state = {
    "timing_stats": {
        "total_requests": 0,
        "total_processing_time": 0.0,
        "average_processing_time": 0.0,
        "min_processing_time": float('inf'),
        "max_processing_time": 0.0,
        "agent_usage": {},
        "error_count": 0,
        "successful_requests": 0,
    },
    "start_time": datetime.utcnow(),
    "request_count": 0
}

def update_timing_stats(processing_time: float, agent_used: str, success: bool):
    """Update global timing statistics"""
    stats = app_state["timing_stats"]
    
    stats["total_requests"] += 1
    
    if success:
        stats["successful_requests"] += 1
        stats["total_processing_time"] += processing_time
        stats["average_processing_time"] = stats["total_processing_time"] / stats["successful_requests"]
        stats["min_processing_time"] = min(stats["min_processing_time"], processing_time)
        stats["max_processing_time"] = max(stats["max_processing_time"], processing_time)
        
        # Track agent usage
        if agent_used not in stats["agent_usage"]:
            stats["agent_usage"][agent_used] = {"count": 0, "total_time": 0.0}
        stats["agent_usage"][agent_used]["count"] += 1
        stats["agent_usage"][agent_used]["total_time"] += processing_time
        stats["agent_usage"][agent_used]["average_time"] = (
            stats["agent_usage"][agent_used]["total_time"] / stats["agent_usage"][agent_used]["count"]
        )
    else:
        stats["error_count"] += 1

def print_timing_statistics():
    """Print comprehensive timing statistics"""
    stats = app_state["timing_stats"]
    uptime = (datetime.utcnow() - app_state["start_time"]).total_seconds()
    
    print("\n" + "="*60)
    print("📊 KNOWLEDGE NAVIGATOR API - PERFORMANCE SUMMARY")
    print("="*60)
    print(f"⏰ Uptime: {uptime:.1f}s ({uptime/3600:.2f}h)")
    print(f"🚀 Startup Time: {app_state.get('startup_time', 0):.3f}s")
    print(f"🔧 Workflow Init Time: {app_state.get('workflow_init_time', 0):.3f}s")
    print(f"📈 Total Requests: {stats['total_requests']}")
    print(f"✅ Successful: {stats['successful_requests']}")
    print(f"❌ Errors: {stats['error_count']}")
    print(f"📊 Success Rate: {(stats['successful_requests']/stats['total_requests']*100):.1f}%" if stats['total_requests'] > 0 else "N/A")
    
    if stats["successful_requests"] > 0:
        print(f"\n🕒 TIMING STATISTICS:")
        print(f"   Average Processing Time: {stats['average_processing_time']:.3f}s")
        print(f"   Min Processing Time: {stats['min_processing_time']:.3f}s")
        print(f"   Max Processing Time: {stats['max_processing_time']:.3f}s")
        print(f"   Requests/Hour: {stats['total_requests'] / (uptime/3600):.1f}" if uptime > 0 else "N/A")
        
        print(f"\n🤖 AGENT USAGE:")
        for agent, data in stats["agent_usage"].items():
            print(f"   {agent}: {data['count']} requests ({data['average_time']:.3f}s avg)")
    
    print("="*60)

app = FastAPI(
    title="Knowledge Navigator API",
    description="""
    🧠 **Knowledge Navigator API** - Intelligent Query Processing System
    
    This API provides intelligent query processing using a multi-agent workflow with comprehensive performance monitoring:
    
    ## Features
    - **Smart Query Classification**: Automatically categorizes queries as simple or complex
    - **Adaptive Processing**: Routes queries to appropriate processing agents
    - **Document Retrieval**: Fetches relevant documents for context
    - **Structured Responses**: Returns comprehensive, well-formatted results
    - **Error Handling**: Robust error management with detailed feedback
    - **Performance Monitoring**: Built-in request tracking and detailed timing analytics
    - **Real-time Statistics**: Live performance metrics and agent usage tracking
    
    ## Workflow
    1. **Preprocessing**: Query analysis and classification
    2. **Routing**: Smart routing to local or cloud-based agents
    3. **Processing**: Context-aware response generation
    4. **Response**: Structured output with metadata and timing information
    """,
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
    include_timing: Optional[bool] = Field(
        True,
        description="Whether to include detailed timing information"
    )
    
    @validator('user_query')
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError('Query cannot be empty or whitespace only')
        return v.strip()


class TimingInfo(BaseModel):
    """Detailed timing information"""
    
    total_execution_time: float = Field(..., description="Total execution time in seconds")
    preprocessing_time: Optional[float] = Field(None, description="Preprocessing time in seconds")
    agent_processing_time: Optional[float] = Field(None, description="Agent processing time in seconds")
    postprocessing_time: Optional[float] = Field(None, description="Postprocessing time in seconds")
    queue_time: Optional[float] = Field(None, description="Time spent in queue")
    tokens_per_second: Optional[float] = Field(None, description="Generation speed in tokens/second")
    input_tokens: Optional[int] = Field(None, description="Number of input tokens")
    output_tokens: Optional[int] = Field(None, description="Number of output tokens")


class ProcessingMetadata(BaseModel):
    """Metadata about the processing"""
    
    session_id: Optional[str]
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: str = Field(..., description="ISO timestamp of the response")
    request_id: str = Field(..., description="Unique request identifier")
    routing_type: str = Field(..., description="Processing route taken")
    agent_used: str = Field(..., description="Which agent processed the request")
    timing_details: Optional[TimingInfo] = Field(None, description="Detailed timing breakdown")


class ProcessingData(BaseModel):
    """Data structure for successful processing results"""
    
    user_query: str = Field(..., description="Original user query")
    final_answer: str = Field(..., description="Generated response to the query")
    routing_type: Optional[int] = Field(None, description="Routing classification (1=simple, 2=complex)")
    current_node: Optional[str] = Field(None, description="Current processing node")
    node_status: Optional[Dict[str, str]] = Field(None, description="Status of processing nodes")
    execution_summary: Optional[Dict[str, Any]] = Field(None, description="Execution summary")
    processed_query: Optional[str] = Field(None, description="Preprocessed version of the query")
    classification: Optional[int] = Field(None, description="Query complexity classification (1=simple, 2=complex)")
    retrieved_docs: Optional[List[str]] = Field(None, description="Retrieved document snippets")
    confidence_score: Optional[float] = Field(None, description="Confidence in the response (0-1)")
    token_usage: Optional[Dict[str, int]] = Field(None, description="Token usage statistics")


class QueryResponse(BaseModel):
    """Response model for query processing"""
    
    success: bool = Field(..., description="Whether the request was successful")
    data: Optional[ProcessingData] = Field(None, description="Processing results")
    error: Optional[str] = Field(None, description="Error message if any")
    metadata: ProcessingMetadata = Field(..., description="Processing metadata")


class PerformanceStats(BaseModel):
    """Performance statistics model"""
    
    total_requests: int
    successful_requests: int
    error_count: int
    success_rate: float
    average_processing_time: float
    min_processing_time: float
    max_processing_time: float
    uptime_seconds: float
    requests_per_hour: float
    agent_usage: Dict[str, Dict[str, Any]]


class HealthResponse(BaseModel):
    """Health check response model"""
    
    status: str
    timestamp: str
    uptime_seconds: float
    total_requests: int
    version: str
    system_info: Dict[str, Any]
    performance_stats: Optional[PerformanceStats] = None


# Utility functions
def generate_request_id() -> str:
    """Generate a unique request ID"""
    return f"req_{uuid.uuid4().hex[:12]}"


def get_system_info() -> Dict[str, Any]:
    """Get system information"""
    try:
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        }
    except Exception as e:
        return {"error": str(e)}


def extract_agent_used(result: Dict[str, Any]) -> str:
    """Extract which agent was used from the result"""
    # Check current_node first
    if "current_node" in result:
        return result["current_node"]
    
    # Check node_status for completed nodes
    if "node_status" in result:
        for node, status in result["node_status"].items():
            if status == "completed":
                return node
    
    # Check routing_type
    if "routing_type" in result:
        routing = result["routing_type"]
        if isinstance(routing, dict):
            if "router_decision" in routing:
                return "router_agent"
            elif "type" in routing:
                return f"type_{routing['type']}_agent"
    
    # Default fallback
    return "unknown_agent"


def convert_workflow_result_to_processing_data(result: Dict[str, Any]) -> ProcessingData:
    """Convert workflow result to ProcessingData format"""
    
    # Extract routing_type - it should now be a simple integer
    routing_type = result.get("routing_type")
    if isinstance(routing_type, dict):
        # Handle legacy format if it's still a dict
        if "type" in routing_type:
            classification = routing_type["type"]
        elif "router_decision" in routing_type and isinstance(routing_type["router_decision"], dict):
            classification = routing_type["router_decision"].get("type")
        else:
            classification = 1
        routing_type_value = classification
    else:
        # Handle new format - simple integer
        routing_type_value = routing_type if routing_type in [1, 2] else 1
        classification = routing_type_value
    
    return ProcessingData(
        user_query=result.get("user_query", ""),
        final_answer=result.get("final_answer", ""),
        routing_type=routing_type_value,
        current_node=result.get("current_node"),
        node_status=result.get("node_status"),
        execution_summary=result.get("execution_summary"),
        classification=classification,
        retrieved_docs=result.get("retrieved_docs", []),
        processed_query=result.get("processed_query"),
        confidence_score=result.get("confidence_score"),
        token_usage=result.get("token_usage")
    )


async def process_query_async(initial_state: AgentState) -> Dict[str, Any]:
    """Asynchronously process query using the agent workflow"""
    if agent_runner is None:
        raise RuntimeError("Workflow not initialized")
    
    try:
        # Add timing to the state
        initial_state["start_time"] = time.time()
        
        # Run in executor to prevent blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, agent_runner.invoke, initial_state)
        
        # Add total execution time if not present
        if "timing" not in result:
            result["timing"] = {}
        if "total_execution_time" not in result["timing"]:
            result["timing"]["total_execution_time"] = time.time() - initial_state["start_time"]
        
        return result
    except Exception as e:
        logger.error(f"Agent processing failed: {e}")
        raise


# Background task functions
async def log_request_success(request_id: str, processing_time: float, agent_used: str):
    """Log successful request processing"""
    logger.info(f"✅ Request {request_id} processed successfully by {agent_used} in {processing_time:.3f}s")
    update_timing_stats(processing_time, agent_used, True)


async def log_request_error(request_id: str, error: str, processing_time: float):
    """Log failed request processing"""
    logger.error(f"❌ Request {request_id} failed after {processing_time:.3f}s: {error}")
    update_timing_stats(processing_time, "error", False)


# API Endpoints
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    🏥 **Health Check**
    
    Get comprehensive system health and performance statistics.
    """
    uptime = (datetime.utcnow() - app_state["start_time"]).total_seconds()
    stats = app_state["timing_stats"]
    
    # Create performance stats
    performance_stats = None
    if stats["total_requests"] > 0:
        performance_stats = PerformanceStats(
            total_requests=stats["total_requests"],
            successful_requests=stats["successful_requests"],
            error_count=stats["error_count"],
            success_rate=stats["successful_requests"] / stats["total_requests"] * 100,
            average_processing_time=stats["average_processing_time"],
            min_processing_time=stats["min_processing_time"] if stats["min_processing_time"] != float('inf') else 0.0,
            max_processing_time=stats["max_processing_time"],
            uptime_seconds=uptime,
            requests_per_hour=stats["total_requests"] / (uptime / 3600) if uptime > 0 else 0.0,
            agent_usage=stats["agent_usage"]
        )
    
    return HealthResponse(
        status="healthy" if agent_runner is not None else "initializing",
        timestamp=datetime.utcnow().isoformat(),
        uptime_seconds=round(uptime, 2),
        total_requests=stats["total_requests"],
        version="2.1.0",
        system_info=get_system_info(),
        performance_stats=performance_stats
    )


@app.get("/stats", response_model=Dict[str, Any], tags=["System"])
async def get_performance_stats():
    """
    📊 **Performance Statistics**
    
    Get detailed performance and timing statistics.
    """
    return {
        "system_info": get_system_info(),
        "timing_stats": app_state["timing_stats"],
        "uptime_seconds": (datetime.utcnow() - app_state["start_time"]).total_seconds(),
        "status": "healthy" if agent_runner is not None else "initializing",
        "workflow_initialized": agent_runner is not None,
    }


@app.post("/query", response_model=QueryResponse, tags=["Query Processing"])
async def process_query(
    request: QueryRequest, 
    background_tasks: BackgroundTasks,
    http_request: Request
):
    """
    🧠 **Process User Query**
    
    Process user queries using the intelligent multi-agent workflow with comprehensive timing analysis.
    
    The system automatically:
    - Analyzes and classifies your query
    - Routes to the most appropriate processing agent
    - Retrieves relevant context documents
    - Generates comprehensive, accurate responses
    - Provides detailed timing and performance metrics
    
    **Example Usage:**
    ```json
    {
        "user_query": "Explain quantum computing in simple terms",
        "session_id": "my_session_123",
        "max_response_length": 300,
        "include_timing": true
    }
    ```
    """
    start_time = time.time()
    request_id = generate_request_id()
    timestamp = datetime.utcnow().isoformat()
    
    # Track request
    app_state["request_count"] = app_state.get("request_count", 0) + 1
    
    logger.info(f"🔄 Processing request {request_id}: {request.user_query[:100]}...")
    
    try:
        # Prepare initial state
        preprocessing_start = time.time()
        initial_state: AgentState = {
            "user_query": request.user_query,
            "max_response_length": request.max_response_length,
            "request_id": request_id,
        }
        preprocessing_time = time.time() - preprocessing_start
        
        # Process query asynchronously
        agent_start = time.time()
        result = await process_query_async(initial_state)
        agent_time = time.time() - agent_start
        
        # Post-processing
        postprocessing_start = time.time()
        
        # Extract agent used and calculate processing time
        agent_used = extract_agent_used(result)
        processing_time = time.time() - start_time
        
        # Convert workflow result to ProcessingData format
        processing_data = convert_workflow_result_to_processing_data(result)
        
        postprocessing_time = time.time() - postprocessing_start
        
        # Create detailed timing info
        timing_details = None
        if request.include_timing:
            timing_from_result = result.get("timing", {})
            timing_details = TimingInfo(
                total_execution_time=processing_time,
                preprocessing_time=preprocessing_time,
                agent_processing_time=agent_time,
                postprocessing_time=postprocessing_time,
                tokens_per_second=timing_from_result.get("tokens_per_second"),
                input_tokens=timing_from_result.get("input_tokens"),
                output_tokens=timing_from_result.get("output_tokens")
            )
        
        # Create metadata
        metadata = ProcessingMetadata(
            session_id=request.session_id,
            processing_time=round(processing_time, 3),
            timestamp=timestamp,
            request_id=request_id,
            routing_type=f"type_{result.get('routing_type', 1)}",
            agent_used=agent_used,
            timing_details=timing_details
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


# Testing endpoint for development
@app.post("/test", tags=["Testing"])
async def test_workflow():
    """
    🧪 **Test Workflow**
    
    Test the workflow with a simple query for development purposes.
    """
    if agent_runner is None:
        raise HTTPException(status_code=503, detail="Workflow not initialized")
    
    try:
        initial_state = {
            "user_query": "What is the weather today?",
        }
        
        start_time = time.time()
        result = agent_runner.invoke(initial_state)
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "raw_result": result,  # Return the full raw result for debugging
            "processing_time": round(processing_time, 3),
            "agent_used": extract_agent_used(result)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@app.post("/pdf_save_graph_db", tags=["System"])
async def pdf_save_graph_db():
    """
    Save PDF data to the graph database.
    """
    try:
        from graph_db import save_pdf_to_neo4j
        save_pdf_to_neo4j()
        return {"success": True, "message": "PDF data saved to graph database successfully."}
    except Exception as e:
        logger.error(f"Error saving PDF data to graph database: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving PDF data: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)