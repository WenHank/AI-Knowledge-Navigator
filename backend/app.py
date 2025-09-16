from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import time
from datetime import datetime
import traceback

from run_graph import agent_runner
from agents.state_management import AgentState

app = FastAPI(
    title="Chat History Analysis API",
    description="API for processing user queries using LangGraph agent workflow",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    """Request model for chat analysis"""

    user_query: str = Field(
        ..., description="The user's query to process", min_length=1
    )
    session_id: Optional[str] = Field(None, description="Optional session identifier")


class QueryResponse(BaseModel):
    """Response model for chat analysis"""

    success: bool = Field(..., description="Whether the request was successful")
    data: Optional[dict] = Field(None, description="Processing results")
    error: Optional[str] = Field(None, description="Error message if any")
    timestamp: str = Field(..., description="ISO timestamp of the response")
    processing_time: Optional[float] = Field(
        None, description="Processing time in seconds"
    )


class ProcessingData(BaseModel):
    """Data structure for successful processing results"""

    user_query: str
    processed_query: str
    classification: int
    retrieved_docs: List[str]
    final_answer: str


@app.post("/chathistory_analyze", response_model=QueryResponse)
async def process_input(request: QueryRequest):
    """
    Process user queries using the LangGraph agent workflow

    This endpoint:
    1. Preprocesses the user query
    2. Classifies complexity (1=simple, 2=complex)
    3. Routes to appropriate processing agent
    4. Returns structured results with retrieved documents and final answer
    """
    start_time = time.time()
    timestamp = datetime.utcnow().isoformat()

    try:
        # Validate input
        if not request.user_query.strip():
            raise HTTPException(status_code=400, detail="user_query cannot be empty")

        # Prepare initial state for the agent
        initial_state: AgentState = {"user_query": request.user_query.strip()}

        # Run the agent workflow
        result = agent_runner.invoke(initial_state)

        # Calculate processing time
        processing_time = time.time() - start_time

        # Structure the response data
        processing_data = {
            "user_query": result.get("user_query", request.user_query),
            "processed_query": result.get("processed_query", ""),
            "classification": result.get("classification", 1),
            "retrieved_docs": result.get("retrieved_docs", []),
            "final_answer": result.get("final_answer", "No response generated"),
            "session_id": request.session_id,
        }

        return QueryResponse(
            success=True,
            data=processing_data,
            error=None,
            timestamp=timestamp,
            processing_time=processing_time,
        )

    except Exception as e:
        # Calculate processing time even for errors
        processing_time = time.time() - start_time

        # Log the full error for debugging
        error_details = f"Error: {str(e)}\nTraceback: {traceback.format_exc()}"
        print(error_details)  # You might want to use proper logging here

        # Return structured error response
        return QueryResponse(
            success=False,
            data=None,
            error=f"Processing failed: {str(e)}",
            timestamp=timestamp,
            processing_time=processing_time,
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Chat History Analysis API",
    }


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Chat History Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "/chathistory_analyze (POST)",
            "health": "/health (GET)",
            "docs": "/docs (GET)",
        },
    }


# Example usage and testing
if __name__ == "__main__":
    import uvicorn

    # Run the FastAPI server
    uvicorn.run(
        "main:app",  # Change this to your actual module name
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
