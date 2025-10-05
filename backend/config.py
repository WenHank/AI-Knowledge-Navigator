from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any

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
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()

class GraphQueryResponse(BaseModel):
    success: bool
    query: str
    answer: str
    sources: Optional[list[str]] = None
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
