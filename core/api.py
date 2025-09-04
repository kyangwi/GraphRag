"""
FastAPI Application for Knowledge Graph RAG System

This API provides a simple interface to query the knowledge graph RAG system.
It initializes the vector store and graph database on startup and provides
a single endpoint to answer questions based on the processed documents.
"""

import os
import asyncio
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Import the main processing pipeline
from core.main_with_llmgraphtranformer import (
    main as initialize_system,
    test_system,
    check_chromadb_availability
)

# Load environment variables
load_dotenv()

# Global variables to store the initialized system
chain = None
vector_store = None
graph = None
system_initialized = False
initialization_error = None

class QueryRequest(BaseModel):
    query: str
    
class QueryResponse(BaseModel):
    answer: str
    query: str
    metadata: Optional[Dict[str, Any]] = None

class SystemStatus(BaseModel):
    initialized: bool
    error: Optional[str] = None
    vector_store_type: Optional[str] = None
    documents_processed: Optional[int] = None
    graph_nodes: Optional[int] = None
    graph_relationships: Optional[int] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager - initializes the system on startup
    """
    global chain, vector_store, graph, system_initialized, initialization_error
    
    print("Starting FastAPI application...")
    print("Initializing Knowledge Graph RAG system...")
    
    try:
        # Determine if ChromaDB is available
        # use_chromadb = check_chromadb_availability()
        # if use_chromadb:
        #     print("ChromaDB is available - using persistent vector storage")
        # else:
        #     print("ChromaDB not available - using in-memory vector storage")
        
        # Initialize the system
        chain, vector_store, graph = await asyncio.to_thread(
            initialize_system,
            force_reprocess=False,  # Only process new/changed documents
            clean_neo4j=False,      # Don't clean Neo4j on startup
            use_chromadb=False
        )
        
        system_initialized = True
        print("System initialized successfully!")
        
        # Run a quick test to verify everything is working
        try:
            await asyncio.to_thread(test_system, chain, graph)
            print("System test completed successfully!")
        except Exception as test_error:
            print(f"System test failed (but system is still functional): {test_error}")
        
    except Exception as e:
        initialization_error = str(e)
        print(f"Failed to initialize system: {e}")
        system_initialized = False
    
    yield
    
    # Cleanup on shutdown
    print("Shutting down FastAPI application...")

# Create FastAPI app with lifespan manager
app = FastAPI(
    title="Knowledge Graph RAG API",
    description="API for querying documents using knowledge graph and vector search",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/", response_model=Dict[str, str])
async def root():
    """
    Root endpoint with basic API information
    """
    return {
        "message": "Knowledge Graph RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "/status"
    }

@app.get("/status", response_model=SystemStatus)
async def get_status():
    """
    Get the current status of the system
    """
    global chain, vector_store, graph, system_initialized, initialization_error
    
    status = SystemStatus(
        initialized=system_initialized,
        error=initialization_error
    )
    
    if system_initialized and graph:
        try:
            # Get graph statistics
            nodes_result = await asyncio.to_thread(
                graph.query, "MATCH (n) RETURN count(n) as node_count"
            )
            rels_result = await asyncio.to_thread(
                graph.query, "MATCH ()-[r]->() RETURN count(r) as rel_count"
            )
            
            status.graph_nodes = nodes_result[0]["node_count"] if nodes_result else 0
            status.graph_relationships = rels_result[0]["rel_count"] if rels_result else 0
            
            # Determine vector store type
            if hasattr(vector_store, 'persist_directory'):
                status.vector_store_type = "ChromaDB"
            else:
                status.vector_store_type = "InMemoryVectorStore"
                
        except Exception as e:
            status.error = f"Error getting system stats: {str(e)}"
    
    return status

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the knowledge graph RAG system
    
    Args:
        request: QueryRequest containing the query string
        
    Returns:
        QueryResponse with the answer and metadata
    """
    global chain, system_initialized, initialization_error
    
    # Check if system is initialized
    if not system_initialized:
        raise HTTPException(
            status_code=503,
            detail=f"System not initialized: {initialization_error or 'Unknown error'}"
        )
    
    if not chain:
        raise HTTPException(
            status_code=500,
            detail="RAG chain not available"
        )
    
    try:
        # Process the query
        answer = await asyncio.to_thread(chain.invoke, request.query)
        
        # Prepare response
        response = QueryResponse(
            answer=answer,
            query=request.query,
            metadata={
                "timestamp": asyncio.get_event_loop().time(),
                "vector_store_type": "ChromaDB" if hasattr(vector_store, 'persist_directory') else "InMemoryVectorStore"
            }
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@app.get("/query", response_model=QueryResponse)
async def query_documents_get(query: str = Query(..., description="The question to ask the system")):
    """
    Query the knowledge graph RAG system via GET request
    
    Args:
        query: The question to ask the system
        
    Returns:
        QueryResponse with the answer and metadata
    """
    request = QueryRequest(query=query)
    return await query_documents(request)

@app.post("/reinitialize")
async def reinitialize_system(
    force_reprocess: bool = Query(False, description="Force reprocessing of all documents"),
    clean_neo4j: bool = Query(False, description="Clean Neo4j database before reprocessing"),
    use_chromadb: bool = Query(None, description="Use ChromaDB (auto-detect if None)")
):
    """
    Reinitialize the system (useful for reprocessing documents)
    """
    global chain, vector_store, graph, system_initialized, initialization_error
    
    try:
        print("Reinitializing system...")
        
        # Determine ChromaDB usage
        if use_chromadb is None:
            use_chromadb = check_chromadb_availability()
        
        # Reinitialize the system
        chain, vector_store, graph = await asyncio.to_thread(
            initialize_system,
            force_reprocess=force_reprocess,
            clean_neo4j=clean_neo4j,
            use_chromadb=use_chromadb
        )
        
        system_initialized = True
        initialization_error = None
        
        return {
            "message": "System reinitialized successfully",
            "force_reprocess": force_reprocess,
            "clean_neo4j": clean_neo4j,
            "use_chromadb": use_chromadb
        }
        
    except Exception as e:
        initialization_error = str(e)
        system_initialized = False
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reinitialize system: {str(e)}"
        )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler for unexpected errors
    """
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "path": str(request.url)
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    # Run the FastAPI app
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 