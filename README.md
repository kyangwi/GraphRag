# Knowledge Graph RAG FastAPI Application

This FastAPI application provides a simple REST API interface to query documents using a combination of knowledge graph and vector search technologies.

## Features

- **Document Processing**: Automatically processes `.docx` files and creates semantic chunks
- **Vector Storage**: Supports both in-memory and persistent ChromaDB vector storage
- **Knowledge Graph**: Uses Neo4j to store and query relationships between entities
- **Embedding Caching**: Efficient caching system to avoid regenerating embeddings
- **Graph Retrieval**: Advanced retrieval strategy combining vector similarity and graph traversal
- **RESTful API**: Clean REST API with automatic documentation

## Quick Start

### 1. Environment Setup

Create a `.env` file in the project root with your credentials:

```env
# Google AI API
GOOGLE_API_KEY=your_google_api_key_here

# Neo4j Database
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password_here
```

### 2. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

### 3. Add Documents

Place your `.docx` files in the `core/files/` directory:

```bash
mkdir -p core/files
# Copy your .docx files to the core/files directory
cp your_documents/*.docx core/files/
```

### 4. Start the API

```bash
# Simple start
python start_api.py

# Or with custom options
python start_api.py --port 8080 --reload --log-level debug
```

The API will be available at `http://localhost:8000`

**Note**: By default, the system uses in-memory vector storage and does not clean the Neo4j database on startup. This provides faster startup times and preserves existing graph data.

## API Endpoints

### GET `/`
Basic API information and status

### GET `/status`
Get system status including:
- Initialization status
- Vector store type (ChromaDB or In-Memory)
- Graph statistics (nodes and relationships)
- Error information if any

**Response:**
```json
{
  "initialized": true,
  "vector_store_type": "InMemoryVectorStore",
  "documents_processed": null,
  "graph_nodes": 150,
  "graph_relationships": 75,
  "error": null
}
```

### POST `/query` or GET `/query`
Query the knowledge graph RAG system

**POST Request Body:**
```json
{
  "query": "Who must approve exceptions to the Malware Policy?"
}
```

**GET Request:**
```
GET /query?query=Who must approve exceptions to the Malware Policy?
```

**Response:**
```json
{
  "answer": "According to the policy documents, exceptions to the Malware and Antivirus Policy must be approved by the Chief Information Security Officer (CISO) at [Company Name].",
  "query": "Who must approve exceptions to the Malware Policy?",
  "metadata": {
    "timestamp": 1703123456.789,
    "vector_store_type": "InMemoryVectorStore"
  }
}
```

### POST `/reinitialize`
Reinitialize the system (useful after adding new documents)

**Parameters:**
- `force_reprocess`: Force reprocessing of all documents (default: false)
- `clean_neo4j`: Clean Neo4j database before reprocessing (default: false)
- `use_chromadb`: Use ChromaDB for storage (default: auto-detect)

## System Architecture

### File Structure
```
wolfia/
├── core/
│   ├── api.py                          # FastAPI application
│   ├── main_with_llmgraphtranformer.py # Main processing pipeline
│   ├── embedding_cache.py              # Embedding caching system
│   ├── document_tracker.py             # Document processing tracker
│   ├── config.py                       # Configuration settings
│   ├── files/                          # Document storage directory
│   └── storage/                        # Persistent storage
│       ├── chromadb/                   # ChromaDB storage (when enabled)
│       ├── embedding_cache.json        # Embedding cache file
│       └── document_tracking.json      # Document tracking file
├── start_api.py                        # API startup script
├── requirements.txt                    # Dependencies
└── README_API.md                       # This documentation
```

### Document Processing Pipeline

1. **Document Loading**: Loads `.docx` files from the `core/files/` directory
2. **Semantic Chunking**: Uses embedding-based semantic chunking for optimal text segmentation
3. **Embedding Generation**: Creates embeddings using Google's Gemini embedding model with caching
4. **Vector Storage**: Stores embeddings in either ChromaDB (persistent) or in-memory vector store
5. **Graph Processing**: Extracts entities and relationships using LLM Graph Transformer and stores in Neo4j
6. **Retrieval Setup**: Configures graph-based retrieval with vector similarity fallback

### Query Processing

1. **Query Reception**: API receives user query
2. **Graph Retrieval**: Uses graph traversal strategy to find relevant documents
3. **Vector Fallback**: Falls back to vector similarity if graph retrieval insufficient
4. **Context Assembly**: Combines retrieved documents into context
5. **Answer Generation**: Uses LLM to generate answer based on context
6. **Response Formatting**: Returns structured response with metadata

## Configuration Options

### Vector Storage

The system supports both storage options:
- **In-Memory**: Default storage, faster startup, no persistence, better for development and testing
- **ChromaDB**: Persistent storage, survives restarts, better for production (can be enabled via reinitialize endpoint)

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_API_KEY` | Google AI API key for embeddings and LLM | Yes |
| `NEO4J_URI` | Neo4j database URI | Yes |
| `NEO4J_USERNAME` | Neo4j username | Yes |
| `NEO4J_PASSWORD` | Neo4j password | Yes |

## Usage Examples

### Using curl

```bash
# Query via GET
curl "http://localhost:8000/query?query=What is the password policy?"

# Query via POST
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the password policy?"}'

# Check system status
curl "http://localhost:8000/status"

# Reinitialize with new documents
curl -X POST "http://localhost:8000/reinitialize?force_reprocess=true"
```

### Using Python requests

```python
import requests

# Query the system
response = requests.post(
    "http://localhost:8000/query",
    json={"query": "What are the data retention requirements?"}
)
result = response.json()
print(result["answer"])

# Check status
status = requests.get("http://localhost:8000/status").json()
print(f"System initialized: {status['initialized']}")
print(f"Graph nodes: {status['graph_nodes']}")
```

### Using the Interactive Documentation

Visit `http://localhost:8000/docs` for the automatically generated interactive API documentation where you can test all endpoints directly in your browser.

## Performance and Caching

### Embedding Cache
- Automatically caches embeddings to avoid regeneration using `core/embedding_cache.py`
- Persistent cache stored in `core/storage/embedding_cache.json`
- Provides cache hit rate statistics
- Significantly reduces processing time for repeated content

### Document Tracking
- Tracks processed documents by file hash using `core/document_tracker.py`
- Tracking data stored in `core/storage/document_tracking.json`
- Only processes new or changed documents
- Maintains processing metadata and statistics

### Storage Locations
- **ChromaDB**: `core/storage/chromadb/` (when enabled)
- **Document Cache**: `core/storage/embedding_cache.json`
- **Document Tracking**: `core/storage/document_tracking.json`
- **Documents**: `core/files/` directory

## Troubleshooting

### Common Issues

1. **Missing Environment Variables**
   ```
   Error: System not initialized: Missing required environment variables
   ```
   Solution: Ensure all required environment variables are set in `.env` file

2. **No Documents Found**
   ```
   Warning: No .docx files found in core/files directory
   ```
   Solution: Add `.docx` files to the `core/files/` directory

3. **Neo4j Connection Error**
   ```
   Error connecting to Neo4j: Failed to establish connection
   ```
   Solution: Ensure Neo4j is running and credentials are correct

4. **ChromaDB Issues**
   ```
   ChromaDB is not available. Please install it with: pip install chromadb
   ```
   Solution: Install ChromaDB or use in-memory storage

### Debugging

Enable debug logging:
```bash
python start_api.py --log-level debug
```

Check system status:
```bash
curl http://localhost:8000/status
```

## Development

### Adding New Endpoints

The FastAPI application is modular and can be easily extended. Add new endpoints to `core/api.py`:

```python
@app.post("/new-endpoint")
async def new_endpoint():
    # Your implementation here
    pass
```

### Main Processing Module

The core processing logic is implemented in `core/main_with_llmgraphtranformer.py`, which includes:
- Document loading and processing
- LLM Graph Transformer for entity and relationship extraction
- Vector store initialization (in-memory or ChromaDB)
- Neo4j graph database integration
- RAG chain setup with graph retrieval capabilities

### Customizing Retrieval

The system uses LLM Graph Transformer for entity and relationship extraction. Modify the retrieval strategy in `core/main_with_llmgraphtranformer.py`:

```python
# Customize LLM Graph Transformer settings
llm_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Person", "Organization", "Technology", "Policy"],
    allowed_relationships=["MANAGES", "IMPLEMENTS", "REQUIRES", "APPROVES"]
)

# Customize graph retrieval settings
vector_retriever.search_kwargs = {"k": 10}  # Number of documents to retrieve
```

## Security Considerations

- Store sensitive credentials in environment variables, not in code
- Use HTTPS in production environments
- Consider implementing API authentication for production use
- Regularly update dependencies for security patches

## Monitoring and Logging

The application provides comprehensive logging for:
- System initialization status
- Document processing progress
- Query processing times
- Cache hit rates
- Error conditions

Monitor these logs to ensure optimal performance and catch issues early. 