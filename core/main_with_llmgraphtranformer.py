"""
Knowledge Graph RAG System with Vector Storage Options

This implementation supports both in-memory and persistent vector storage options:
- In-memory: Loads documents fresh on each run (default)
- ChromaDB: Persistent vector storage with collection management

Components:
- Document processing and tracking
- Vector storage (InMemoryVectorStore or ChromaDB)
- Neo4j graph database for relationships
- Graph-based retrieval with fallback to vector similarity
- Shredding transformer for document preprocessing (ChromaDB)
"""

import os
import glob
import time
import torch
from typing import List
from datetime import datetime
from graph_retriever.strategies import Eager
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import  ChatGoogleGenerativeAI

from langchain_graph_retriever import GraphRetriever
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
# from langchain_community.document_loaders import Docx2txtLoader
from langchain.document_loaders import PyPDFLoader

from langchain_experimental.text_splitter import SemanticChunker
from dotenv import load_dotenv
from core.document_tracker import DocumentTracker
from core.embedding_cache import EmbeddingCache, CachedEmbeddings
from core.config import FILES_DIRECTORY, CHROMADB_PERSIST_DIRECTORY, CHROMADB_COLLECTION_NAME, GOOGLE_API_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
from langchain.embeddings import HuggingFaceEmbeddings
load_dotenv()

# =============================================================================
# ENVIRONMENT SETUP AND CONFIGURATION
# =============================================================================

# Set up environment variables for Neo4j and Gemini



# =============================================================================
# INITIALIZE CORE COMPONENTS
# =============================================================================

# Initialize embeddings with caching and LLM
try:
    # Initialize the base embeddings model
    # base_embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07", google_api_key=GOOGLE_API_KEY)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_embeddings = HuggingFaceEmbeddings(
                model_name="intfloat/multilingual-e5-large",
                model_kwargs={"device": device}
            )
    # Initialize embedding cache
    embedding_cache = EmbeddingCache()

    
    # Create cached embeddings wrapper
    embeddings = CachedEmbeddings(base_embeddings, embedding_cache)
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash-preview-05-20", google_api_key=GOOGLE_API_KEY)
    
    print("Successfully initialized cached embeddings and LLM")
except Exception as e:
    print(f"Error initializing embeddings/LLM: {str(e)}")
    raise

# Initialize Neo4j graph
try:
    graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
    print("Successfully connected to Neo4j")
except Exception as e:
    print(f"Error connecting to Neo4j: {str(e)}")
    raise

# =============================================================================
# DOCUMENT PROCESSING FUNCTIONS
# =============================================================================

def load_and_chunk_docx_files(tracker: DocumentTracker, force_reprocess: bool = False) -> List[Document]:
    """
    Load and chunk PDF files, skipping already processed documents unless force_reprocess is True
    """
    documents = []
    pdf_files = glob.glob(os.path.join(FILES_DIRECTORY, "*.pdf"))
    print(pdf_files)
    
    if not pdf_files:
        print(f"No .pdf files found in {FILES_DIRECTORY} directory")
        return documents
    
    # Initialize semantic chunker
    try:
        text_splitter = SemanticChunker(embeddings)
    except Exception as e:
        print(f"Error initializing semantic chunker: {str(e)}")
        raise
    
    processed_count = 0
    skipped_count = 0
    
    for pdf_file in pdf_files:
        try:
            # Check if document needs processing
            if not force_reprocess and tracker.is_document_processed(pdf_file):
                print(f"Skipping {pdf_file}: already processed and unchanged")
                skipped_count += 1
                continue
            
            print(f"📄 Processing {pdf_file}...")
            
            # Use PyPDFLoader for PDF files (make sure to install pypdf: pip install pypdf)
            loader = PyPDFLoader(pdf_file)
            raw_docs = loader.load()
            
            if not raw_docs:
                print(f"⚠️  Warning: No content loaded from {pdf_file}")
                continue
            
            # Apply semantic chunking
            print(f"🔪 Applying semantic chunking (this generates embeddings for chunking)...")
            chunks = text_splitter.split_documents(raw_docs)
            print(f"✅ Created {len(chunks)} semantic chunks")
            
            # Add file metadata to chunks
            for chunk in chunks:
                chunk.metadata.update({
                    "source_file": os.path.basename(pdf_file),
                    "file_path": pdf_file,
                    "processed_at": datetime.now().isoformat()
                })
            
            documents.extend(chunks)
            processed_count += 1
            
            # Mark document as processed
            tracker.mark_document_processed(pdf_file, len(chunks))
            
            print(f"Processed {pdf_file}: {len(chunks)} chunks created")
            
        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")
            import traceback
            traceback.print_exc()  # This will give more detailed error information
            continue
        time.sleep(2)  # Reduced sleep time from 30 to 2 seconds
    
    print(f"Document processing complete: {processed_count} processed, {skipped_count} skipped")
    return documents[:10]

def sanitize_type(type_name: str) -> str:
    """Sanitize type names to be valid Neo4j labels/relationship types"""
    # Replace spaces and special characters with underscores
    sanitized = type_name.replace(" ", "_").replace("-", "_")
    # Remove any other special characters and keep only alphanumeric and underscores
    sanitized = ''.join(c if c.isalnum() or c == '_' else '_' for c in sanitized)
    # Remove consecutive underscores and strip leading/trailing underscores
    while '__' in sanitized:
        sanitized = sanitized.replace('__', '_')
    sanitized = sanitized.strip('_')
    # Ensure it starts with a letter (Neo4j requirement)
    if sanitized and not sanitized[0].isalpha():
        sanitized = 'Node_' + sanitized
    # Fallback if empty
    return sanitized if sanitized else 'UnknownType'

def process_graph_documents(graph_documents: List, incremental: bool = True):
    """
    Process graph documents and add to Neo4j with optional incremental updates
    """
    if not incremental:
        print("Cleaning Neo4j database...")
        try:
            graph.query("MATCH (n) DETACH DELETE n")
            print("Neo4j database cleaned successfully.")
        except Exception as e:
            print(f"Error cleaning Neo4j database: {str(e)}")
            raise
    
    print("Adding graph documents to Neo4j...")
    node_count = 0
    relationship_count = 0
    
    for i, graph_doc in enumerate(graph_documents):
        print(f"Processing graph document {i+1}/{len(graph_documents)}")
        
        # Add nodes
        for node in graph_doc.nodes:
            try:
                sanitized_type = sanitize_type(node.type)
                print(f"Adding node: {node.type} -> {sanitized_type} (ID: {node.id})")
                graph.query(
                    f"MERGE (n:{sanitized_type} {{id: $id}})",
                    {"id": node.id}
                )
                node_count += 1
            except Exception as e:
                print(f"Error adding node {node.id} of type {node.type}: {str(e)}")
                continue
        
        # Add relationships
        for rel in graph_doc.relationships:
            try:
                sanitized_rel_type = sanitize_type(rel.type)
                print(f"Adding relationship: {rel.type} -> {sanitized_rel_type} ({rel.source.id} -> {rel.target.id})")
                graph.query(
                    f"MATCH (s {{id: $source}}), (t {{id: $target}}) "
                    f"MERGE (s)-[:{sanitized_rel_type}]->(t)",
                    {"source": rel.source.id, "target": rel.target.id}
                )
                relationship_count += 1
            except Exception as e:
                print(f"Error adding relationship {rel.type} from {rel.source.id} to {rel.target.id}: {str(e)}")
                continue
    
    print(f"Successfully added {node_count} nodes and {relationship_count} relationships to Neo4j")

# =============================================================================
# VECTOR STORE INTEGRATION (IN-MEMORY AND CHROMADB)
# =============================================================================
# This system supports both in-memory and persistent ChromaDB vector storage.

def check_chromadb_availability() -> bool:
    """Check if ChromaDB is properly installed and available"""
    try:
        import chromadb
        from langchain_chroma.vectorstores import Chroma
        from langchain_graph_retriever.transformers import ShreddingTransformer
        return True
    except ImportError:
        return False

def initialize_in_memory_vector_store(documents: List[Document] = None) -> InMemoryVectorStore:
    """
    Initialize InMemoryVectorStore with optional documents.
    Note: This is a purely in-memory store that doesn't persist data between runs.
    """
    try:
        if documents:
            print(f"🏪 Creating in-memory vector store with {len(documents)} documents...")
            print(f"⚠️  WARNING: This will generate embeddings AGAIN for vector storage!")
            vector_store = InMemoryVectorStore.from_documents(
                documents=documents,
                embedding=embeddings,
            )
        else:
            print("🏪 Creating empty in-memory vector store...")
            vector_store = InMemoryVectorStore(embeddings)
        
        print("✅ Successfully initialized in-memory vector store")
        return vector_store
        
    except Exception as e:
        print(f"Error initializing in-memory vector store: {str(e)}")
        raise

def add_documents_to_in_memory_store(vector_store: InMemoryVectorStore, documents: List[Document]):
    """
    Add new documents to existing in-memory vector store.
    Note: Changes are only available during the current session.
    """
    if not documents:
        print("No documents to add to vector store")
        return
    
    try:
        print(f"Adding {len(documents)} documents to in-memory vector store...")
        vector_store.add_documents(documents)
        print("Successfully added documents to in-memory vector store")
    except Exception as e:
        print(f"Error adding documents to in-memory vector store: {str(e)}")
        raise

def initialize_chromadb_vector_store(documents: List[Document] = None, force_recreate: bool = False):
    """
    Initialize ChromaDB vector store with optional documents.
    Uses ShreddingTransformer for document preprocessing as recommended.
    """
    try:
        # Import ChromaDB dependencies
        from langchain_chroma.vectorstores import Chroma
        from langchain_graph_retriever.transformers import ShreddingTransformer
        
        # Ensure storage directory exists
        os.makedirs(CHROMADB_PERSIST_DIRECTORY, exist_ok=True)
        
        if documents:
            print(f"Creating ChromaDB vector store with {len(documents)} documents...")
            
            # Apply shredding transformation as recommended
            shredding_transformer = ShreddingTransformer()
            transformed_documents = list(shredding_transformer.transform_documents(documents))
            print(f"Applied shredding transformation: {len(documents)} -> {len(transformed_documents)} chunks")
            
            # Create or update ChromaDB store
            vector_store = Chroma.from_documents(
                documents=transformed_documents,
                embedding=embeddings,
                collection_name=CHROMADB_COLLECTION_NAME,
                persist_directory=CHROMADB_PERSIST_DIRECTORY,
            )
        else:
            print("Creating empty ChromaDB vector store...")
            vector_store = Chroma(
                embedding_function=embeddings,
                collection_name=CHROMADB_COLLECTION_NAME,
                persist_directory=CHROMADB_PERSIST_DIRECTORY,
            )
        
        print(f"Successfully initialized ChromaDB vector store (persist_directory: {CHROMADB_PERSIST_DIRECTORY})")
        return vector_store
        
    except Exception as e:
        print(f"Error initializing ChromaDB vector store: {str(e)}")
        raise

def add_documents_to_chromadb_store(vector_store, documents: List[Document]):
    """
    Add new documents to existing ChromaDB vector store.
    Applies shredding transformation before adding documents.
    """
    if not documents:
        print("No documents to add to ChromaDB vector store")
        return
    
    try:
        print(f"Adding {len(documents)} documents to ChromaDB vector store...")
        
        # Import ChromaDB dependencies
        from langchain_graph_retriever.transformers import ShreddingTransformer
        
        # Apply shredding transformation
        shredding_transformer = ShreddingTransformer()
        transformed_documents = list(shredding_transformer.transform_documents(documents))
        print(f"Applied shredding transformation: {len(documents)} -> {len(transformed_documents)} chunks")
        
        vector_store.add_documents(transformed_documents)
        print("Successfully added documents to ChromaDB vector store")
    except Exception as e:
        print(f"Error adding documents to ChromaDB vector store: {str(e)}")
        raise

# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================

def main(force_reprocess: bool = False, clean_neo4j: bool = False, use_chromadb: bool = False):
    """
    Main processing pipeline with configurable vector store backend
    
    Args:
        force_reprocess: Force reprocessing of all documents
        clean_neo4j: Clean Neo4j database before processing
        use_chromadb: Use ChromaDB instead of in-memory vector store
    """
    try:
        # Check ChromaDB availability if requested
        if use_chromadb and not check_chromadb_availability():
            raise ImportError(
                "ChromaDB is not available. Please install it with: pip install chromadb"
            )
        
        # Initialize document tracker
        tracker = DocumentTracker()
        
        # Load and process documents efficiently - avoiding multiple calls
        print("Loading and processing documents...")
        
        # Single document loading call based on requirements
        if force_reprocess:
            print("Force reprocessing all documents...")
            all_documents = load_and_chunk_docx_files(tracker, force_reprocess=True)
            documents_for_graph = all_documents
            documents_for_vector_store = all_documents
        else:
            print("Processing only new/changed documents...")
            new_documents = load_and_chunk_docx_files(tracker, force_reprocess=False)
            documents_for_graph = new_documents
            
            if use_chromadb:
                # ChromaDB can handle incremental updates
                documents_for_vector_store = new_documents
            else:
                # In-memory needs all documents, but we can still avoid reprocessing unchanged files
                if new_documents:
                    print("New documents found, loading all documents for in-memory store...")
                    all_documents = load_and_chunk_docx_files(tracker, force_reprocess=True)
                    documents_for_vector_store = all_documents
                else:
                    print("No new documents, loading previously processed documents for in-memory store...")
                    all_documents = load_and_chunk_docx_files(tracker, force_reprocess=True)
                    documents_for_vector_store = all_documents
        
        # Initialize vector store based on selected backend
        if use_chromadb:
            print("Using ChromaDB vector store with persistence")
            if force_reprocess:
                vector_store = initialize_chromadb_vector_store(documents=documents_for_vector_store, force_recreate=True)
            elif documents_for_vector_store:
                # Load existing store and add new documents
                vector_store = initialize_chromadb_vector_store()
                add_documents_to_chromadb_store(vector_store, documents_for_vector_store)
            else:
                # Load existing ChromaDB store
                vector_store = initialize_chromadb_vector_store()
                print("Loaded existing ChromaDB vector store - no new documents to process")
        else:
            print("Using in-memory vector store")
            vector_store = initialize_in_memory_vector_store(documents=documents_for_vector_store)
        
        # Process graph documents if we have documents to process
        if documents_for_graph:
            print("Converting documents to graph format...")
            try:
                graph_transformer = LLMGraphTransformer(llm=llm)
                graph_documents = graph_transformer.convert_to_graph_documents(documents_for_graph)
                
                # Add to Neo4j (incremental unless clean_neo4j=True)
                process_graph_documents(graph_documents, incremental=not clean_neo4j)
                
            except Exception as e:
                print(f"Error processing graph documents: {str(e)}")
                # Continue with vector store functionality even if graph processing fails
        
        # Configure graph retriever
        try:
            traversal_config = Eager(
                select_k=5,
                start_k=2,
                adjacent_k=3,
                max_depth=2
            )
            
            retriever = GraphRetriever(
                store=vector_store,
                strategy=traversal_config,
            )
            
            store_type = "ChromaDB" if use_chromadb else "in-memory"
            print(f"Successfully configured GraphRetriever with {store_type} vector store")
            
        except Exception as e:
            print(f"Error configuring GraphRetriever: {str(e)}")
            # Fallback to simple vector store retriever
            retriever = vector_store.as_retriever()
            store_type = "ChromaDB" if use_chromadb else "in-memory"
            print(f"Using fallback {store_type} vector store retriever")
        
        # Set up RAG chain
        prompt = ChatPromptTemplate.from_template(
            """Answer the question based only on the context provided.
            Context: {context}
            Question: {question}"""
        )
        
        def format_docs(docs):
            return "\n\n".join(f"text: {doc.page_content} metadata: {doc.metadata}" for doc in docs)
        
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        store_type = "ChromaDB" if use_chromadb else "in-memory"
        print(f"RAG chain configured successfully with {store_type} vector store")
        
        # Save embedding cache and display statistics
        try:
            embedding_cache.save()
            cache_stats = embedding_cache.get_stats()
            
            print("\n" + "="*50)
            print("EMBEDDING CACHE STATISTICS")
            print("="*50)
            print(f"Total cached embeddings: {cache_stats['total_cached']}")
            print(f"Cache hits: {cache_stats['cache_hits']}")
            print(f"Cache misses: {cache_stats['cache_misses']}")
            print(f"New embeddings generated: {cache_stats['new_embeddings']}")
            if cache_stats['cache_hits'] + cache_stats['cache_misses'] > 0:
                print(f"Cache hit rate: {cache_stats['hit_rate']:.2%}")
            print("="*50)
            
        except Exception as e:
            print(f"Error saving embedding cache: {str(e)}")
        
        return chain, vector_store, graph
        
    except Exception as e:
        print(f"Error in main processing pipeline: {str(e)}")
        raise

# =============================================================================
# TESTING AND VALIDATION
# =============================================================================

def test_system(chain, graph):
    """Test the system with sample queries"""
    try:
        # Query Neo4j to verify graph structure
        print("\n" + "="*50)
        print("TESTING SYSTEM")
        print("="*50)
        
        nodes = graph.query("MATCH (n) RETURN count(n) as node_count")
        relationships = graph.query("MATCH ()-[r]->() RETURN count(r) as rel_count")
        
        node_count = nodes[0]["node_count"] if nodes else 0
        rel_count = relationships[0]["rel_count"] if relationships else 0
        
        print(f"Neo4j Graph Statistics:")
        print(f"  - Nodes: {node_count}")
        print(f"  - Relationships: {rel_count}")
        
        # Test query
        question = "What is this document all about, is this document inclusive, say why??"
        print(f"\nTest Question: {question}")
        
        result = chain.invoke(question)
        print(f"Answer: {result}")
        
    except Exception as e:
        print(f"Error during system testing: {str(e)}")

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Knowledge Graph RAG System with Vector Store Options")
    parser.add_argument("--force-reprocess", action="store_true", 
                       help="Force reprocessing of all documents")
    parser.add_argument("--clean-neo4j", action="store_true", 
                       help="Clean Neo4j database before processing")
    parser.add_argument("--chromadb", action="store_true", 
                       help="Use ChromaDB for persistent vector storage (default: in-memory)")
    parser.add_argument("--test-only", action="store_true", 
                       help="Only run tests, skip processing")
    parser.add_argument("--clear-cache", action="store_true", 
                       help="Clear the embedding cache before processing")
    
    args = parser.parse_args()
    
    try:
        # Ensure storage directory exists
        os.makedirs("storage", exist_ok=True)
        
        # Check ChromaDB availability if requested (before doing any work)
        if args.chromadb and not check_chromadb_availability():
            print("ChromaDB is not available. Please install it with:")
            print("pip install chromadb langchain-chroma")
            print("\nFalling back to in-memory vector store...")
            args.chromadb = False
        
        # Clear embedding cache if requested
        if args.clear_cache:
            print("Clearing embedding cache...")
            try:
                cache = EmbeddingCache()
                cache.clear_cache()
            except Exception as e:
                print(f"Error clearing cache: {str(e)}")
        
        if not args.test_only:
            chain, vector_store, graph = main(
                force_reprocess=args.force_reprocess,
                clean_neo4j=args.clean_neo4j,
                use_chromadb=args.chromadb
            )
            
            # Run tests
            test_system(chain, graph)
        else:
            # Test-only mode
            store_type = "ChromaDB" if args.chromadb else "in-memory"
            print(f"Test-only mode: Creating new {store_type} vector store with all documents...")
            
            # Connect to existing systems for testing
            tracker = DocumentTracker()
            all_documents = load_and_chunk_docx_files(tracker, force_reprocess=True)
            
            if args.chromadb:
                vector_store = initialize_chromadb_vector_store(documents=all_documents)
            else:
                vector_store = initialize_in_memory_vector_store(documents=all_documents)
            
            traversal_config = Eager(select_k=5, start_k=2, adjacent_k=3, max_depth=2)
            retriever = GraphRetriever(store=vector_store, strategy=traversal_config)
            
            prompt = ChatPromptTemplate.from_template(
                """Answer the question based only on the context provided.
                Context: {context}
                Question: {question}"""
            )
            
            def format_docs(docs):
                return "\n\n".join(f"text: {doc.page_content} metadata: {doc.metadata}" for doc in docs)
            
            chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            test_system(chain, graph)
    
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        raise