import os
from graph_retriever.strategies import Eager
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_graph_retriever import GraphRetriever
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph

# Set up environment variables for Neo4j and Gemini
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = "AIzaSyC0iIUYf3uOw3rGvSZ-kY6g3FKxfjp0zCY"

if not os.environ.get("NEO4J_URI"):
    os.environ["NEO4J_URI"] = "bolt://localhost:7687"
if not os.environ.get("NEO4J_USERNAME"):
    os.environ["NEO4J_USERNAME"] = "neo4j"
if not os.environ.get("NEO4J_PASSWORD"):
    os.environ["NEO4J_PASSWORD"] = "password"

# Initialize embeddings and LLM
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash-preview-05-20")

# Initialize Neo4j graph
graph = Neo4jGraph()

# Sample unstructured text

raw_text = [
    Document(
        page_content="Capybaras are large rodents native to South America, often found in wetlands. They share habitats with herons and crocodiles.",
        metadata={"source": "animal_facts_1", "animal": "capybara", "habitat": "wetlands", "related_animals": ["heron", "crocodile"]}
    ),
    Document(
        page_content="Lions are large cats native to Africa, living in savannas and grasslands.",
        metadata={"source": "animal_facts_2", "animal": "lion", "habitat": "savanna", "related_animals": []}
    ),
]

# Initialize LLMGraphTransformer
graph_transformer = LLMGraphTransformer(llm=llm)

# Convert unstructured text to graph documents
graph_documents = graph_transformer.convert_to_graph_documents(raw_text)

# Add nodes and relationships to Neo4j
for graph_doc in graph_documents:
    for node in graph_doc.nodes:
        graph.query(
            f"MERGE (n:{node.type} {{id: $id}})",
            {"id": node.id}
        )
    for rel in graph_doc.relationships:
        graph.query(
            f"MATCH (s {{id: $source}}), (t {{id: $target}}) "
            f"MERGE (s)-[:{rel.type}]->(t)",
            {"source": rel.source.id, "target": rel.target.id}
        )

# Create in-memory vector store with original documents
vector_store = InMemoryVectorStore.from_documents(
    documents=raw_text,
    embedding=embeddings,
)

# Configure graph retriever with metadata-based traversal
traversal_config = Eager(
    select_k=5,
    start_k=2,
    adjacent_k=3,
    max_depth=2
)

retriever = GraphRetriever(
    store=vector_store,
    # embeddings=embeddings,
    strategy=traversal_config,
)

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

# Test the chain and inspect the graph
if __name__ == "__main__":
    # Query Neo4j to verify graph structure
    nodes = graph.query("MATCH (n) RETURN n")
    relationships = graph.query("MATCH ()-[r]->() RETURN r")
    print("Neo4j Nodes:", [record["n"] for record in nodes])
    print("Neo4j Relationships:", [record["r"] for record in relationships])
    
    question = "What animals could be found near a capybara?"
    result = chain.invoke(question)
    print(f"\nQuestion: {question}")
    print(f"Answer: {result}")