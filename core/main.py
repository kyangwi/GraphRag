import os
import json
import logging
import re
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_neo4j import Neo4jGraph
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Incooporate llmgraphtransformers
# 

# Set up logging
# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure logging to write to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'app.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Set environment variables (replace with your credentials)
os.environ["GOOGLE_API_KEY"] = " AIzaSyAxnVOuLEjDev8Zy-Oz_H5l-yXVDKq7Dm0"
NEO4J_URI = "bolt://localhost:7687"  # Replace with your Neo4j URI
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"

# Initialize models
llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash-preview-05-20")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")  # Updated model name
graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

# Load and split text file
try:
    with open("sample.txt", "r") as file:
        text = file.read()
except FileNotFoundError:
    logger.error("sample.txt not found. Please create the file.")
    exit(1)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text(text)
documents = [Document(page_content=chunk) for chunk in chunks]

# Prompt for entity and relationship extraction
extraction_prompt = PromptTemplate(
    input_variables=["text"],
    template="""Extract entities and relationships from the following text. 
    Return a JSON object with two fields: 
    - "entities": a list of [entity_name, entity_type] arrays
    - "relationships": a list of [entity1, relationship, entity2] arrays
    If no relationships are found, return an empty list for relationships.
    Ensure the response is valid JSON, wrapped in ```json``` code blocks.
    Text: {text}
    Example output:
    ```json
    {
      "entities": [["Acme Corp", "Company"], ["Jane Doe", "Person"]],
      "relationships": [["Jane Doe", "founded", "Acme Corp"]]
    }
    ```
    """
)

# Function to clean and extract JSON from Gemini response
def extract_json_from_response(response_text):
    try:
        # Extract content between ```json and ``` markers
        json_match = re.search(r"```json\n([\s\S]*?)\n```", response_text)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Fallback: treat the entire response as JSON
            json_str = response_text.strip()
        
        # Parse JSON
        data = json.loads(json_str)
        if not isinstance(data, dict) or "entities" not in data or "relationships" not in data:
            raise ValueError("Response JSON missing required fields: entities, relationships")
        return data.get("entities", []), data.get("relationships", [])
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to parse JSON response: {response_text}, Error: {e}")
        return [], []

# Extract entities and relationships
for doc in documents:
    try:
        response = llm.invoke(extraction_prompt.format(text=doc.page_content))
        entities, relationships = extract_json_from_response(response.content)

        # Add entities to Neo4j
        for entity in entities:
            if not isinstance(entity, list) or len(entity) != 2:
                logger.warning(f"Skipping invalid entity: {entity}")
                continue
            entity_name, entity_type = entity
            graph.query(
                """
                MERGE (e:Entity {name: $name, type: $type})
                """,
                {"name": str(entity_name), "type": str(entity_type)}
            )

        # Add relationships to Neo4j
        for rel in relationships:
            if not isinstance(rel, list) or len(rel) != 3:
                logger.warning(f"Skipping invalid relationship: {rel}")
                continue
            entity1, rel_type, entity2 = rel
            graph.query(
                """
                MATCH (e1:Entity {name: $entity1}), (e2:Entity {name: $entity2})
                MERGE (e1)-[r:RELATION {type: $rel_type}]->(e2)
                """,
                {"entity1": str(entity1), "entity2": str(entity2), "rel_type": str(rel_type)}
            )

    except Exception as e:
        logger.error(f"Error processing document chunk: {e}")
        continue

# Create vector store
try:
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents)
except Exception as e:
    logger.error(f"Failed to create vector store: {e}")
    exit(1)

# Define RAG prompt
rag_prompt = PromptTemplate(
    input_variables=["context", "graph_data", "question"],
    template="""Use the following context and graph data to answer the question.
    Context: {context}
    Graph Data: {graph_data}
    Question: {question}
    Answer concisely in up to three sentences. Thanks for asking!
    """
)

# GraphRAG query function
def graph_rag_query(question):
    try:
        # Vector search
        context_docs = vector_store.similarity_search(question, k=2)
        context = "\n".join([doc.page_content for doc in context_docs])
        # Graph query (example: find related entities)
        graph_result = graph.query(
            """
            MATCH (e:Entity)-[r:RELATION]->(e2:Entity)
            WHERE toLower(e.name) CONTAINS toLower($query) OR toLower(e2.name) CONTAINS toLower($query)
            RETURN e.name, r.type, e2.name LIMIT 5
            """,
            {"query": question}
        )
        graph_data = "\n".join([f"{rec['e.name']} {rec['r.type']} {rec['e2.name']}" for rec in graph_result])

        # Combine and generate answer
        chain = (
            {"context": lambda x: context, "graph_data": lambda x: graph_data, "question": RunnablePassthrough()}
            | rag_prompt
            | llm
        )
        return chain.invoke(question).content
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        return "Failed to process query due to an error."

# Test query
question = "Which college did Ananya go to?"
answer = graph_rag_query(question)
print(f"Question: {question}\nAnswer: {answer}")