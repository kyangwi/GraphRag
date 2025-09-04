import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
NEO4J_URI = os.environ.get("NEO4J_URI")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
TRACKING_FILE = "../core/storage/document_tracking.json"
EMBEDDING_CACHE_FILE = "./core/storage/embedding_cache.json"
FILES_DIRECTORY = "./core/files"
CHROMADB_PERSIST_DIRECTORY = "./core/storage/chromadb"
CHROMADB_COLLECTION_NAME = "documents"