import os
import json
from typing import Dict, List
from datetime import datetime
import hashlib
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from core.config import EMBEDDING_CACHE_FILE


class EmbeddingCache:
    """
    Persistent cache for embeddings to avoid regenerating identical embeddings.
    Stores embeddings keyed by content hash for efficient retrieval.
    """
    
    def __init__(self, cache_file: str = EMBEDDING_CACHE_FILE):
        self.cache_file = cache_file
        self.cache_data = self._load_cache_data()
        self.cache_hits = 0
        self.cache_misses = 0
        self.new_embeddings = 0
    
    def _load_cache_data(self) -> Dict:
        """Load existing cache data from JSON file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    print(f"Loaded embedding cache with {len(data.get('embeddings', {}))} entries")
                    return data
            return {
                "embeddings": {},
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "last_updated": None,
                    "total_embeddings": 0,
                    "embedding_model": "models/gemini-embedding-exp-03-07"
                }
            }
        except Exception as e:
            print(f"Error loading embedding cache: {str(e)}")
            return {
                "embeddings": {},
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "last_updated": None,
                    "total_embeddings": 0,
                    "embedding_model": "models/gemini-embedding-exp-03-07"
                }
            }
    
    def _save_cache_data(self):
        """Save cache data to JSON file"""
        try:
            # Ensure storage directory exists
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            
            # Update metadata
            self.cache_data["metadata"]["last_updated"] = datetime.now().isoformat()
            self.cache_data["metadata"]["total_embeddings"] = len(self.cache_data["embeddings"])
            
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache_data, f, indent=2)
            print(f"Saved embedding cache with {len(self.cache_data['embeddings'])} entries")
        except Exception as e:
            print(f"Error saving embedding cache: {str(e)}")
    
    def _calculate_content_hash(self, content: str) -> str:
        """Calculate SHA256 hash of text content"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def get_embedding(self, content: str) -> List[float]:
        """
        Get embedding for content from cache, return None if not found
        """
        content_hash = self._calculate_content_hash(content)
        
        if content_hash in self.cache_data["embeddings"]:
            self.cache_hits += 1
            return self.cache_data["embeddings"][content_hash]["embedding"]
        
        self.cache_misses += 1
        return None
    
    def store_embedding(self, content: str, embedding: List[float]):
        """
        Store embedding in cache with content hash as key
        """
        content_hash = self._calculate_content_hash(content)
        
        self.cache_data["embeddings"][content_hash] = {
            "embedding": embedding,
            "content_length": len(content),
            "cached_at": datetime.now().isoformat(),
            "content_preview": content[:100] + "..." if len(content) > 100 else content
        }
        
        self.new_embeddings += 1
    
    def get_embeddings_batch(self, contents: List[str]) -> Dict[str, List[float]]:
        """
        Get embeddings for multiple contents from cache
        Returns dict with content_hash -> embedding for cached items
        """
        cached_embeddings = {}
        
        for content in contents:
            content_hash = self._calculate_content_hash(content)
            if content_hash in self.cache_data["embeddings"]:
                cached_embeddings[content_hash] = self.cache_data["embeddings"][content_hash]["embedding"]
                self.cache_hits += 1
            else:
                self.cache_misses += 1
        
        return cached_embeddings
    
    def store_embeddings_batch(self, content_embedding_pairs: List[tuple]):
        """
        Store multiple embeddings in batch
        content_embedding_pairs: List of (content, embedding) tuples
        """
        for content, embedding in content_embedding_pairs:
            self.store_embedding(content, embedding)
        
        self.new_embeddings += len(content_embedding_pairs)
    
    def save(self):
        """Save cache to file"""
        self._save_cache_data()
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            "total_cached": len(self.cache_data["embeddings"]),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "new_embeddings": self.new_embeddings,
            "hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        }
    
    def clear_cache(self):
        """Clear all cached embeddings"""
        self.cache_data["embeddings"] = {}
        self.cache_data["metadata"]["last_updated"] = datetime.now().isoformat()
        self._save_cache_data()
        print("Embedding cache cleared")

class CachedEmbeddings:
    """
    Wrapper around GoogleGenerativeAIEmbeddings that uses caching
    """
    
    def __init__(self, embeddings_model: GoogleGenerativeAIEmbeddings, cache: EmbeddingCache):
        self.embeddings_model = embeddings_model
        self.cache = cache
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for documents with caching
        """
        if not texts:
            return []
        
        # Check cache for existing embeddings
        cached_embeddings = self.cache.get_embeddings_batch(texts)
        
        # Separate cached and uncached texts
        uncached_texts = []
        uncached_indices = []
        text_hashes = []
        
        for i, text in enumerate(texts):
            text_hash = self.cache._calculate_content_hash(text)
            text_hashes.append(text_hash)
            
            if text_hash not in cached_embeddings:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Generate embeddings for uncached texts
        new_embeddings = []
        if uncached_texts:
            print(f"🔄 EMBEDDING GENERATION: Creating {len(uncached_texts)} new embeddings (cached: {len(texts) - len(uncached_texts)})")
            print(f"   📝 Sample uncached content: {uncached_texts[0][:100]}..." if uncached_texts else "")
            try:
                new_embeddings = self.embeddings_model.embed_documents(uncached_texts)
                
                # Store new embeddings in cache
                content_embedding_pairs = list(zip(uncached_texts, new_embeddings))
                self.cache.store_embeddings_batch(content_embedding_pairs)
                print(f"   ✅ Stored {len(new_embeddings)} new embeddings in cache")
                
            except Exception as e:
                print(f"❌ Error generating embeddings: {str(e)}")
                raise
        else:
            print(f"🎯 CACHE HIT: All {len(texts)} embeddings found in cache!")
        
        # Combine cached and new embeddings in correct order
        result_embeddings = [None] * len(texts)
        
        # Place cached embeddings
        for i, text_hash in enumerate(text_hashes):
            if text_hash in cached_embeddings:
                result_embeddings[i] = cached_embeddings[text_hash]
        
        # Place new embeddings
        new_embedding_idx = 0
        for i in uncached_indices:
            result_embeddings[i] = new_embeddings[new_embedding_idx]
            new_embedding_idx += 1
        
        return result_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query with caching
        """
        cached_embedding = self.cache.get_embedding(text)
        if cached_embedding is not None:
            print("🎯 QUERY CACHE HIT: Query embedding found in cache")
            return cached_embedding
        
        print("🔄 QUERY EMBEDDING: Generating new query embedding")
        try:
            embedding = self.embeddings_model.embed_query(text)
            self.cache.store_embedding(text, embedding)
            return embedding
        except Exception as e:
            print(f"Error generating query embedding: {str(e)}")
            raise

# =