import os
import json
from typing import Dict, Set
from datetime import datetime
import hashlib
from core.config import TRACKING_FILE

class DocumentTracker:
    """Manages tracking of processed documents to prevent redundant processing"""
    
    def __init__(self, tracking_file: str = TRACKING_FILE):
        self.tracking_file = tracking_file
        self.tracking_data = self._load_tracking_data()
    
    def _load_tracking_data(self) -> Dict:
        """Load existing tracking data from JSON file"""
        try:
            if os.path.exists(self.tracking_file):
                with open(self.tracking_file, 'r') as f:
                    return json.load(f)
            return {"documents": {}, "last_updated": None}
        except Exception as e:
            print(f"Error loading tracking data: {str(e)}")
            return {"documents": {}, "last_updated": None}
    
    def _save_tracking_data(self):
        """Save tracking data to JSON file"""
        try:
            self.tracking_data["last_updated"] = datetime.now().isoformat()
            with open(self.tracking_file, 'w') as f:
                json.dump(self.tracking_data, f, indent=2)
        except Exception as e:
            print(f"Error saving tracking data: {str(e)}")
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file content"""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            print(f"Error calculating hash for {file_path}: {str(e)}")
            return ""
    
    def is_document_processed(self, file_path: str) -> bool:
        """Check if document has been processed and is unchanged"""
        try:
            if not os.path.exists(file_path):
                return False
            
            file_key = os.path.abspath(file_path)
            current_hash = self._calculate_file_hash(file_path)
            
            if file_key in self.tracking_data["documents"]:
                stored_hash = self.tracking_data["documents"][file_key].get("hash", "")
                return stored_hash == current_hash
            
            return False
        except Exception as e:
            print(f"Error checking if document is processed: {str(e)}")
            return False
    
    def mark_document_processed(self, file_path: str, chunk_count: int = 0):
        """Mark document as processed with metadata"""
        try:
            file_key = os.path.abspath(file_path)
            file_hash = self._calculate_file_hash(file_path)
            
            self.tracking_data["documents"][file_key] = {
                "hash": file_hash,
                "processed_at": datetime.now().isoformat(),
                "chunk_count": chunk_count,
                "file_size": os.path.getsize(file_path)
            }
            
            self._save_tracking_data()
        except Exception as e:
            print(f"Error marking document as processed: {str(e)}")
    
    def get_processed_documents(self) -> Set[str]:
        """Get set of all processed document paths"""
        return set(self.tracking_data["documents"].keys())
