"""
Multi-tenant vector database with session isolation
Each session gets its own HNSW index directory
"""
import json
import numpy as np
import hnswlib
from typing import List, Dict, Optional
from pathlib import Path
import pickle
import hashlib
from datetime import datetime


class SessionVectorStore:
    """
    Session-isolated vector store.
    Each user/session has completely separate index files.
    """
    
    def __init__(self, base_path: str = "./data"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.active_sessions = {}  # Cache for loaded sessions
    
    def _get_session_path(self, session_id: str) -> Path:
        """Get directory path for a specific session"""
        # Sanitize session ID
        safe_id = hashlib.md5(session_id.encode()).hexdigest()[:16]
        session_dir = self.base_path / f"session_{safe_id}"
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir
    
    def create_session(self, session_id: str) -> Dict:
        """Create a new empty session"""
        session_path = self._get_session_path(session_id)
        
        # Initialize empty metadata
        metadata_path = session_path / "metadata.pkl"
        if not metadata_path.exists():
            with open(metadata_path, 'wb') as f:
                pickle.dump([], f)
        
        return {
            "session_id": session_id,
            "path": str(session_path),
            "created_at": datetime.now().isoformat(),
            "document_count": 0
        }
    
    def add_documents(self, session_id: str, chunks: List[str], 
                      embeddings: List[List[float]], 
                      metadatas: Optional[List[Dict]] = None):
        """Add documents to a specific session's vector store"""
        session_path = self._get_session_path(session_id)
        vectors = np.array(embeddings).astype('float32')
        
        # Load or create index
        index_file = session_path / "index.bin"
        metadata_file = session_path / "metadata.pkl"
        
        # Load existing metadata
        if metadata_file.exists():
            with open(metadata_file, 'rb') as f:
                existing_metadata = pickle.load(f)
        else:
            existing_metadata = []
        
        # Initialize or load index
        if index_file.exists():
            dimension = np.load(session_path / "dimension.npy").item()
            index = hnswlib.Index(space='cosine', dim=dimension)
            index.load_index(str(index_file))
            current_count = len(existing_metadata)
        else:
            dimension = vectors.shape[1]
            index = hnswlib.Index(space='cosine', dim=dimension)
            index.init_index(max_elements=10000, ef_construction=200, M=16)
            current_count = 0
            np.save(session_path / "dimension.npy", dimension)
        
        # Add new vectors
        labels = np.arange(current_count, current_count + len(chunks))
        index.add_items(vectors, labels)
        
        # Update metadata
        for i, (chunk, label) in enumerate(zip(chunks, labels)):
            meta = {
                'id': int(label),
                'text': chunk,
                'metadata': metadatas[i] if metadatas else {},
                'added_at': datetime.now().isoformat()
            }
            existing_metadata.append(meta)
        
        # Save everything
        index.save_index(str(index_file))
        with open(metadata_file, 'wb') as f:
            pickle.dump(existing_metadata, f)
        
        # Update cache
        self.active_sessions[session_id] = index
        
        return len(chunks)
    
    def similarity_search(self, session_id: str, query_embedding: List[float], 
                          k: int = 3) -> List[Dict]:
        """Search for similar documents in a session"""
        session_path = self._get_session_path(session_id)
        index_file = session_path / "index.bin"
        metadata_file = session_path / "metadata.pkl"
        
        # Check if session exists
        if not index_file.exists() or not metadata_file.exists():
            return []
        
        # Load metadata
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        if len(metadata) == 0:
            return []
        
        # Load or get index from cache
        if session_id in self.active_sessions:
            index = self.active_sessions[session_id]
        else:
            dimension = np.load(session_path / "dimension.npy").item()
            index = hnswlib.Index(space='cosine', dim=dimension)
            index.load_index(str(index_file))
            self.active_sessions[session_id] = index
        
        # Search
        query_vector = np.array(query_embedding).astype('float32').reshape(1, -1)
        labels, distances = index.knn_query(query_vector, k=min(k, len(metadata)))
        
        # Format results
        results = []
        for label, distance in zip(labels[0], distances[0]):
            doc = metadata[int(label)].copy()
            doc['score'] = float(1 - distance)
            results.append(doc)
        
        return results
    
    def get_session_info(self, session_id: str) -> Dict:
        """Get information about a session"""
        session_path = self._get_session_path(session_id)
        metadata_file = session_path / "metadata.pkl"
        
        if metadata_file.exists():
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            
            # Extract unique sources
            sources = set()
            for doc in metadata:
                if 'metadata' in doc and 'source' in doc['metadata']:
                    sources.add(doc['metadata']['source'])
            
            return {
                "session_id": session_id,
                "exists": True,
                "document_count": len(metadata),
                "sources": list(sources),
                "vector_dimension": np.load(session_path / "dimension.npy").item() 
                                    if (session_path / "dimension.npy").exists() else None
            }
        else:
            return {
                "session_id": session_id,
                "exists": False,
                "document_count": 0
            }
    
    def delete_session(self, session_id: str):
        """Delete a session and all its data"""
        import shutil
        session_path = self._get_session_path(session_id)
        if session_path.exists():
            shutil.rmtree(session_path)
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]