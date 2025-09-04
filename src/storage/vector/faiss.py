import faiss
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import pickle

from .base import VectorStorageBase, VectorStorageError
from utils.types import DocumentChunk
from utils.logger import logger


class FAISSError(VectorStorageError):
    """FAISS-specific exception for operations"""
    pass


class FAISSVectorDB(VectorStorageBase):
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize FAISS vector database"""
        super().__init__(config)
        
        index_path = self.config.get("index_path", "data/.faiss/index")
        dimension = self.config.get("dimension", 768)
        
        self.index_path = Path(index_path)
        self.dimension = dimension
        self.metadata_path = self.index_path.with_suffix('.metadata')
        
        # Ensure directory exists
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize or load index
        self.index = None
        self.metadata = {}  # Store metadata separately
        self._initialize_index()
    
    @property
    def provider_name(self) -> str:
        return "faiss"
    
    def _initialize_index(self):
        """Initialize or load FAISS index"""
        if self.index_path.exists():
            logger.info(f"Loading existing FAISS index from {self.index_path}")
            self.index = faiss.read_index(str(self.index_path))
            
            # Load metadata if exists
            if self.metadata_path.exists():
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
        else:
            logger.info(f"Creating new FAISS index with dimension {self.dimension}")
            # Use IndexFlatIP (Inner Product) for similarity search
            self.index = faiss.IndexFlatIP(self.dimension)
            self._save_index()
    
    def _save_index(self):
        """Save index and metadata to disk"""
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
    
    def upload(self, chunk: DocumentChunk) -> Any:
        """Upload a DocumentChunk with embeddings to FAISS"""
        try:
            if not chunk.embeddding:  # Note: keeping original typo for compatibility
                raise FAISSError(f"Chunk {chunk.id} has no embeddings")
            
            # Convert embedding to numpy array and normalize
            embedding = np.array(chunk.embeddding, dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(embedding)  # Normalize for cosine similarity
            
            # Add to index
            self.index.add(embedding)
            
            # Store metadata
            vector_id = len(self.metadata)  # Use current count as ID
            self.metadata[vector_id] = {
                'chunk_id': chunk.id,
                'text': chunk.text,
                'document': chunk.document,
                'type_chunk': chunk.type_chunk
            }
            
            self._save_index()
            logger.debug(f"Uploaded chunk {chunk.id} to FAISS index")
            return vector_id
            
        except Exception as e:
            raise FAISSError(f"Failed to upload chunk {chunk.id}: {str(e)}")
    
    def retrieve_from_id(self, vector_id: str) -> Any:
        """Retrieve metadata by vector ID"""
        try:
            vector_id_int = int(vector_id)
            if vector_id_int in self.metadata:
                return self.metadata[vector_id_int]
            return None
        except Exception as e:
            raise FAISSError(f"Failed to retrieve vector {vector_id}: {str(e)}")
    
    def query(
        self, 
        vector: List[float], 
        top_k: int = 10, 
        include_metadata: bool = True, 
        filter: Optional[Dict[str, Any]] = None, 
    ) -> Any:
        """Query FAISS for similar vectors"""
        try:
            # Convert to numpy and normalize
            query_vector = np.array(vector, dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(query_vector)
            
            # Search
            scores, indices = self.index.search(query_vector, top_k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:  # No more results
                    break
                
                result = {
                    'id': str(idx),
                    'score': float(score),
                }
                
                if include_metadata and idx in self.metadata:
                    result['metadata'] = self.metadata[idx]
                
                # Apply filter if provided
                if filter:
                    metadata = self.metadata.get(idx, {})
                    should_include = True
                    for key, value in filter.items():
                        if metadata.get(key) != value:
                            should_include = False
                            break
                    if not should_include:
                        continue
                
                results.append(result)
            
            return results
            
        except Exception as e:
            raise FAISSError(f"Failed to query vectors: {str(e)}")
    
    def delete(self, ids: List[str]) -> Any:
        """Delete vectors by IDs (FAISS doesn't support deletion, so we mark as deleted)"""
        try:
            deleted_count = 0
            for vector_id in ids:
                vector_id_int = int(vector_id)
                if vector_id_int in self.metadata:
                    # Mark as deleted instead of actually deleting
                    self.metadata[vector_id_int]['deleted'] = True
                    deleted_count += 1
            
            self._save_index()
            logger.info(f"Marked {deleted_count} vectors as deleted")
            return deleted_count
            
        except Exception as e:
            raise FAISSError(f"Failed to delete vectors: {str(e)}")
    
    def clear(self) -> Any:
        """Clear all vectors from FAISS"""
        try:
            # Create new empty index
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata = {}
            self._save_index()
            logger.info("Cleared all vectors from FAISS index")
            return True
            
        except Exception as e:
            raise FAISSError(f"Failed to clear index: {str(e)}")