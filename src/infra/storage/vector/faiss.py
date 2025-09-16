import faiss
import numpy as np
from typing import List, Optional, Any
from pathlib import Path
import pickle

from .base import VectorStorageBase, VectorStorageError
from models import DocumentChunk
from models.configs.storage import VectorConfig
from utils.logger import logger


class FAISSError(VectorStorageError):
    """FAISS-specific exception for operations"""
    pass


class FAISSVectorDB(VectorStorageBase):
    def __init__(self, config: VectorConfig):
        """Initialize FAISS vector database"""
        super().__init__(config)

        # Configuration
        index_name = self.config.path or self.config.index or self.config.index_name or "data/.faiss/index"
        dimension = self.config.dimension or 768

        self.index_path = Path(index_name)
        self.dimension = dimension
        self.metadata_path = self.index_path.with_suffix('.metadata')

        # Ensure directory exists
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize storage
        self.index = None
        self.metadata = []
        self._initialize()

    @property
    def provider_name(self) -> str:
        return "faiss"

    def _initialize(self):
        """Initialize or load FAISS index and metadata"""
        if self.index_path.exists() and self.metadata_path.exists():
            try:
                logger.info(f"Loading existing FAISS index from {self.index_path}")
                self.index = faiss.read_index(str(self.index_path))

                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)

                # Ensure metadata is a list
                if not isinstance(self.metadata, list):
                    logger.warning("Converting metadata to list format")
                    self.metadata = []

            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}. Creating new index.")
                self._create_new_index()
        else:
            self._create_new_index()

    def _create_new_index(self):
        """Create a new FAISS index"""
        logger.info(f"Creating new FAISS index with dimension {self.dimension}")
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        self._save()

    def _save(self):
        """Save index and metadata to disk"""
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)

    def upload(self, chunk: DocumentChunk) -> Any:
        """Upload a DocumentChunk with embeddings to FAISS"""
        try:
            if not chunk.embedding:
                raise FAISSError(f"Chunk {chunk.id} has no embeddings")

            # Convert embedding to numpy array
            embedding = np.array(chunk.embedding, dtype=np.float32).reshape(1, -1)

            # Check dimension compatibility
            if embedding.shape[1] != self.dimension:
                logger.info(f"Dimension mismatch. Recreating index with dimension {embedding.shape[1]}")
                self.dimension = embedding.shape[1]
                self._create_new_index()

            # Normalize for cosine similarity
            faiss.normalize_L2(embedding)

            # Add to index
            self.index.add(embedding)

            # Store metadata
            vector_id = len(self.metadata)
            metadata_entry = {
                'chunk_id': chunk.id,
                'text': chunk.text,
                'document': chunk.document.name if hasattr(chunk.document, 'name') else str(chunk.document),
                'type_chunk': getattr(chunk, 'type_chunk', None)
            }
            self.metadata.append(metadata_entry)

            # Save to disk
            self._save()
            return str(vector_id)

        except Exception as e:
            raise FAISSError(f"Failed to upload chunk {chunk.id}: {str(e)}")

    def retrieve_from_id(self, vector_id: str) -> Any:
        """Retrieve metadata by vector ID"""
        try:
            vector_id_int = int(vector_id)
            if 0 <= vector_id_int < len(self.metadata):
                return self.metadata[vector_id_int]
            return None
        except Exception as e:
            raise FAISSError(f"Failed to retrieve vector {vector_id}: {str(e)}")

    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        include_metadata: bool = True,
        filter: Optional[dict] = None,
    ) -> Any:
        """Query FAISS for similar vectors"""
        try:
            # Convert to numpy and normalize
            query_vector = np.array(vector, dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(query_vector)

            # Search
            scores, indices = self.index.search(query_vector, top_k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # No more results
                    break

                result = {
                    'id': str(idx),
                    'score': float(score),
                }

                # Add metadata if requested and available
                if include_metadata and 0 <= idx < len(self.metadata):
                    metadata = self.metadata[idx]

                    # Apply filter if provided
                    if filter:
                        should_include = True
                        for key, value in filter.items():
                            if metadata.get(key) != value:
                                should_include = False
                                break
                        if not should_include:
                            continue

                    result['metadata'] = metadata

                results.append(result)

            return results

        except Exception as e:
            raise FAISSError(f"Failed to query vectors: {str(e)}")

    def delete(self, ids: List[str]) -> Any:
        """Delete vectors by IDs (mark as deleted since FAISS doesn't support true deletion)"""
        try:
            deleted_count = 0
            for vector_id in ids:
                vector_id_int = int(vector_id)
                if 0 <= vector_id_int < len(self.metadata):
                    self.metadata[vector_id_int]['deleted'] = True
                    deleted_count += 1

            self._save()
            logger.info(f"Marked {deleted_count} vectors as deleted")
            return deleted_count

        except Exception as e:
            raise FAISSError(f"Failed to delete vectors: {str(e)}")

    def clear(self) -> Any:
        """Clear all vectors from FAISS"""
        try:
            self._create_new_index()
            logger.info("Cleared all vectors from FAISS index")
            return True

        except Exception as e:
            raise FAISSError(f"Failed to clear index: {str(e)}")