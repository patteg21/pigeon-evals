from abc import ABC, abstractmethod
from typing import List, Dict, Any
from utils.typing.chunks import DocumentChunk
from utils import logger


class BaseEmbedder(ABC):
    """Base class for all embedding providers."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.reducer = None
        self._setup_dimensional_reduction()
    
    def _setup_dimensional_reduction(self):
        """Setup dimensional reduction if configured."""
        dimension_reduction = self.config.get("dimension_reduction")
        if dimension_reduction:
            reduction_type = dimension_reduction.get("type")
            if reduction_type == "PCA":
                from .dimensional_reduction import PCAReducer
                self.reducer = PCAReducer(dimension_reduction)
                logger.info(f"Configured PCA reduction to {dimension_reduction.get('dims', 512)} dimensions")
            elif reduction_type in ["UMAP", "T-SNE"]:
                raise NotImplementedError(f"{reduction_type} dimensional reduction not implemented yet")
            else:
                logger.warning(f"Unknown dimensional reduction type: {reduction_type}")
    
    @abstractmethod
    async def _embed_chunk_raw(self, chunk: DocumentChunk) -> List[float]:
        """Get raw embeddings for a single chunk (to be implemented by subclasses)."""
        raise NotImplementedError
    
    async def _embed_chunks_raw(self, chunks: List[DocumentChunk]) -> List[List[float]]:
        """Get raw embeddings for multiple chunks (can be overridden for batch efficiency)."""
        embeddings = []
        for chunk in chunks:
            embedding = await self._embed_chunk_raw(chunk)
            embeddings.append(embedding)
        return embeddings
    
    async def embed_chunk(self, chunk: DocumentChunk) -> DocumentChunk:
        """Embed a single chunk and return it with embeddings (with auto dimensional reduction)."""
        # For single chunks, use the batch method to ensure consistency
        return (await self.embed_chunks([chunk]))[0]
    
    async def embed_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Embed multiple chunks with automatic dimensional reduction."""
        # Get raw embeddings
        raw_embeddings = await self._embed_chunks_raw(chunks)
        
        # Apply dimensional reduction if configured
        if self.reducer:
            logger.info(f"Applying {self.reducer.name} dimensional reduction")
            # Train and apply reduction
            reduced_embeddings = self.reducer.fit_transform(raw_embeddings)
            # Save the trained model
            self.reducer.save()
        else:
            reduced_embeddings = raw_embeddings
        
        # Create embedded chunks
        embedded_chunks = []
        for chunk, embedding in zip(chunks, reduced_embeddings):
            embedded_chunk = DocumentChunk(
                id=chunk.id,
                text=chunk.text,
                type_chunk=chunk.type_chunk,
                document=chunk.document,
                embeddding=embedding  # Note: keeping original typo for compatibility
            )
            embedded_chunks.append(embedded_chunk)
        
        return embedded_chunks
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass