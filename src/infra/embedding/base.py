from abc import ABC, abstractmethod
from typing import List, Iterable
import asyncio
import time
import numpy as np
import diskcache as dc
from models import DocumentChunk, Pooling
from models.configs import EmbeddingConfig
from utils import logger

cache = dc.Cache("data/.cache")


class BaseEmbedder(ABC):
    """Base class for all embedding providers."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.reducer = None
        self._setup_dimensional_reduction()
    
    def _setup_dimensional_reduction(self):
        """Setup dimensional reduction if configured."""
        dimension_reduction = self.config.dimension_reduction
        if dimension_reduction:
            reduction_type = dimension_reduction.type
            if reduction_type == "PCA":
                from .dimensional_reduction import PCAReducer
                self.reducer = PCAReducer(dimension_reduction)
                logger.info(f"Configured PCA reduction to {dimension_reduction.dims} dimensions")
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
        # Get raw embeddings from provider (OpenAI/HuggingFace)
        raw_embeddings = await self._embed_chunks_raw(chunks)
        
        # Apply dimensional reduction if configured
        if self.reducer:
            logger.info(f"Applying {self.reducer.name} dimensional reduction")
            # Train PCA on ALL raw embeddings, then transform them
            reduced_embeddings = self.reducer.fit_transform(raw_embeddings)
            # Save trained PCA model to disk for later use
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
    
    @staticmethod
    def _l2n(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """L2 normalization to put between units between 0-1 """
        n = np.linalg.norm(v) + eps
        return v / n
    
    @staticmethod
    def _pool(
        vecs: np.ndarray, 
        strategy: Pooling, 
        weights: Iterable[float] | None = None
    ) -> np.ndarray:
        if strategy == "mean":
            return vecs.mean(axis=0)
        if strategy == "max":
            return vecs.max(axis=0)
        if strategy == "weighted":
            w = np.asarray(list(weights) if weights is not None else [1.0]*len(vecs), dtype=np.float32)
            w = w / (w.sum() if w.sum() > 0 else 1.0)
            return (vecs * w[:, None]).sum(axis=0)
        if strategy == "smooth_decay":
            # Exponential decay by chunk index (earlier chunks weigh slightly more)
            idx = np.arange(len(vecs), dtype=np.float32)
            # decay factor ~0.9 per step; adjust if needed
            w = np.power(0.9, idx)
            w = w / w.sum()
            return (vecs * w[:, None]).sum(axis=0)
        raise ValueError(f"Unknown pooling strategy: {strategy}")
    
    async def _retry_with_backoff(self, func, *args, max_retries=5, **kwargs):
        """Execute a function with exponential backoff retry logic"""
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                
                # Exponential backoff with jitter
                delay = base_delay * (2 ** attempt) + (time.time() % 1)
                await asyncio.sleep(delay)