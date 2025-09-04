from typing import Dict, Any, List
import os
import asyncio
import time

import tiktoken
import numpy as np
import diskcache as dc
from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError

from utils.types import DocumentChunk
from utils.types import Pooling
from utils import logger

from .dimensional_reduction import PCAReducer

from .base import BaseEmbedder

load_dotenv()

cache = dc.Cache("data/.cache")


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embedding provider."""
    
    TOKEN_LIMITS: Dict[str, int] = {
        "text-embedding-3-small": 8191,
        # add models as needed
    }
    
    def __init__(self, config: Dict[str, Any] | None = None, pca_path: str | None = None):
        super().__init__(config)
        self.model = self.config.get("model", "text-embedding-3-small")
        self.pooling_strategy = self.config.get("pooling_strategy", "mean")
        
        if self.model not in self.TOKEN_LIMITS:
            raise ValueError(f"Unsupported model '{self.model}'. Provide max_tokens explicitly or add to TOKEN_LIMITS.")

        self.max_tokens: int = self.TOKEN_LIMITS[self.model]
        self.client: AsyncOpenAI = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.encoding = tiktoken.encoding_for_model(self.model)
        

        self.pca_reducer:  PCAReducer | None = None
        if pca_path:
            logger.warning("PCA REDUCER HAS BEEN ACTIVATED")
            try:
                self.pca_reducer = PCAReducer({"path": pca_path}).load()
            except Exception:
                self.pca_reducer = None

        logger.info(f"Initializing OpenAI embedder with model: {self.model}, pooling_strategy: {self.pooling_strategy}")
    

    @property
    def provider_name(self) -> str:
        return "openai"
    

    async def _embeddings(self, text: str) -> List[float]:
            """Create embedding for a single text with rate limit handling"""
            max_retries = 5
            base_delay = 1.0
            
            for attempt in range(max_retries):
                try:
                    response = await self.client.embeddings.create(
                        input=text,
                        model=self.model
                    )
                    return response.data[0].embedding
                except RateLimitError as e:
                    if attempt == max_retries - 1:
                        raise e
                    
                    # Exponential backoff with jitter
                    delay = base_delay * (2 ** attempt) + (time.time() % 1)
                    await asyncio.sleep(delay)
    
    async def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for multiple texts with rate limit handling"""
        async def _embed_batch():
            response = await self.client.embeddings.create(
                input=texts,
                model=self.model
            )
            return [data.embedding for data in response.data]
        
        return await self._retry_with_backoff(_embed_batch)
    
    async def count_tokens(self, text: str) -> int:
        """Count tokens in a text for the current model"""
        return len(self.encoding.encode(text))
    
    async def _is_too_large(self, tokens):
        return tokens > self.max_tokens

    def _chunk_by_tokens(self, text: str, max_tokens: int, overlap: int) -> List[str]:
        """Chunking based on the overlap and max_tokens"""
        ids = self.encoding.encode(text)
        if len(ids) <= max_tokens:
            return [text]
        chunks, start = [], 0
        while start < len(ids):
            end = min(start + max_tokens, len(ids))
            chunks.append(self.encoding.decode(ids[start:end]))
            if end == len(ids): 
                break
            start = max(0, end - overlap)
        return chunks

    async def create_embedding(
        self,
        text: str,
        strategy: Pooling = "mean",
        *,
        chunk_max_tokens: int = 2048,
        overlap_tokens: int = 128,
        batch_size: int = 64,
        normalize_chunks: bool = True,
        normalize_output: bool = True,
        weighted_by_length: bool = True,
    ) -> List[float]:
        """
        Return a single pooled embedding vector for the given text.
        """
        if text in cache:
            return cache[text]
        
        total = await self.count_tokens(text)
        if not await self._is_too_large(total):
            vec = np.asarray(await self._embeddings(text), dtype=np.float32)
            return self._l2n(vec).tolist() if normalize_output else vec.tolist()

        if chunk_max_tokens > self.max_tokens:
            raise ValueError(
                f"`chunk_max_tokens` ({chunk_max_tokens}) cannot exceed model limit ({self.max_tokens})."
            )
        
        chunks = self._chunk_by_tokens(text, chunk_max_tokens, overlap_tokens)

        embs: List[List[float]] = []
        for i in range(0, len(chunks), batch_size):
            embs.extend(await self.create_embeddings_batch(chunks[i:i+batch_size]))

        vecs = np.asarray(embs, dtype=np.float32)
        if normalize_chunks:
            vecs = np.vstack([self._l2n(v) for v in vecs])

        weights = [await self.count_tokens(c) for c in chunks] if (strategy == "weighted" and weighted_by_length) else None
        pooled = self._pool(vecs, strategy=strategy, weights=weights)
        if normalize_output:
            pooled = self._l2n(pooled)
        out = pooled.astype(np.float32).tolist()

        cache[text] = out
        return out
    
    async def _embed_chunk_raw(self, chunk: DocumentChunk) -> List[float]:
        """Get raw OpenAI embeddings for a single chunk."""
        return await self.create_embedding(chunk.text, strategy=self.pooling_strategy)
    
    async def _embed_chunks_raw(self, chunks: List[DocumentChunk], batch_size=32) -> List[List[float]]:
        """Get raw OpenAI embeddings for multiple chunks (batch optimized)."""
        embeddings = []
        oversized_chunks = []
        oversized_indices = []
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_texts = []
            batch_chunk_indices = []
            
            # Separate normal and oversized chunks
            for j, chunk in enumerate(batch_chunks):
                token_count = await self.count_tokens(chunk.text)
                if token_count <= self.max_tokens:
                    batch_texts.append(chunk.text)
                    batch_chunk_indices.append(i + j)
                else:
                    oversized_chunks.append(chunk)
                    oversized_indices.append(i + j)
            
            # Process normal chunks in batch
            if batch_texts:
                batch_embeddings = await self.create_embeddings_batch(batch_texts)
                # Insert embeddings at correct positions
                for embedding, idx in zip(batch_embeddings, batch_chunk_indices):
                    while len(embeddings) <= idx:
                        embeddings.append(None)
                    embeddings[idx] = embedding
        
        # Process oversized chunks individually with configured pooling strategy
        for chunk, idx in zip(oversized_chunks, oversized_indices):
            embedding = await self.create_embedding(chunk.text, strategy=self.pooling_strategy)
            while len(embeddings) <= idx:
                embeddings.append(None)
            embeddings[idx] = embedding
        
        return embeddings
    

    """
===============================================
Atypical Implementation for MCP Server
===============================================
    """

    def _apply_pca_reduction(self, embedding: List[float]) -> List[float]:
        """Apply PCA reduction if available"""
        if self.pca_reducer and self.pca_reducer.model is not None:
            return self.pca_reducer.transform_one(embedding)
        # identity + L2 normalize to keep cosine geometry stable if no PCA
        v = np.asarray(embedding, dtype=np.float32)
        v = v / (np.linalg.norm(v) + 1e-9)
        return v.tolist()

    async def create_pinecone_embeddings(
        self,
        text: str,
        strategy: Pooling = "mean",
        *,
        chunk_max_tokens: int = 2048,
        overlap_tokens: int = 128,
        batch_size: int = 64,
        normalize_chunks: bool = True,
        normalize_output: bool = True,
        weighted_by_length: bool = True,
    ) -> List[float]:
        """
        Create embedding and apply PCA reduction if configured.
        """
        embedding = await self.create_embedding(
            text=text,
            strategy=strategy,
            chunk_max_tokens=chunk_max_tokens,
            overlap_tokens=overlap_tokens,
            batch_size=batch_size,
            normalize_chunks=normalize_chunks,
            normalize_output=normalize_output,
            weighted_by_length=weighted_by_length,
        )
        return self._apply_pca_reduction(embedding)