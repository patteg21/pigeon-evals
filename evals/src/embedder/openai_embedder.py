from typing import Dict, Any, List
from utils.typing.chunks import DocumentChunk
from utils import logger
from mcp_server.clients.embedding import EmbeddingModel
from .base import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embedding provider."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        model = self.config.get("model", "text-embedding-3-small")
        self.pooling_strategy = self.config.get("pooling_strategy", "mean")
        
        logger.info(f"Initializing OpenAI embedder with model: {model}, pooling_strategy: {self.pooling_strategy}")
        # Don't pass pca_path to EmbeddingModel, base class handles reduction
        self.embedding_model = EmbeddingModel(model=model, pca_path=None)
    
    @property
    def provider_name(self) -> str:
        return "openai"
    
    async def _embed_chunk_raw(self, chunk: DocumentChunk) -> List[float]:
        """Get raw OpenAI embeddings for a single chunk."""
        return await self.embedding_model.create_embedding(chunk.text, strategy=self.pooling_strategy)
    
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
                token_count = await self.embedding_model.count_tokens(chunk.text)
                if token_count <= self.embedding_model.max_tokens:
                    batch_texts.append(chunk.text)
                    batch_chunk_indices.append(i + j)
                else:
                    oversized_chunks.append(chunk)
                    oversized_indices.append(i + j)
            
            # Process normal chunks in batch
            if batch_texts:
                batch_embeddings = await self.embedding_model.create_embeddings_batch(batch_texts)
                # Insert embeddings at correct positions
                for embedding, idx in zip(batch_embeddings, batch_chunk_indices):
                    while len(embeddings) <= idx:
                        embeddings.append(None)
                    embeddings[idx] = embedding
        
        # Process oversized chunks individually with configured pooling strategy
        for chunk, idx in zip(oversized_chunks, oversized_indices):
            embedding = await self.embedding_model.create_embedding(chunk.text, strategy=self.pooling_strategy)
            while len(embeddings) <= idx:
                embeddings.append(None)
            embeddings[idx] = embedding
        
        return embeddings