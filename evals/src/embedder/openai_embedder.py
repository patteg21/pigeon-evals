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
        
        logger.info(f"Initializing OpenAI embedder with model: {model}")
        # Don't pass pca_path to EmbeddingModel, base class handles reduction
        self.embedding_model = EmbeddingModel(model=model, pca_path=None)
    
    @property
    def provider_name(self) -> str:
        return "openai"
    
    async def _embed_chunk_raw(self, chunk: DocumentChunk) -> List[float]:
        """Get raw OpenAI embeddings for a single chunk."""
        return await self.embedding_model.create_embedding(chunk.text)
    
    async def _embed_chunks_raw(self, chunks: List[DocumentChunk]) -> List[List[float]]:
        """Get raw OpenAI embeddings for multiple chunks (batch optimized)."""
        texts = [chunk.text for chunk in chunks]
        return await self.embedding_model.create_embeddings_batch(texts)