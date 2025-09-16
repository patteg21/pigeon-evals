from typing import List

from runner.base import Runner
from models import DocumentChunk
from utils.dry_run import dry_response, mock_embedding_chunks
from utils.config_manager import ConfigManager

from infra.embedding import EmbedderFactory, BaseEmbedder


class EmbeddingRunner(Runner):

    def __init__(self):
        super().__init__()
        self.embedder: BaseEmbedder = EmbedderFactory.create_from_config()

        # Determine dimensions for mock embeddings based on config
        config = ConfigManager().config
        if (config.embedding and
            config.embedding.dimension_reduction and
            config.embedding.dimension_reduction.dims):
            mock_dims = config.embedding.dimension_reduction.dims
        else:
            mock_dims = 384  # Default embedding dimensions

    @dry_response(mock_factory=lambda self, chunks: self._mock_with_config_dims(chunks))
    async def run(
            self,
            chunks: List[DocumentChunk]
        ) -> List[DocumentChunk]:

        return await self.embedder.embed_chunks(chunks)

    def _mock_with_config_dims(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Generate mock embeddings with dimensions from config."""
        config = ConfigManager().config
        if (config.embedding and
            config.embedding.dimension_reduction and
            config.embedding.dimension_reduction.dims):
            dimensions = config.embedding.dimension_reduction.dims
        else:
            dimensions = 384  # Default embedding dimensions

        return mock_embedding_chunks(dimensions)(self, chunks)