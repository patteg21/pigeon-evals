from typing import List

from runner.base import Runner
from models import DocumentChunk
from utils.dry_run import dry_response, mock_embedding_chunks

from infra.embedding import EmbedderFactory, BaseEmbedder


class EmbeddingRunner(Runner):

    def __init__(self):
        super().__init__()
        self.embedder: BaseEmbedder = EmbedderFactory.create_from_config()
    
    @dry_response(mock_factory=mock_embedding_chunks(dimensions=384))
    async def run(
            self, 
            chunks: List[DocumentChunk] 
        ) -> List[DocumentChunk]:
        
        return await self.embedder.embed_chunks(chunks)