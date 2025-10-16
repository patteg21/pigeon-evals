from typing import List

from tqdm.asyncio import tqdm

from runner.base import RunnerBase
from models import DocumentChunk


from infra.embedding import EmbedderFactory, BaseEmbedder


class EmbeddingRunner(RunnerBase):

    def __init__(self):
        super().__init__()
        self.embedder: BaseEmbedder = EmbedderFactory.create_from_config()

    async def run(
            self,
            chunks: List[DocumentChunk]
        ) -> List[DocumentChunk]:


        return await self.embedder.embed_chunks(chunks)