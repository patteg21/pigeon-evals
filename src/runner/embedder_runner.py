from typing import List

from runner.base import Runner
from models import DocumentChunk

from infra.embedding import EmbedderFactory


class EmebeddingRunner(Runner):
    
    def __init__(self):
        super().__init__()
        pass
    
    async def run(
            self, 
            chunks: List[DocumentChunk] 
        ):

        for chunk in chunks:
            pass

        pass