from typing import List

from models import DocumentChunk

from infra.storage.text import TextStorageFactory
from infra.storage.vector import VectorStorageFactory
from runner.base import Runner


class StorageRunner(Runner):
    
    def __init__(self):
        super().__init__()
        pass
    
    async def run(
            self,
            documents: List[DocumentChunk] 
        ) -> None:


        pass