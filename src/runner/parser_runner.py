from typing import List

from runner.base import RunnerBase
from parser.builder import TextSplitterBuilder
from models import ParserConfig, Document, DocumentChunk

class ParserRunner(RunnerBase):
    
    def __init__(
            self,
            config: ParserConfig,
        ):
        super().__init__()
        self.config = config
    
    async def run(
            self,
            documents: List[Document] 
        ) -> List[DocumentChunk]:

        splitter = TextSplitterBuilder(config=self.config)
        
        all_chunks: List[DocumentChunk] = [] 
        for document in documents:
            chunks = splitter.process(document=document)
            all_chunks.extend(chunks)
        

        return all_chunks