from typing import List, Dict, Optional
import re
from uuid import uuid4

from models import DocumentChunk, Document
from models.configs.parser import ParserConfig, ProcessConfig, StepConfig

from utils import logger

class TextSplitterBuilder:

    def __init__(self, config: ParserConfig):

        self.config = config
        pass

    
    def process(self, document: Document) -> List[DocumentChunk]:
        document_chunks: List[DocumentChunk] = []

        for process in self.config.processes:

            chunks = self._process(process, document)




        pass

    def _process(
            self, 
            process: ProcessConfig,
            document: Document
        ) -> List[DocumentChunk]:
        logger.info(f"  Documnet Processor: {process.name}")

        # start with a single chunk which is just the document

        orignal_chunk = DocumentChunk(
            id=uuid4().hex,
            text=document.text,
            document=Document
        )

        chunks: List[DocumentChunk] = [orignal_chunk]
        for step in process.steps:

            chunks: List[DocumentChunk] = self._process_step(step, chunks)


    def _process_step(
            self, 
            step: StepConfig, 
            chunks: List[DocumentChunk]
        ) -> DocumentChunk:
        pass

    def _split