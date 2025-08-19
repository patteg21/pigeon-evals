from typing import List
from uuid import uuid4

from utils.typing import SECDocument
from utils.typing.chunks import DocumentChunk
from utils import logger
from .base import BaseProcessor


class BreaksProcessor(BaseProcessor):
    """Processor for splitting documents on page breaks."""
    
    @property
    def name(self) -> str:
        return "breaks"
    
    def process(self, document: SECDocument) -> List[DocumentChunk]:
        """Split document on [PAGE_BREAK] markers and return page chunks."""
        logger.info(f"Processing page breaks for document {document.ticker}")
        
        chunks = self._split_on_page_breaks(document)
        
        logger.info(f"Split document into {len(chunks)} page chunks")
        return chunks
    
    def _split_on_page_breaks(self, document: SECDocument) -> List[DocumentChunk]:
        """Split document text on [PAGE_BREAK] markers and create chunks."""
        body = document.text or ""
        if not body:
            return []
        
        # Split on [PAGE_BREAK] markers
        pages = body.split("[PAGE_BREAK]")
        
        # Create chunks for each page
        chunks = []
        for i, page_text in enumerate(pages):
            page_text = page_text.strip()
            if not page_text:  # Skip empty pages
                continue
                
            chunk = DocumentChunk(
                id=uuid4().hex,
                text=page_text,
                type_chunk="page",
                document=document
            )
            chunks.append(chunk)
        
        return chunks