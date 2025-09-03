import re
from typing import List
from uuid import uuid4

from evals.src.utils.types import SECDocument
from evals.src.utils.types.chunks import DocumentChunk
from evals.src.utils import logger
from .base import BaseProcessor


class TablesProcessor(BaseProcessor):
    """Processor for extracting tables from documents."""
    
    @property
    def name(self) -> str:
        return "tables"
    
    def process(self, document: SECDocument) -> List[DocumentChunk]:
        """Extract all tables from document text and return table chunks."""
        logger.info(f"Processing tables for document {document.ticker}")
        
        chunks = self._extract_tables(document)
        
        logger.info(f"Extracted {len(chunks)} table chunks")
        return chunks
    
    def _extract_tables(self, document: SECDocument) -> List[DocumentChunk]:
        """Extract all tables from document text and create chunks."""
        TABLE_RE = re.compile(r"\[TABLE_START\](.*?)\[[ ]*TABLE_END\]", re.DOTALL | re.IGNORECASE)
        body = document.text or ""
        chunks: List[DocumentChunk] = []

        for m in TABLE_RE.finditer(body):
            text = (m.group(1) or "").strip()
            if not text:  # Skip empty tables
                continue

            chunk = DocumentChunk(
                id=uuid4().hex,
                text=text,
                type_chunk="table",
                document=document
            )
            chunks.append(chunk)

        return chunks