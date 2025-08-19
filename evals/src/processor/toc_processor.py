import re
from typing import List
from uuid import uuid4

from utils.typing import SECDocument, SECTable
from utils.typing.chunks import DocumentChunk
from utils import logger
from .base import BaseProcessor


class TOCProcessor(BaseProcessor):
    """Processor for extracting and parsing Table of Contents."""
    
    @property
    def name(self) -> str:
        return "toc"
    
    def process(self, document: SECDocument) -> List[DocumentChunk]:
        """Extract TOC and return TOC chunks."""
        logger.info(f"Processing TOC for document {document.ticker}")
        
        # Extract tables first to find TOC
        tables = self._extract_tables(document)
        toc = self._find_toc_in_tables(tables)
        
        chunks = []
        if toc:
            # Create chunk for TOC
            chunk = DocumentChunk(
                id=uuid4().hex,
                text=toc.text,
                type_chunk="toc",
                document=document
            )
            chunks.append(chunk)
            logger.info("Found and extracted TOC chunk")
        else:
            logger.warning(f"No TOC found for document {document.ticker}")
            
        return chunks
    
    def _extract_tables(self, document: SECDocument) -> List[SECTable]:
        """Extract all tables from document text."""
        TABLE_RE = re.compile(r"\[TABLE_START\](.*?)\[[ ]*TABLE_END\]", re.DOTALL | re.IGNORECASE)
        body = document.text or ""
        tables: List[SECTable] = []

        for m in TABLE_RE.finditer(body):
            start_pos = m.start()
            text = (m.group(1) or "").strip()
            page = self._compute_page_number(body, start_pos)

            tables.append(SECTable(
                id=uuid4().hex,
                page_number=page,
                text=text,
            ))

        return tables
    
    def _find_toc_in_tables(self, tables: List[SECTable]) -> SECTable:
        """Find the table of contents in the list of tables."""
        for table in tables:
            if "Item 1." in table.text:
                return table
        return None