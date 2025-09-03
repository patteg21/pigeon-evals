import re
from typing import List, Dict
from uuid import uuid4

from evals.src.utils.types import SECDocument
from evals.src.utils.types.chunks import DocumentChunk
from evals.src.utils import logger
from .base import BaseProcessor




class TOCProcessor(BaseProcessor):
    """Processor for extracting and parsing Table of Contents."""
    
    @property
    def name(self) -> str:
        return "toc"
    
    def process(self, document: SECDocument) -> List[DocumentChunk]:
        """Extract TOC and return TOC chunks."""
        logger.info(f"Processing TOC for document {document.ticker}")

        # Split document by pages and get report text (all pages except first)
        pages = document.text.split("[PAGE BREAK]")
        toc_page = pages[1]
        report_text = "[PAGE BREAK]".join(pages[1:]) if len(pages) > 1 else ""
        
        toc_structure = self._parse_out_toc(toc_page)
        
        chunks = []
        
        # Breaks up the document based on the TOC
        # Extract text sections for each part and item
        for part_name, items in toc_structure.items():
            # Find part text in report
            part_text = self._extract_section_text(report_text, part_name, items[0] if items else None)
            
            if part_text:
                part_chunk = DocumentChunk(
                    id=uuid4().hex,
                    text=part_text,
                    type_chunk="part",
                    document=document
                )
                chunks.append(part_chunk)
            
            # Extract text for each item
            for i, item in enumerate(items):
                next_item = items[i + 1] if i + 1 < len(items) else None
                item_text = self._extract_section_text(report_text, item, next_item)
                
                if item_text:
                    item_chunk = DocumentChunk(
                        id=uuid4().hex,
                        text=item_text,
                        type_chunk="item",
                        document=document
                    )
                    chunks.append(item_chunk)
        
        logger.info(f"Extracted {len(chunks)} chunks from TOC and sections")
        return chunks
    
    def _extract_section_text(self, report_text: str, section_label: str, next_section: str = None) -> str:
        """Extract text for a specific section using fuzzy matching."""
        start, _ = self._find_span_fuzzy(report_text, section_label)
        if start == -1:
            return ""
        
        # Find end position
        if next_section:
            end, _ = self._find_span_fuzzy(report_text, next_section)
            if end == -1 or end <= start:
                end = len(report_text)
        else:
            end = len(report_text)
        
        return report_text[start:end].strip()

    def _compile_fuzzy(self, label: str) -> re.Pattern:
        base = re.sub(r'[\s|]+', '', label.strip())  
        parts = []
        for ch in base:
            if ch == '.':
                parts.append(r'\.?\s*')           # dot often missing; make optional
            else:
                parts.append(re.escape(ch) + r'[\s|]*')
        return re.compile(''.join(parts), re.IGNORECASE)


    def _find_span_fuzzy(self, text: str, label: str) -> tuple[int, int]:
        m = self._compile_fuzzy(label).search(text)
        return (m.start(), m.end()) if m else (-1, -1)


    PARSE_PROMPT = """
Given a TOC clean the page to return a JSON of the Structure of the Document

Sample Output: 
{
    "Part I": [
        "Item 1.", 
        ...
    ],
    "Part II": [
    
    ]
}

"""

    def _parse_out_toc(self, toc_page: str) -> Dict[str, List[str]]:
        """Find the table of contents in the list of tables."""
        from evals.src.utils.generator import OpenAILLM

        llm_client: OpenAILLM = OpenAILLM()

        response = llm_client.generate_response(
            query=toc_page,
            prompt=self.PARSE_PROMPT
        ) 

        return response