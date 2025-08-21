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
        
        # Create TOC chunk
        toc_chunk = DocumentChunk(
            id=uuid4().hex,
            text=toc_page,
            type_chunk="toc",
            document=document
        )
        chunks.append(toc_chunk)
        
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


    def extract_item_sections(self, document: SECDocument, report_text: str) -> None:
        """
        Build a single linked sequence across parts and items via prev_chunk/next_chunk.
        Also sets page_number for both parts and items based on [PAGE_BREAK] markers,
        and fills item.text spans. Handles OCR like 'I TEM 11.' by fuzzy matching.
        """
        if not report_text or not getattr(document, "parts", None):
            return

        positions: list[tuple[int, object]] = []

        for part in document.parts or []:
            if not part.get("items"):
                continue

            # Locate PART header fuzzily (handles 'P A R T  I', etc.)
            part_start, _part_end = self._find_span_fuzzy(report_text, part.get("title") or part.get("section") or "PART")
            if part_start == -1:
                part_start, _part_end = (0, 0)  # fallback

            part["page_number"] = self._compute_page_number(report_text, part_start)
            positions.append((part_start, part))

            # Preface = between PART header and first Item (fuzzy)
            if part["items"]:
                f0 = part["items"][0]
                first_item_start, _ = self._find_span_fuzzy(report_text, f0["subsection"])
                if first_item_start != -1 and first_item_start > part_start:
                    part["text"] = report_text[part_start:first_item_start].strip()

            # Items
            for j, item in enumerate(part["items"]):
                start, _ = self._find_span_fuzzy(report_text, item["subsection"])
                if start == -1:
                    continue

                if j + 1 < len(part["items"]):
                    # end at next item's fuzzy start
                    next_label = part["items"][j + 1]["subsection"]
                    end, _dummy = self._find_span_fuzzy(report_text, next_label)
                    if end == -1 or end <= start:
                        end = len(report_text)
                else:
                    end = len(report_text)

                item["text"] = report_text[start:end].strip()
                item["page_number"] = self._compute_page_number(report_text, start)
                positions.append((start, item))


    
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
        from evals.src.utils.llm import OpenAILLM

        llm_client: OpenAILLM = OpenAILLM()

        response = llm_client.generate_response(
            query=toc_page,
            prompt=self.PARSE_PROMPT
        ) 

        return response