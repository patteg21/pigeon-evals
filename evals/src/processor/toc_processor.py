import re
from typing import List, Any
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

        raise NotImplementedError

        
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
        
    ROMAN_PART = re.compile(r"(PART\s+[IVXLCDM]+\.?)", re.IGNORECASE)
    ITEM_ROW   = re.compile(r"(Item\s+\d+[A-Z]?\.)\s*\|\s*(.*?)\s*\|\s*(\d+)", re.IGNORECASE)
    def parse_table_of_contents(self, document: SECDocument) -> None:
        toc = document.toc.text

        # TODO: FIX The OCR or handle with REGEX 
        toc = toc.replace("P | art | I", "Part I")
        toc = toc.replace("P | art | II", "Part II")
        toc = toc.replace("P | art | III", "Part III")
        toc = toc.replace("P | art | IV", "Part IV")

        chunks = self.ROMAN_PART.split(toc) 

        parts: List[Any] = []
        items: List[Any] = []

        it = iter(chunks)
        for chunk in it:
            chunk = chunk.strip()
            if not chunk:
                continue

            if self.ROMAN_PART.fullmatch(chunk):
                section = chunk.upper().rstrip(".")
                body: str = (next(it, "") or "").strip()

                sec_part = {
                    "id": uuid4().hex,
                    "title": section,         # e.g. "PART I"
                    "section": section
                }
                parts.append(sec_part)
                document.parts.append(sec_part)

                for m in self.ITEM_ROW.finditer(body):
                    subsection = m.group(1).strip()   # "Item 1."
                    title = m.group(2).strip()        # "Financial Statements"
                    _page = m.group(3).strip()         # "3"

                    sec_item = {
                        "id":uuid4().hex,
                        "title": title,
                        "subsection": subsection,
                    }
                    sec_part.items.append(sec_item)
                    items.append(sec_item)



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


    def extract_item_sections(self, document: SECDocument) -> None:
        """
        Build a single linked sequence across parts and items via prev_chunk/next_chunk.
        Also sets page_number for both parts and items based on [PAGE_BREAK] markers,
        and fills item.text spans. Handles OCR like 'I TEM 11.' by fuzzy matching.
        """
        body = document.report_text or ""
        if not body or not getattr(document, "parts", None):
            return

        positions: list[tuple[int, object]] = []

        for part in document.parts or []:
            if not getattr(part, "items", None):
                continue

            # Locate PART header fuzzily (handles 'P A R T  I', etc.)
            part_start, _part_end = self._find_span_fuzzy(body, part.title or part.section or "PART")
            if part_start == -1:
                part_start, _part_end = (0, 0)  # fallback

            part.page_number = self._compute_page_number(body, part_start)
            positions.append((part_start, part))

            # Preface = between PART header and first Item (fuzzy)
            if part.items:
                f0 = part.items[0]
                first_item_start, _ = self._find_span_fuzzy(body, f0.subsection)
                if first_item_start != -1 and first_item_start > part_start:
                    part.text = body[part_start:first_item_start].strip()

            # Items
            for j, item in enumerate(part.items):
                start, _ = self._find_span_fuzzy(body, item.subsection)
                if start == -1:
                    continue

                if j + 1 < len(part.items):
                    # end at next item's fuzzy start
                    next_label = part.items[j + 1].subsection
                    end, _dummy = self._find_span_fuzzy(body, next_label)
                    if end == -1 or end <= start:
                        end = len(body)
                else:
                    end = len(body)

                item.text = body[start:end].strip()
                item.page_number = self._compute_page_number(body, start)
                positions.append((start, item))



    def _extract_tables(self, document: SECDocument) -> List[SECTable]:
        """Extract all tables from document text."""
        TABLE_RE = re.compile(r"\[TABLE_START\](.*?)\[[ ]*TABLE_END\]", re.DOTALL | re.IGNORECASE)
        # TOC found on second page
        body: str = document.text.split("[PAGE BREAK]")[1]

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