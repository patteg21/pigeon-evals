from typing import List

from uuid import uuid4
import re


from utils.typing import (
    SECDocument, 
    SECPart, 
    SECItem,
    SECTable
)

def _compute_page_number(body: str, pos: int) -> int:
    """Pages start at 1; count [PAGE_BREAK] before `pos`."""
    if pos < 0:
        pos = 0
    return body[:pos].count("[PAGE_BREAK]") + 1


def extract_toc(document: SECDocument) -> SECTable:
    tables: List[SECTable]= extract_tables(document)
    
    filtered_tables: List[str] = []


    for table in tables:
        if "Item 1." in table.text:
            table_of_contents = table  # store the first TOC-like table
        else:
            filtered_tables.append(table)

    document.toc = table_of_contents
    document.tables = filtered_tables
    return table_of_contents


# TODO: Improve the naive search for table of contents
TABLE_RE  = re.compile(r"\[TABLE_START\](.*?)\[[ ]*TABLE_END\]", re.DOTALL | re.IGNORECASE)
def extract_tables(entity: SECDocument | SECItem | SECPart) -> List[SECTable]:
    """
    Extract tables and attach a page number based on their position in `entity.text`.
    """
    body = getattr(entity, "text", "") or ""
    tables: List[SECTable] = []

    for m in TABLE_RE.finditer(body):
        # Page is computed at the start of the table block
        start_pos = m.start()                 # start of [TABLE_START]
        text = (m.group(1) or "").strip()     # inner table contents only
        page = _compute_page_number(body, start_pos)

        tables.append(SECTable(
            id=uuid4().hex,
            page_number=page,
            text=text,
        ))

    return tables





ROMAN_PART = re.compile(r"(PART\s+[IVXLCDM]+\.?)", re.IGNORECASE)
ITEM_ROW   = re.compile(r"(Item\s+\d+[A-Z]?\.)\s*\|\s*(.*?)\s*\|\s*(\d+)", re.IGNORECASE)
def parse_table_of_contents(document: SECDocument) -> None:
    toc = document.toc.text

    # TODO: FIX The OCR or handle with REGEX 
    toc = toc.replace("P | art | I", "Part I")
    toc = toc.replace("P | art | II", "Part II")
    toc = toc.replace("P | art | III", "Part III")
    toc = toc.replace("P | art | IV", "Part IV")

    chunks = ROMAN_PART.split(toc) 

    parts: List[SECPart] = []
    items: List[SECItem] = []

    it = iter(chunks)
    for chunk in it:
        chunk = chunk.strip()
        if not chunk:
            continue

        if ROMAN_PART.fullmatch(chunk):
            section = chunk.upper().rstrip(".")
            body: str = (next(it, "") or "").strip()

            sec_part = SECPart(
                id=uuid4().hex,
                title=section,         # e.g. "PART I"
                section=section
            )
            parts.append(sec_part)
            document.parts.append(sec_part)

            for m in ITEM_ROW.finditer(body):
                subsection = m.group(1).strip()   # "Item 1."
                title = m.group(2).strip()        # "Financial Statements"
                _page = m.group(3).strip()         # "3"

                sec_item = SECItem(
                    id=uuid4().hex,
                    title=title,
                    subsection=subsection,
                )
                sec_part.items.append(sec_item)
                items.append(sec_item)





def _compile_fuzzy(label: str) -> re.Pattern:
    base = re.sub(r'[\s|]+', '', label.strip())  
    parts = []
    for ch in base:
        if ch == '.':
            parts.append(r'\.?\s*')           # dot often missing; make optional
        else:
            parts.append(re.escape(ch) + r'[\s|]*')
    return re.compile(''.join(parts), re.IGNORECASE)


def _find_span_fuzzy(text: str, label: str) -> tuple[int, int]:
    m = _compile_fuzzy(label).search(text)
    return (m.start(), m.end()) if m else (-1, -1)


def extract_item_sections(document: SECDocument) -> None:
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
        part_start, part_end = _find_span_fuzzy(body, part.title or part.section or "PART")
        if part_start == -1:
            part_start, part_end = (0, 0)  # fallback

        part.page_number = _compute_page_number(body, part_start)
        positions.append((part_start, part))

        # Preface = between PART header and first Item (fuzzy)
        if part.items:
            f0 = part.items[0]
            first_item_start, _ = _find_span_fuzzy(body, f0.subsection)
            if first_item_start != -1 and first_item_start > part_start:
                part.text = body[part_start:first_item_start].strip()

        # Items
        for j, item in enumerate(part.items):
            start, _ = _find_span_fuzzy(body, item.subsection)
            if start == -1:
                continue

            if j + 1 < len(part.items):
                # end at next item's fuzzy start
                next_label = part.items[j + 1].subsection
                end, _dummy = _find_span_fuzzy(body, next_label)
                if end == -1 or end <= start:
                    end = len(body)
            else:
                end = len(body)

            item.text = body[start:end].strip()
            item.page_number = _compute_page_number(body, start)
            positions.append((start, item))

    # Wire prev/next
    positions.sort(key=lambda x: x[0])
    for i, (_, node) in enumerate(positions):
        setattr(node, "prev_chunk", positions[i - 1][1] if i > 0 else None)
        setattr(node, "next_chunk", positions[i + 1][1] if i + 1 < len(positions) else None)