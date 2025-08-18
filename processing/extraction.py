from typing import List, Tuple

from uuid import uuid4
import re


from clients import EmbeddingModel

from utils.typing import (
    SECDocument, 
    SECPart, 
    SECItem
)

def extract_toc(document: SECDocument):
    tables = extract_tables(document)
    
    table_of_contents = None
    filtered_tables: List[str] = []


    for table in tables:
        if not table_of_contents and "Item 1." in table:
            table_of_contents = table  # store the first TOC-like table
        else:
            filtered_tables.append(table)

    document.toc = table_of_contents
    document.tables = filtered_tables


# TODO: Improve the naive search for table of contents
def extract_tables(entity: SECDocument | SECItem | SECPart):
    """

    """
    table_pattern = re.compile(r"\[TABLE_START\](.*?)\[[ ]*TABLE_END\]", re.DOTALL | re.IGNORECASE)
    tables: List[str] = [t.strip() for t in table_pattern.findall(entity.text)]

    return tables




ROMAN_PART = re.compile(r"(PART\s+[IVXLCDM]+\.?)", re.IGNORECASE)
ITEM_ROW   = re.compile(r"(Item\s+\d+[A-Z]?\.)\s*\|\s*(.*?)\s*\|\s*(\d+)", re.IGNORECASE)
def parse_table_of_contents(document: SECDocument) -> Tuple[List[SECPart], List[SECItem]]:
    toc = document.toc
    chunks = ROMAN_PART.split(toc)  # ["...", "PART I.", "body...", "PART II.", "body...", ...]

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

            # preface = body before first Item
            first_item = ITEM_ROW.search(body)
            preface = body[: first_item.start()].strip() if first_item else body

            sec_part = SECPart(
                id=uuid4().hex,
                title=section,         # e.g. "PART I"
                preface=preface,       # free text before first Item
                section=section
            )
            parts.append(sec_part)
            document.parts.append(sec_part)

            for m in ITEM_ROW.finditer(body):
                subsection = m.group(1).strip()   # "Item 1."
                title = m.group(2).strip()        # "Financial Statements"
                page = m.group(3).strip()         # "3"

                sec_item = SECItem(
                    title=title,
                    subsection=subsection,
                    page=page,
                )
                sec_part.items.append(sec_item)
                items.append(sec_item)

    return parts, items




def extract_item_sections(document: SECDocument) -> None:
    """
    Break document.report_text into sections and update:
      - SECPart.preface with text between PART header and its first Item
      - SECItem.text with the text for each Item
    """
    body = document.report_text or document.text or ""
    if not body or not document.parts:
        return

    low = body.lower()

    for part in document.parts:
        if not part.items:
            continue

        # Find start of this PART in the body
        part_start = low.find(part.title.lower())
        if part_start == -1:
            part_start = 0

        # Preface = between PART header and first Item
        first_item = part.items[0]
        first_item_start = low.find(first_item.subsection.lower(), part_start)
        if first_item_start != -1:
            part.preface = body[part_start:first_item_start].strip()

        # Now update text for each Item
        for idx, item in enumerate(part.items):
            start = low.find(item.subsection.lower(), part_start)
            if start == -1:
                continue

            if idx + 1 < len(part.items):
                # Until the next item
                next_label = part.items[idx + 1].subsection.lower()
                end = low.find(next_label, start + 1)
                if end == -1:
                    end = len(body)
            else:
                # Last item goes to end of body
                end = len(body)

            # SECItem.text for the chunk
            item.text = body[start:end].strip()


# TODO: For compute, cache system or search in DB to see if it exists already
def embedded_text(embedding_model: EmbeddingModel, text: str):
    tokens: int = embedding_model.count_tokens()
    
    pass

