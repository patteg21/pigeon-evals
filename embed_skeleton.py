from typing import List, Dict, Literal, Tuple
from pathlib import Path
from uuid import uuid4
import os
import asyncio
import re
import json

from clients import VectorDB, EmbeddingModel

from utils.typing import (
    VectorObject,
    SECDocument, 
    SECPart, 
    SECItem
)


async def process(
        document_text: str, 
        company_name: str, 
        form_type: str, 
        filing_date: str,
        file_path: str
    ):
    """
    Args:
        document_text (str): The processed text content of the filing
        company_name (str): The company ticker symbol
        form_type (str): The type of filing (10K or 10Q)
        filing_date (str): The filing date in YYYY-MM-DD format
        file_path (str): The file path
    """    

    # initialize clients
    vector_db_client = VectorDB()
    embedding_model = EmbeddingModel()

    document = SECDocument(
        ticker=company_name,
        date=filing_date,
        text=document_text,
        form_type=form_type,
        path=file_path
    )

    get_sec_metadata(document, form_type)
    extract_toc(document)
    toc_index = document.text.find(document.toc)

    # Get all the text after the table of contents
    document.report_text = document.text[toc_index + len(document.toc):]

    parse_table_of_contents(document)
    extract_item_sections(document)
    

    TABLE_BLOCK_RE = re.compile(r"\[TABLE_START\].*?\[TABLE_END\]", re.DOTALL)

    all_vector_objects: List[VectorObject] = []
    for part in document.parts:
        # For the Preface
        vec_obj = VectorObject(
            id=uuid4().hex,
            ticker=document.ticker,
            date=document.date,
            commission_number=document.commission_number,
            period_end=document.period_end,
            document_path=document.path,
            embeddings=[],
            text=part.preface,
            chunk_type="preface",
        )
        
        all_vector_objects.append(vec_obj)

        for item in part.items:
            # Get current text safely
            text: str = item.text

            tokens = embedding_model.count_tokens(text)
            if tokens > 8191:
                # INLINE removal of tables
                cleaned = TABLE_BLOCK_RE.sub("", text).strip()
                item.text = cleaned  # update in place

                # optional re-check
                new_tokens = embedding_model.count_tokens(item.text)
                if new_tokens > 8191:
                    
                    # A naive break by page
                    pages = item.text.split("[PAGE BREAK]")

                    for page in pages:
                        new_tokens = embedding_model.count_tokens(page)
                        if new_tokens > 8191:
                            print(f"still too big -- {document.path} -- ({new_tokens} tokens) -> {getattr(item, 'title', 'unknown item')}")
        
                        



    # Write result to JSON file
    Path("outputs").mkdir(exist_ok=True)
    output_filename = f"outputs/{company_name}_{filing_date}.json"
    with open(output_filename, "w") as f:
        f.write(document.model_dump_json(indent=4))


    




async def process_filings():
    """
    Iterate through all processed filings and call the process function for each.
    """
    base_dir: Path = "data"
    
    # Check if the directory exists
    if not os.path.exists(base_dir):
        print(f"Error: {base_dir} directory not found.")
        return
    
    # Iterate through each company directory
    for company_name in os.listdir(base_dir):
        company_dir = os.path.join(base_dir, company_name)
        
        # Skip if not a directory
        if not os.path.isdir(company_dir):
            continue
            
        # print(f"\nProcessing filings for {company_name}...")
        
        # Iterate through each file in the company directory
        for filename in os.listdir(company_dir):
            if not filename.endswith('.txt'):
                continue
                
            # Parse file information from filename
            # Format: {ticker}_{file_type}_{date}.txt
            parts = filename.replace('.txt', '').split('_')
            if len(parts) != 3:
                print(f"Warning: Skipping {filename} - invalid filename format")
                continue
            
            ticker: str
            form_type: str
            filing_date: str
            ticker, form_type, filing_date = parts
            
            # Read the file content
            file_path = os.path.join(company_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    document_text = f.read()
                
                # Call the process function
                await process(document_text, company_name, form_type, filing_date, file_path)
                # print(f"Processed {filename}")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")



# TODO: Get more metadata such as the various Q&A
def get_sec_metadata(document: SECDocument, form_type: Literal["10K", "10Q"]) -> Dict[str, str]:
    """Extract SEC metadata."""

    # Date patterns
    if form_type == "10K":
        pattern = r"For the fiscal year ended\s+([A-Za-z]+\s+\d{1,2}\s*,\s*\d{4})"
    elif form_type == "10Q":
        pattern = r"For the quarterly period ended\s+([A-Za-z]+\s+\d{1,2}\s*,\s*\d{4})"
    else:
        pattern = ""

    dates = re.findall(pattern, document.text, re.IGNORECASE) if pattern else []
    period_end = dates[1] if len(dates) > 1 else (dates[0] if dates else None)

    # Extract commission file number
    comm_match = re.search(
        r"Commission\s+[Ff]ile\s+(?:[Nn]umber|[Nn]o\.?):?\s*([0-9]{1,3}-[0-9]{5})",
        document.text,
        re.IGNORECASE,
    )
    commission = comm_match.group(1) if comm_match else None

    document.commission_number = commission
    document.period_end = period_end

    return {"period_end": period_end, "commission_number": commission}



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

            # ✅ Update the actual nested SECItem.text
            item.text = body[start:end].strip()


# TODO: For compute, cache system or search in DB to see if it exists already
def embedded_text(embedding_model: EmbeddingModel, text: str):
    tokens: int = embedding_model.count_tokens()
    
    pass

if __name__ == "__main__":
    asyncio.run(process_filings()) 