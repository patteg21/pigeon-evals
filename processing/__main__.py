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

from processing.extraction import extract_tables, extract_toc, parse_table_of_contents, extract_item_sections
from processing.metadata import get_sec_metadata

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


if __name__ == "__main__":
    asyncio.run(process_filings()) 