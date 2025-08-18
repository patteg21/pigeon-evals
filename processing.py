from typing import List
from pathlib import Path
from uuid import uuid4
import os
import asyncio

from clients import VectorDB, EmbeddingModel

from utils.typing import (
    VectorObject,
    SECDocument
)
from utils import logger

from preprocessing.extraction import extract_toc, parse_table_of_contents, extract_item_sections
from preprocessing.metadata import get_sec_metadata

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
        id=uuid4().hex,
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
    
    all_vector_objects: List[VectorObject] = []



async def process_filings():
    """
    Iterate through all processed filings and call the process function for each.
    """
    base_dir: Path = "data"
    
    # Check if the directory exists
    if not os.path.exists(base_dir):
        logger.error(f"Error: {base_dir} directory not found.")
        return
    
    # Iterate through each company directory
    for company_name in os.listdir(base_dir):
        company_dir = os.path.join(base_dir, company_name)
        
        # Skip if not a directory
        if not os.path.isdir(company_dir):
            continue
            
        logger.info(f"\nProcessing filings for {company_name}...")
        
        # Iterate through each file in the company directory
        for filename in os.listdir(company_dir):
            if not filename.endswith('.txt'):
                continue
                
            # Parse file information from filename
            # Format: {ticker}_{file_type}_{date}.txt
            parts = filename.replace('.txt', '').split('_')
            if len(parts) != 3:
                logger.warning(f"Warning: Skipping {filename} - invalid filename format")
                continue
            
            form_type: str
            filing_date: str
            _ticker, form_type, filing_date = parts # Ticker is unused as already collected as `company_name`
            
            # Read the file content
            file_path = os.path.join(company_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    document_text = f.read()
                
                # Call the process function
                await process(document_text, company_name, form_type, filing_date, file_path)
                logger.info(f"Processed {filename}")
                
            except Exception as e:
                logger.info(f"Error processing {filename}: {str(e)}")



if __name__ == "__main__":
    asyncio.run(process_filings()) 