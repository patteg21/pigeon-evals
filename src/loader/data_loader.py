from typing import List
from pathlib import Path
import os
from uuid import uuid4
from evals.src.utils.types import (
    SECDocument
)
from evals.src.utils import logger


class DataLoader:

    def __init__(self, path: str ="data"):
        self.base_dir = path
        self.documents = []
    
    @classmethod
    async def create(cls, path: str = "data"):
        """Factory method to create and initialize DataLoader."""
        loader = cls(path)
        loader.documents = await loader.process_filings()
        return loader


    async def process_filings(self):
        """
        Iterate through all processed filings and call the process function for each.
        """
        base_dir = self.base_dir
        documents: List[SECDocument] = []
        
        # Check if the directory exists
        if not os.path.exists(base_dir):
            logger.info(f"Error: {base_dir} directory not found.")
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
                    logger.info(f"Warning: Skipping {filename} - invalid filename format")
                    continue
                    
                _ticker, form_type, filing_date = parts
                
                # Read the file content
                file_path = os.path.join(company_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        document_text = f.read()
                    
                    document = SECDocument(
                        id=uuid4().hex,
                        ticker=company_name,
                        date=filing_date,
                        text=document_text,
                        form_type=form_type,
                        path=file_path
                    )
                    documents.append(document)               
                    logger.info(f"Processed {filename}")
                    
                except Exception as e:
                    logger.error(f"Error processing {filename}: {str(e)}")

        return documents