from typing import List
from pathlib import Path
from uuid import uuid4
import os
import asyncio
import json

import numpy as np
from sklearn.decomposition import PCA

from clients import VectorDB, EmbeddingModel
from utils.typing import (
    VectorObject,
    SECDocument
)
from utils import logger
from preprocessing.extraction import extract_toc, parse_table_of_contents, extract_item_sections
from preprocessing.metadata import get_sec_metadata

# TODO: Add in args for various different functionality


async def process(
        document_text: str, 
        company_name: str, 
        form_type: str, 
        filing_date: str,
        file_path: str,
    ) -> List[VectorObject]:
    """
    Args:
        document_text (str): The processed text content of the filing
        company_name (str): The company ticker symbol
        form_type (str): The type of filing (10K or 10Q)
        filing_date (str): The filing date in YYYY-MM-DD format
        file_path (str): The file path
    """    

    # initialize embedding client
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
    
    # TODO: Handle Tables Individually + Better Chunking

    # basic data from SECDocument to be shared across all items
    document_data = {
        "ticker" : document.ticker,
        "date": document.date,
        "form_type": document.form_type,
        "document_path": document.path,
        "commission_number": document.commission_number,
        "period_end": document.period_end
    }

    # a List of the VectorObject we are looking to write
    vector_objects: List[VectorObject] = []
    for part in document.parts:
        for item in part.items:
            section = part.section
            # Create a vector Object for each Item
            links = {
                "prev_chunk_id": getattr(item.prev_chunk, "id", None),
                "next_chunk_id": getattr(item.next_chunk, "id", None),
            }
            # embeddings = await embedding_model.create_embedding(item.text, strategy="mean")
            item_data = item.__dict__
            vo = VectorObject(
                **document_data,
                **links,
                **item_data,
                section=section,
                embeddings=[],
                entity_type="Item"
            )
            vector_objects.append(vo)

        # Create a vector Object for each Part
        links = {
            "prev_chunk_id": getattr(part.prev_chunk, "id", None),
            "next_chunk_id": getattr(part.next_chunk, "id", None),
        }
        # embeddings = await embedding_model.create_embedding(part.text, strategy="mean")
        part_data = part.__dict__
        vo = VectorObject(
            **document_data,
            **links,
            **part_data,
            embeddings=[],
            entity_type="Part"
        )
        vector_objects.append(vo)

    Path("outputs").mkdir(exist_ok=True)
    output_filename = f"outputs/{company_name}_{filing_date}.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump([vo.model_dump() for vo in vector_objects], f, indent=4)
    return vector_objects



async def process_filings():
    """
    Iterate through all processed filings and call the process function for each.
    """
    base_dir: Path = "data"
    all_vector_objects: List[VectorObject] = []
    
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
            with open(file_path, 'r', encoding='utf-8') as f:
                document_text = f.read()
            
            # Call the process function
            vector_objects: List[VectorObject] = await process(document_text, company_name, form_type, filing_date, file_path)
            all_vector_objects.extend(vector_objects)
            logger.info(f"Processed {filename}")
                

    collect(all_vector_objects)

# TODO: This was implemented due to the Pinecone index size of 512. Alternative to PCA is UMAP 
def collect(vector_objects: List["VectorObject"], target_dim: int = 512):
    """Collect for PCA: reduce in-place, then upload."""
    _reduce_dimensionality(vector_objects, target_dim=target_dim)
    vector_db_client = VectorDB()

    # TODO: clear the previous state of embeddings though only given PCA implementation
    vector_db_client.clear()

    for vector in vector_objects:
        vector_db_client.upload(vector)


# TODO: PCA is typically only suited for single use fitting meaning ingesting new data may be difficult or unoptimal
#       We are assuming as well the Dimensionality with always be larger than target Dim as well
def _reduce_dimensionality(vector_objects: List[VectorObject], target_dim: int = 512, seed: int = 42) -> None:
    X = np.asarray([v.embeddings for v in vector_objects], dtype=np.float32)
    Z = PCA(n_components=target_dim, random_state=seed).fit_transform(X)

    # L2-normalize to keep cosine geometry stable
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)

    # Write back
    for obj, z in zip(vector_objects, Z):
        obj.embeddings = z.tolist()

if __name__ == "__main__":
    asyncio.run(process_filings()) 