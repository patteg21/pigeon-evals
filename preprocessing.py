from typing import List
from pathlib import Path
from uuid import uuid4
import os
import asyncio
import json

from mcp_server.clients import VectorDB, EmbeddingModel
from utils.typing import (
    VectorObject,
    SECDocument,
    SECTable,
)
from utils import logger
from utils.pca import PCALoader
from preprocess.extraction import extract_toc, parse_table_of_contents, extract_item_sections
from preprocess.metadata import get_sec_metadata, create_vector_objects


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
    # initialize clients
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
    table_of_contents: SECTable = extract_toc(document)
    toc_index = document.text.find(table_of_contents.text)

    # Get all the text after the table of contents
    document.report_text = document.text[toc_index + len(table_of_contents.text):]

    parse_table_of_contents(document)
    extract_item_sections(document)

    # create the VectorObject we are looking to write
    vector_objects: List[VectorObject] = create_vector_objects(document, embedding_model)

    if False:
        Path("outputs").mkdir(exist_ok=True)
        output_filename = f"outputs/{company_name}_{filing_date}.json"
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump([vo.model_dump() for vo in vector_objects], f, indent=4)
    return vector_objects



async def _process_file_worker(semaphore: asyncio.Semaphore, company_name: str, filename: str, company_dir: str) -> List[VectorObject]:
    """Worker function to process a single file with threshold control"""
    async with semaphore:
        # Parse file information from filename
        # Format: {ticker}_{file_type}_{date}.txt
        parts = filename.replace('.txt', '').split('_')
        if len(parts) != 3:
            logger.warning(f"Warning: Skipping {filename} - invalid filename format")
            return []
        
        form_type: str
        filing_date: str
        _ticker, form_type, filing_date = parts # Ticker is unused as already collected as `company_name`
        
        try:
            # Read the file content
            file_path = os.path.join(company_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                document_text = f.read()
            
            # Call the process function
            vector_objects: List[VectorObject] = await process(document_text, company_name, form_type, filing_date, file_path)
            logger.info(f"Processed {filename}")
            return vector_objects
        except Exception as e:
            logger.error(f"ERROR processing {filename}: {e}")
            return []


async def process_filings():
    """
    Iterate through all processed filings and call the process function for each.
    Uses 8 threshold workers to control concurrent processing.
    """
    base_dir: Path = "data"
    all_vector_objects: List[VectorObject] = []
    
    # Create semaphore for 8 threshold workers
    semaphore = asyncio.Semaphore(8)
    
    # Check if the directory exists
    if not os.path.exists(base_dir):
        logger.error(f"Error: {base_dir} directory not found.")
        return
    
    # Collect all files to process
    tasks = []
    
    # Iterate through each company directory
    for company_name in os.listdir(base_dir):
        company_dir = os.path.join(base_dir, company_name)
        
        # Skip if not a directory
        if not os.path.isdir(company_dir):
            continue
            
        logger.info(f"Queueing filings for {company_name}...")
        
        # Iterate through each file in the company directory
        for filename in os.listdir(company_dir):
            if not filename.endswith('.txt'):
                continue
            
            # Create task for worker processing
            task = _process_file_worker(semaphore, company_name, filename, company_dir)
            tasks.append(task)
    
    logger.info(f"Processing {len(tasks)} files with 8 threshold workers...")
    
    # Execute all tasks concurrently with threshold control
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Collect all vector objects from successful results
    for result in results:
        if isinstance(result, list):
            all_vector_objects.extend(result)
        elif isinstance(result, Exception):
            logger.error(f"Task failed with exception: {result}")

    await collect(all_vector_objects)



# TODO: This was implemented due to the Pinecone index size of 512. Alternative to PCA is UMAP  
def train_pca_and_reduce_in_place(vector_objects: List["VectorObject"], pca: PCALoader) -> None:
    logger.info("PCA FITTING")
    X = [v.embeddings for v in vector_objects]
    pca.fit(X).save()
    Z = pca.transform(X)
    for obj, z in zip(vector_objects, Z):
        obj.embeddings = z.tolist()
    logger.info("PCA COMPLETE")


async def collect(vector_objects: List["VectorObject"], target_dim: int = 512, pca_path: str = "artifacts/sec_pca_512.joblib"):
    """
    Train once and persist (first run), or load and apply (subsequent runs),
    then upload to VectorDB.
    """
    pca = PCALoader(path=pca_path, target_dim=target_dim, seed=42)


    train_pca_and_reduce_in_place(vector_objects, pca)

    vector_db_client = VectorDB()
    try: 
        vector_db_client.clear()
    except Exception as e:
        logger.error(f"Unable to Clear Pinecone DB: {e}")

    for vector in vector_objects:
        # ensure empty text fields are strings if needed
        if getattr(vector, "text", None) is None:
            vector.text = ""
        vector_db_client.upload(vector)


if __name__ == "__main__":
    asyncio.run(process_filings()) 