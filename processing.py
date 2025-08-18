from typing import List
from pathlib import Path
from uuid import uuid4
import os
import asyncio
import json
import re

from clients import VectorDB, EmbeddingModel
from utils.typing import (
    VectorObject,
    SECDocument,
    SECTable,
    SECItem
)
from utils import logger
from utils.pca import PCALoader

from preprocessing.extraction import extract_toc, parse_table_of_contents, extract_item_sections
from preprocessing.metadata import get_sec_metadata


import re
from uuid import uuid4
from typing import List

def _chunk_by_page_and_max_tokens(item: SECItem, embedding_model: EmbeddingModel) -> List[SECItem]:
    max_tokens = embedding_model.max_tokens
    overlap = 128

    # split into pages
    pages: List[str] = [p.strip() for p in item.text.split("[PAGE BREAK]") if p.strip()]
    
    chunks, cur_text, cur_start_page = [], "", item.page_number

    # pack into chunks across multiple pages
    for offset, page in enumerate(pages):
        candidate = (cur_text + " " + page).strip()
        if embedding_model.count_tokens(candidate) <= max_tokens:
            # safe to keep accumulating
            if not cur_text:
                cur_start_page = item.page_number + offset
            cur_text = candidate
        else:
            if cur_text:
                chunks.append((cur_text.strip(), cur_start_page))
            # start new chunk from this page
            cur_text = page
            cur_start_page = item.page_number + offset
    if cur_text:
        chunks.append((cur_text.strip(), cur_start_page))

    # add overlap
    overlapped = []
    for i, (chunk, start_page) in enumerate(chunks):
        if i > 0:
            prev_tokens = re.findall(r"\S+", chunks[i-1][0])
            overlap_tokens = prev_tokens[-overlap:]
            chunk = " ".join(overlap_tokens) + " " + chunk
        overlapped.append((chunk.strip(), start_page))

    # build new SECItems
    results: List[SECItem] = []
    for i, (txt, start_page) in enumerate(overlapped):
        sec = SECItem(
            id=uuid4().hex,
            text=txt,
            title=item.title,
            subsection=item.subsection,
            page_number=start_page,   # starting page number of this chunk
            prev_chunk=None,
            next_chunk=None,
        )
        results.append(sec)

    # wire up prev/next
    for i, sec in enumerate(results):
        if i == 0:
            sec.prev_chunk = item.prev_chunk
        else:
            sec.prev_chunk = results[i-1].id
        if i == len(results) - 1:
            sec.next_chunk = item.next_chunk
        else:
            sec.next_chunk = results[i+1]

    # updates the pre-exisiting 
    if item.prev_chunk:
        item.prev_chunk.next_chunk = results[0].id
    if item.next_chunk:
        item.next_chunk.prev_chunk = results[-1].id
    
    return results



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
    table_of_contents: SECTable = extract_toc(document)
    toc_index = document.text.find(table_of_contents.text)

    # Get all the text after the table of contents
    document.report_text = document.text[toc_index + len(table_of_contents.text):]

    parse_table_of_contents(document)
    extract_item_sections(document)
    
    # TODO: Refactor into more composible code
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
            embeddings = await embedding_model.create_embedding(item.text, strategy="mean")
            if embedding_model.count_tokens(item.text) > embedding_model.max_tokens:
                # TODO: handles very large files and chunks them due to metadata limitation 
                split_sec_items: List[SECItem] = _chunk_by_page_and_max_tokens(item, embedding_model) 
                for split in split_sec_items: 
                    item_data = split.__dict__
                    links = {
                        "prev_chunk_id": getattr(item.prev_chunk, "id", None),
                        "next_chunk_id": getattr(item.next_chunk, "id", None),
                    }
                    vo = VectorObject(
                        **document_data,
                        **links,
                        **item_data,
                        section=section,
                        embeddings=embeddings,
                        entity_type="Item"
                    )
                    vector_objects.append(vo)
            else:
                item_data = item.__dict__

                vo = VectorObject(
                    **document_data,
                    **links,
                    **item_data,
                    section=section,
                    embeddings=embeddings,
                    entity_type="Item"
                )
                vector_objects.append(vo)

        # Create a vector Object for each Part
        links = {
            "prev_chunk_id": getattr(part.prev_chunk, "id", None),
            "next_chunk_id": getattr(part.next_chunk, "id", None),
        }
        embeddings = await embedding_model.create_embedding(part.text, strategy="mean")
        part_data = part.__dict__
        vo = VectorObject(
            **document_data,
            **links,
            **part_data,
            embeddings=embeddings,
            entity_type="Part"
        )
        vector_objects.append(vo)


    for table in document.tables:
        table_data = table.__dict__
        embeddings = await embedding_model.create_embedding(table.text, strategy="mean")
        vo = VectorObject(
            **document_data,
            **table_data,
            embeddings=embeddings,
            entity_type="Table"
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
    for company_name in os.listdir(base_dir)[:2]:
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
            
            try:
                # Read the file content
                file_path = os.path.join(company_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    document_text = f.read()
                
                # Call the process function
                vector_objects: List[VectorObject] = await process(document_text, company_name, form_type, filing_date, file_path)
                all_vector_objects.extend(vector_objects)
                logger.info(f"Processed {filename}")
            except Exception as e:
                logger.error(f"ERROR: {e}")

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

    for vector in vector_objects:
        # ensure empty text fields are strings if needed
        if getattr(vector, "text", None) is None:
            vector.text = ""
        vector_db_client.upload(vector)


if __name__ == "__main__":
    asyncio.run(process_filings()) 