from typing import Dict, List
import re
from uuid import uuid4

from utils.typing import (
    SECDocument,
    SECItem,
    VectorObject,
    FormType
)

from mcp_server.clients import EmbeddingModel, SQLClient


# TODO: Get more metadata such as the various Q&A
def get_sec_metadata(document: SECDocument, form_type: FormType) -> Dict[str, str]:
    """
    Extract SEC metadata. Dynamically changed based on the form type
    """

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


def _store_text_in_sqlite(sql_client: SQLClient, text_content: str) -> str:
    """Store text content in SQLite and return document ID"""
    doc_id = uuid4().hex
    doc_data = {'text': text_content}
    sql_client.store_document(doc_id, doc_data)
    return doc_id


async def _chunk_by_page_and_max_tokens(item: SECItem, embedding_model: EmbeddingModel) -> List[SECItem]:
    max_tokens = embedding_model.max_tokens
    overlap = 128

    # split into pages
    pages: List[str] = [p.strip() for p in item.text.split("[PAGE BREAK]") if p.strip()]
    
    chunks, cur_text, cur_start_page = [], "", item.page_number

    # pack into chunks across multiple pages
    for offset, page in enumerate(pages):
        candidate = (cur_text + " " + page).strip()
        if await embedding_model.count_tokens(candidate) <= max_tokens:
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
        item.prev_chunk.next_chunk = results[0]
    if item.next_chunk:
        item.next_chunk.prev_chunk = results[-1]
    
    return results

async def create_vector_objects(document: SECDocument, embedding_model: EmbeddingModel) -> List[VectorObject]:
    """ Further deconstructs from the TOC deconstructions and hands back based on a document the writable VectorObject.
    It also writes the text for each item to the SQL Database 
    """
    sql_client: SQLClient = SQLClient()

    document_data = {
        "ticker" : document.ticker,
        "date": document.date,
        "form_type": document.form_type,
        "document_path": document.path,
        "commission_number": document.commission_number,
        "period_end": document.period_end
    }

    vector_objects: List[VectorObject] = []
    for part in document.parts:
        for item in part.items:
            section = part.section
            # Create a vector Object for each Item
            links = {
                "prev_chunk_id": getattr(item.prev_chunk, "id", None),
                "next_chunk_id": getattr(item.next_chunk, "id", None),
            }
            token_count = await embedding_model.count_tokens(item.text)
            if token_count > embedding_model.max_tokens:
                # TODO: handles very large files and chunks them due to metadata limitation 
                split_sec_items: List[SECItem] = await _chunk_by_page_and_max_tokens(item, embedding_model) 
                for split in split_sec_items: 
                    item_data = split.__dict__.copy()
                    embeddings = await embedding_model.create_embedding(split.text, strategy="mean")

                    # Store text in SQLite and get document ID
                    doc_id = _store_text_in_sqlite(sql_client, split.text)
                    item_data['text'] = ""  # Clear text from vector metadata
                    item_data['document_id'] = doc_id

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
                embeddings = await embedding_model.create_embedding(item.text, strategy="mean")
                item_data = item.__dict__.copy()
                
                # Store text in SQLite and get document ID
                doc_id = _store_text_in_sqlite(sql_client, item.text)
                item_data['text'] = ""  # Clear text from vector metadata
                item_data['document_id'] = doc_id
                
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
        part_data = part.__dict__.copy()
        
        # Store text in SQLite and get document ID
        doc_id = _store_text_in_sqlite(sql_client, part.text)
        part_data['text'] = ""  # Clear text from vector metadata
        part_data['document_id'] = doc_id
        
        vo = VectorObject(
            **document_data,
            **links,
            **part_data,
            embeddings=embeddings,
            entity_type="Part"
        )
        vector_objects.append(vo)


    for table in document.tables:
        table_data = table.__dict__.copy()
        embeddings = await embedding_model.create_embedding(table.text, strategy="mean")
        
        # Store text in SQLite and get document ID
        doc_id = _store_text_in_sqlite(sql_client, table.text)
        table_data['text'] = ""  # Clear text from vector metadata
        table_data['document_id'] = doc_id
        
        vo = VectorObject(
            **document_data,
            **table_data,
            embeddings=embeddings,
            entity_type="Table"
        )
        vector_objects.append(vo)
