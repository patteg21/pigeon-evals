from typing import Optional, List, Any
from datetime import datetime

from pydantic import BaseModel, model_validator, Field


class Metadata(BaseModel):
    """Comprehensive metadata extracted from  documents."""
    pass

class Document(BaseModel):
    id: str
    name: str



class DocumentChunk(BaseModel):
    id: str
    text: str
    type_chunk: str
    document: Document 
    embeddding: Optional[List[float]] = None


class Table(BaseModel):
    id: str
    page_number: Optional[int] = None
    text: str
