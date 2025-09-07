from typing import Optional, List
from uuid import uuid4

from pydantic import BaseModel


class Metadata(BaseModel):
    """Comprehensive metadata extracted from  documents."""
    pass

class Document(BaseModel):
    id: str = uuid4().hex
    name: str
    path: str
    text: str



class DocumentChunk(BaseModel):
    id: str = uuid4().hex
    text: str
    document: Document 
    embeddding: Optional[List[float]] = None


class Table(BaseModel):
    id: str
    page_number: Optional[int] = None
    text: str
