from typing import Optional, List
from uuid import uuid4

from pydantic import BaseModel, Field


class Metadata(BaseModel):
    """Comprehensive metadata extracted from  documents."""
    pass

class Document(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    name: str
    path: str
    text: str



class DocumentChunk(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    text: str
    document: Document
    embedding: Optional[List[float]] = None
    type_chunk: Optional[str] = None


class Table(BaseModel):
    id: str
    page_number: Optional[int] = None
    text: str
