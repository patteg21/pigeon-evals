from typing import Optional, List

from pydantic import BaseModel


class Metadata(BaseModel):
    """Comprehensive metadata extracted from  documents."""
    pass

class Document(BaseModel):
    id: str
    name: str
    text: str



class DocumentChunk(BaseModel):
    id: str
    text: str
    document: Document 
    embeddding: Optional[List[float]] = None


class Table(BaseModel):
    id: str
    page_number: Optional[int] = None
    text: str
