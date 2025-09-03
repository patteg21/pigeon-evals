from typing import Optional, List

from pydantic import BaseModel

from src.utils.types.sec_files import SECDocument

class DocumentChunk(BaseModel):
    id: str
    text: str
    type_chunk: str
    document: SECDocument 
    embeddding: Optional[List[float]] = None