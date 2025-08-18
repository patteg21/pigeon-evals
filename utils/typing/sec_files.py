from typing import Optional, List

from pydantic import BaseModel

from utils.typing.common import (
    FormType
)

class SECDocument(BaseModel):
    ticker: str
    company: Optional[str] = None
    date: str
    text: str
    path: str
    report_text: Optional[str] = None   # text that is after the TOC 
    toc: Optional["SECTable"] = None 
    tables: Optional[List["SECTable"]] = [] 
    form_type: FormType
    period_end: Optional[str | None] = None
    commission_number: Optional[str | None] = None
    parts: List["SECPart"] = []


class SECPart(BaseModel):
    id: str
    title: str
    text: Optional[str] = ""
    section: Optional[str] = None
    items: List["SECItem"] = []
    page_number: int = 0
    prev_chunk: Optional["SECPart | SECItem"] = None
    next_chunk: Optional["SECPart | SECItem"] = None

class SECItem(BaseModel):
    id: str
    text: Optional[str] = None # SECItems will throw Pydantic Error when added to VectorObject
    title: str
    subsection: str
    page_number: int = 0
    prev_chunk: Optional["SECPart | SECItem"] = None
    next_chunk: Optional["SECPart | SECItem"] = None


class SECTable(BaseModel):
    id: str
    text: str
    page_number: int

