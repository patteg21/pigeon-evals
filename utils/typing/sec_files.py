from typing import Optional, List
from pathlib import Path

from pydantic import BaseModel

from utils.typing.common import (
    FormType
)

class SECDocument(BaseModel):
    ticker: str
    company: Optional[str] = None
    date: str
    text: str
    path: Path
    report_text: Optional[str] = None # text that is after the TOC 
    toc: Optional[str] = None 
    tables: Optional[List[str]] = None 
    form_type: FormType
    period_end: Optional[str | None] = None
    commission_number: Optional[str | None] = None
    parts: List["SECPart"] = []


class SECPart(BaseModel):
    title: str
    preface: str
    section: str
    page: Optional[int] = None
    items: List["SECItem"] = []

class SECItem(BaseModel):
    text: Optional[str] = None
    title: str
    page: Optional[int] = None
    subsection: str
    page: str
