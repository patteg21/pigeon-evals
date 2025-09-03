from typing import Optional, List, Any
from datetime import datetime

from pydantic import BaseModel, model_validator

from evals.src.utils.types.common import (
    FormType
)

class SECMetadata(BaseModel):
    """Comprehensive metadata extracted from SEC documents."""
    company_name: Optional[str] = None
    period_end: Optional[str] = None
    commission_number: Optional[str] = None
    state_of_incorporation: Optional[str] = None
    ein: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    phone: Optional[str] = None
    market_value: Optional[str] = None
    shares_outstanding: Optional[str] = None
    shares_outstanding_date: Optional[str] = None
    filer_status: Optional[str] = None

class SECDocument(BaseModel):
    ticker: str
    company: Optional[str] = None
    year: Optional[str] = None
    date: str
    text: str
    path: str
    form_type: FormType
    sec_data: Optional[dict] = None
    sec_metadata: Optional[SECMetadata] = None
    toc: Optional['SECTable'] = None
    parts: List[Any] = []

    @model_validator(mode="after")
    def set_year(self):
        if self.date:
            try:
                self.year = datetime.strptime(self.date, "%Y-%m-%d").year
            except ValueError:
                self.year = None
        return self


class SECTable(BaseModel):
    id: str
    page_number: Optional[int] = None
    text: str
