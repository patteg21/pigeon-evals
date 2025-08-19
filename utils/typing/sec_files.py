from typing import Optional
from datetime import datetime

from pydantic import BaseModel, field_validator

from utils.typing.common import (
    FormType
)

class SECDocument(BaseModel):
    ticker: str
    company: Optional[str] = None
    year: Optional[str] = None
    date: str
    text: str
    path: str
    form_type: FormType
    sec_data: Optional[dict] = None

    @field_validator("year", mode="before")
    def extract_year(cls, v, values):
        if v is None and "date" in values:
            try:
                return datetime.strptime(values["date"], "%Y-%m-%d").year
            except ValueError:
                return None
        return v


class SECTable(BaseModel):
    id: str
    page_number: Optional[int] = None
    text: str
