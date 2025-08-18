from typing import Optional, List
import re
from datetime import datetime

from pydantic import BaseModel, field_validator

from utils.typing.common import EntityType

class VectorObject(BaseModel):
    id: str
    ticker: str
    company: Optional[str] = None
    date: str
    year: Optional[int] = None # Year is autofilled based on Date
    form_type: str
    text: str
    page_number: int
    section: str
    title: str
    document_path: str
    entity_type: EntityType 
    subsection: Optional[str] = None
    embeddings: List[float]
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    commission_number: Optional[str | None] = None
    period_end: Optional[str | None] = None

    # TODO: Include / Inherit from the SECDocument Object so it can be unpacked directly

 
    @field_validator('date')
    @classmethod
    def validate_date(cls, v: str) -> str:
        """Validate date format is YYYY-MM-DD"""
        return cls._verify_date_format(v)

    @field_validator('year', mode='before')
    @classmethod
    def extract_year_from_date(cls, v, info):
        """Extract year from date field if year is not provided"""
        if v is not None:
            return v
        
        # Get the date field value
        if 'date' in info.data:
            date_str = info.data['date']
            # Validate the date format first
            validated_date = cls._verify_date_format(date_str)
            # Extract year from the validated date
            return int(validated_date.split('-')[0])
        
        raise ValueError("Year could not be extracted from date")

    @staticmethod
    def _verify_date_format(date: str):
        """YYYY-MM-DD"""
        pattern = r'^\d{4}-\d{2}-\d{2}'
        if not re.match(pattern, date):
            raise ValueError(f"Date must be in YYYY-MM-DD format, got: {date}")
        
        try:
            datetime.strptime(date, '%Y-%m-%d')
        except ValueError as e:
            raise ValueError(f"Invalid date: {date}. {str(e)}")
        
        return date