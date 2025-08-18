from typing import Optional, List, Dict
import re
from datetime import datetime

from pydantic import BaseModel, field_validator

from utils.typing.common import EntityType

class VectorObject(BaseModel):
    id: str
    ticker: str
    date: str
    year: Optional[int] = None # Year is autofilled based on Date
    form_type: str
    text: str
    page_number: int
    document_path: str
    entity_type: EntityType 
    embeddings: List[float]

    title: Optional[str] = None
    section: Optional[str] = None
    subsection: Optional[str] = None
    company: Optional[str] = None
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    commission_number: Optional[str | None] = None
    period_end: Optional[str | None] = None


    @property
    def pinecone_metadata(self) -> Dict:
        """ Meant for uploading this as metadata, drops all the None / Null Values"""
        # exclude embeddings/id and drop None
        meta = self.model_dump(exclude={'id', 'embeddings'}, exclude_none=True)
        # Pinecone requires: str | number | bool | list[str]
        for k, v in list(meta.items()):
            if isinstance(v, (list, tuple, set)):
                meta[k] = [str(x) for x in v if x is not None]
            elif not isinstance(v, (str, int, float, bool)):
                meta[k] = str(v)
        return meta

 
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