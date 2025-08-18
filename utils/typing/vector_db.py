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
    document_id: Optional[str] = None  # Reference to SQLite document ID for text retrieval


    @property
    def pinecone_metadata(self) -> Dict:
        """ Meant for uploading this as metadata, drops all the None / Null Values"""
        # exclude embeddings/id and drop None
        meta = self.model_dump(exclude={'id', 'embeddings', 'text'}, exclude_none=True)
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
    

class PineconeResponse(VectorObject):
    """Response object from Pinecone queries with human-readable conversion"""
    embeddings: Optional[List[float]] = None
    score: Optional[float] = None

    @classmethod
    def from_pinecone_match(cls, match_dict: Dict) -> 'PineconeResponse':
        """Create PineconeResponse from Pinecone query match result"""
        metadata = match_dict.get('metadata', {})
        
        return cls(
            id=match_dict.get('id', ''),
            score=match_dict.get('score'),
            embeddings=match_dict.get('values'),
            ticker=metadata.get('ticker', ''),
            date=metadata.get('date', ''),
            year=metadata.get('year'),
            form_type=metadata.get('form_type', ''),
            text=metadata.get('text', ''),
            page_number=metadata.get('page_number', 0),
            document_path=metadata.get('document_path', ''),
            entity_type=EntityType(metadata.get('entity_type', 'UNKNOWN')),
            title=metadata.get('title'),
            section=metadata.get('section'),
            subsection=metadata.get('subsection'),
            company=metadata.get('company'),
            prev_chunk_id=metadata.get('prev_chunk_id'),
            next_chunk_id=metadata.get('next_chunk_id'),
            commission_number=metadata.get('commission_number'),
            period_end=metadata.get('period_end')
        )

    @property
    def llm_readable(self) -> str:
        """Convert to human-readable format for LLM consumption"""
        readable_parts = []
        
        # Document identification
        readable_parts.append(f"Document: {self.form_type} filing for {self.ticker}")
        if self.company:
            readable_parts.append(f"Company: {self.company}")
        
        # Date and period information
        readable_parts.append(f"Date: {self.date}")
        if self.period_end:
            readable_parts.append(f"Period End: {self.period_end}")
        
        # Document structure
        if self.title:
            readable_parts.append(f"Title: {self.title}")
        if self.section:
            readable_parts.append(f"Section: {self.section}")
        if self.subsection:
            readable_parts.append(f"Subsection: {self.subsection}")
        
        readable_parts.append(f"Page: {self.page_number}")
        
        # Similarity score if available
        if self.score is not None:
            readable_parts.append(f"Relevance Score: {self.score:.4f}")
        
        # Main content
        readable_parts.append(f"\nContent:\n{self.text}")
        
        # Navigation information
        if self.prev_chunk_id or self.next_chunk_id:
            nav_info = []
            if self.prev_chunk_id:
                nav_info.append(f"Previous: {self.prev_chunk_id}")
            if self.next_chunk_id:
                nav_info.append(f"Next: {self.next_chunk_id}")
            readable_parts.append(f"Navigation: {' | '.join(nav_info)}")
        
        return "\n".join(readable_parts)

    def to_text(self) -> str:
        """Alias for llm_readable for backward compatibility"""
        return self.llm_readable