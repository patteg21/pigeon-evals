from typing import Optional
from pydantic import BaseModel, Field


class RegexParser(BaseModel):
    pass


class ParserConfig(BaseModel):
    todo: Optional[str] = Field(None, description="Not implemented")