from typing import Optional

from pydantic import BaseModel, Field



# === Vector DB Config

class VectorConfig(BaseModel):
    clear: bool = Field(default=False, description="Whether to clear existing vectors")
    index: Optional[str] = Field(None, description="Index name for vector storage")
    index_name: Optional[str] = Field(None, description="Alternative index name field")
    
    upload: bool = Field(default=False, description="Whether to upload vectors")


# === Text Store 

class TextStoreConfig(BaseModel):
    client: Optional[str] = Field(None, description="Path to SQLite database")
    path: Optional[str] = Field(None, description="Path to SQLite database")
    upload: bool = Field(default=False, description="Whether to upload / save text")



class StorageConfig(BaseModel):
    text_store: Optional[TextStoreConfig] = Field(None, description="Text storage backend")
    vector: Optional[VectorConfig] = Field(None, description="Vector storage configuration")

