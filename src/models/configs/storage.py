from typing import Optional, Literal

from pydantic import BaseModel, Field



# === Vector DB Config

class VectorConfig(BaseModel):
    provider: Optional[str] = Field(default="faiss", description="Vector storage provider")
    path:  Optional[str] = Field(None, description="Vector storage provider")
    clear: bool = Field(default=False, description="Whether to clear existing vectors")
    index: Optional[str] = Field(None, description="Index name for vector storage")
    index_name: Optional[str] = Field(None, description="Alternative index name field")
    dimension: Optional[int] = Field(default=768, description="Vector dimension size")

    upload: bool = Field(default=False, description="Whether to upload vectors")


# === Text Store

class TextStoreConfig(BaseModel):
    client: Optional[Literal["sqlite", "postgres", "s3", "file"]] = Field(default="sqlite", description="Storage client type")
    upload: bool = Field(default=False, description="Whether to upload / save text")

    # SQLite specific
    path: Optional[str] = Field(default="data/.sql/chunks.db", description="Path to SQLite database file")

    # PostgreSQL specific
    host: Optional[str] = Field(default="localhost", description="PostgreSQL host")
    port: Optional[int] = Field(default=5432, description="PostgreSQL port")
    database: Optional[str] = Field(default="pigeon_evals", description="Database name")
    user: Optional[str] = Field(default="postgres", description="Database user")
    password: Optional[str] = Field(default="", description="Database password")

    # S3 specific
    bucket_name: Optional[str] = Field(default="pigeon-evals-documents", description="S3 bucket name")
    prefix: Optional[str] = Field(default="documents/", description="S3 object prefix")
    access_key_id: Optional[str] = Field(None, description="AWS access key ID")
    secret_access_key: Optional[str] = Field(None, description="AWS secret access key")
    region: Optional[str] = Field(default="us-east-1", description="AWS region")

    # File store specific
    base_path: Optional[str] = Field(default="data/documents", description="Base path for file storage")


class SqliteConfig(BaseModel):
    path: str = Field(default="data/.sql/chunks.db", description="Path to SQLite database file")


class PostgresConfig(BaseModel):
    host: str = Field(default="localhost", description="PostgreSQL host")
    port: int = Field(default=5432, description="PostgreSQL port")
    database: str = Field(default="pigeon_evals", description="Database name")
    user: str = Field(default="postgres", description="Database user")
    password: str = Field(default="", description="Database password")


class S3Config(BaseModel):
    bucket_name: str = Field(default="pigeon-evals-documents", description="S3 bucket name")
    prefix: str = Field(default="documents/", description="S3 object prefix")
    access_key_id: Optional[str] = Field(None, description="AWS access key ID")
    secret_access_key: Optional[str] = Field(None, description="AWS secret access key")
    region: str = Field(default="us-east-1", description="AWS region")


class FileStoreConfig(BaseModel):
    base_path: str = Field(default="data/documents", description="Base path for file storage")



class StorageConfig(BaseModel):
    text_store: Optional[TextStoreConfig] = Field(None, description="Text storage backend")
    vector: Optional[VectorConfig] = Field(None, description="Vector storage configuration")

