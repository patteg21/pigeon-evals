from typing import Optional, List, Dict, Any, Union, Literal
from pathlib import Path
from uuid import uuid4
import yaml

from pydantic import BaseModel, Field, field_validator


class DimensionReduction(BaseModel):
    type: str = Field(..., description="Type of dimension reduction (PCA, UMAP, T-SNE)")
    dims: int = Field(..., description="Target dimensions")


class Embedding(BaseModel):
    provider: str = Field(..., description="Embedding provider (openai, huggingface, etc.)")
    model: str = Field(..., description="Model name")
    pooling_strategy: str = Field(default="mean", description="Pooling strategy: mean, max, weighted, smooth_decay")
    dimension_reduction: Optional[DimensionReduction] = None
    use_threading: bool = Field(default=True, description="Whether to use threading")
    max_workers: int = Field(default=8, description="Maximum number of worker threads")


class VectorConfig(BaseModel):
    upload: bool = Field(default=False, description="Whether to upload vectors")
    clear: bool = Field(default=False, description="Whether to clear existing vectors")
    index: Optional[str] = Field(None, description="Index name for vector storage")
    index_name: Optional[str] = Field(None, description="Alternative index name field")


class Storage(BaseModel):
    text_store: Optional[str] = Field(None, description="Text storage backend")
    sqlite_path: Optional[str] = Field(None, description="Path to SQLite database")
    vector: Optional[VectorConfig] = Field(None, description="Vector storage configuration")
    vector_db: Optional[Dict[str, Any]] = Field(None, description="Vector database configuration")
    outputs: List[Literal["chunks", "documents"]] = Field(default_factory=list, description="Output types to store")


class Generator(BaseModel):
    provider: str = Field(..., description="Generator provider")
    model: str = Field(..., description="Generator model name")


class Calibration(BaseModel):
    gold_fraction: float = Field(..., description="Fraction of gold standard examples")


class Judge(BaseModel):
    type: str = Field(..., description="Judge type (llm, rule-based, etc.)")
    prompt: str = Field(..., description="Judge prompt")
    calibration: Optional[Calibration] = None


class Retrieval(BaseModel):
    type: str = Field(..., description="Retrieval type (cosine, etc.)")
    top_k: int = Field(..., description="Number of top results to retrieve")


class Config(BaseModel):
    run_id: str = Field(uuid4().hex, description="Task name")
    task: str = Field(..., description="Task name")
    dataset_path: str = Field(..., description="Path to dataset")
    sec_metadata: List[str] = Field(default_factory=list, description="SEC metadata fields to extract")
    processors: List[str] = Field(default_factory=list, description="List of processors to use")
    embedding: Optional[Embedding] = Field(None, description="Embedding configuration")
    storage: Optional[Storage] = Field(None, description="Storage configuration")
    generator: Optional[Generator] = Field(None, description="Generator configuration")
    judge: Optional[Judge] = Field(None, description="Judge configuration")
    retrival: Optional[Retrieval] = Field(None, description="Retrieval configuration")
    
    @classmethod
    def from_yaml(cls, file_path: Union[str, Path]) -> 'Config':
        """Load configuration from YAML file"""
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(
            **data
            )
    
    def to_yaml(self, file_path: Union[str, Path]) -> None:
        """Save configuration to YAML file"""
        with open(file_path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)
    
    @field_validator('dataset_path')
    @classmethod
    def validate_dataset_path(cls, v):
        """Ensure dataset path exists or is valid"""
        if not Path(v).exists() and not v.startswith(('http://', 'https://', 's3://')):
            raise ValueError(f"Path error {v}")
        return v