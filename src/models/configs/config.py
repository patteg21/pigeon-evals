from typing import Optional, Union, Literal, List
from pathlib import Path
from uuid import uuid4
import yaml

from pydantic import BaseModel, Field, field_validator

from .embedding import EmbeddingConfig
from .eval import EvaluationConfig
from .parser import ParserConfig
from .storage import StorageConfig

# === Checkpoint Config

class CheckPoint(BaseModel):
    pass


# === Threading Config

class ThreadingConfig(BaseModel):
    max_workers: Optional[int] = Field(4, description="Worker threads implemented")



# === Dataset Config

class PreprocessConfig(BaseModel):
    ocr: Optional[Literal["easyocr", "tesseract"]] = Field(None, description="OCR for file types...") 
    vllm: Optional[bool] = Field(None, description="VLLM for processing...")


class DatasetConfig(BaseModel):
    provider: Optional[Literal['local', 's3']] = Field("local", description="File Path for the files...")
    path: str = Field(..., description="Path to dataset")

    allowed_types: Optional[List[str]] = ["txt"] # todo handle multiple files



# === General Config


class YamlConfig(BaseModel):
    run_id: str = Field(uuid4().hex, description="Task name")
    task: str = Field(..., description="Task name")

    
    # general
    dataset: Optional[DatasetConfig] = Field(None, description="Dataset and Configs")
    threading: Optional[ThreadingConfig] = Field(None, description="Threading number of workers")


    # document processing
    parser: Optional[ParserConfig] = Field(None, description="Document Parser configuration")
    embedding: Optional[EmbeddingConfig] = Field(None, description="Embedding configuration")
    
    # document saving
    storage: Optional[StorageConfig] = Field(None, description="Storage configuration")
    
    # testing 
    eval: Optional[EvaluationConfig] = Field(None, description="Test Cases to report configuration")

    @classmethod
    def from_yaml(cls, file_path: Union[str, Path]) -> 'YamlConfig':
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
    
