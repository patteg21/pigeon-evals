from typing import Optional, List, Dict, Any, Union, Literal
from pathlib import Path
from uuid import uuid4
import yaml

from pydantic import BaseModel, Field, field_validator

# === Emebedding Models

class DimensionReduction(BaseModel):
    type: str = Field(..., description="Type of dimension reduction (PCA, UMAP, T-SNE)")
    dims: int = Field(..., description="Target dimensions")


class EmbeddingConfig(BaseModel):
    provider: Literal["huggingface", "openai"] = Field(..., description="Embedding provider (openai, huggingface, etc.)")
    model: str = Field(..., description="Model name")
    pooling_strategy: str = Field(default="mean", description="Pooling strategy: mean, max, weighted, smooth_decay")
    dimension_reduction: Optional[DimensionReduction] = None
    use_threading: bool = Field(default=True, description="Whether to use threading")
    max_workers: int = Field(default=8, description="Maximum number of worker threads")

# === Vector DB

class VectorConfig(BaseModel):
    upload: bool = Field(default=False, description="Whether to upload vectors")
    clear: bool = Field(default=False, description="Whether to clear existing vectors")
    index: Optional[str] = Field(None, description="Index name for vector storage")
    index_name: Optional[str] = Field(None, description="Alternative index name field")


# === Text Store 

class TextStoreConfig(BaseModel):
    client: Optional[str] = Field(None, description="Path to SQLite database")
    path: Optional[str] = Field(None, description="Path to SQLite database")
    upload: bool = Field(default=False, description="Whether to upload / save text")



class StorageConfig(BaseModel):
    text_store: Optional[TextStoreConfig] = Field(None, description="Text storage backend")
    vector: Optional[VectorConfig] = Field(None, description="Vector storage configuration")
    vector_db: Optional[Dict[str, Any]] = Field(None, description="Vector database configuration")
    outputs: List[Literal["chunks", "documents"]] = Field(default_factory=list, description="Output types to store")


# === Test Cases


class MCPConfig(BaseModel):
    command: str
    args: List[str]


class AgentTest(BaseModel):
    type: str = Field("agent", description="Type discriminator for this test")
    name: str = Field(..., description="Name of this Test Case")
    query: str = Field(..., description="Query for the Vector Database")
    prompt: Optional[str] = Field(None, description="Prompt for LLM-based tests")
    mcp: MCPConfig = Field(..., description="Command to Test the MCP server")

class LLMTest(BaseModel):
    type: str = Field("llm", description="Type discriminator for this test")
    name: str = Field(..., description="Name of this Test Case")
    query: str = Field(..., description="Query for the Vector Database")
    prompt: Optional[str] = Field(None, description="Prompt for LLM-based tests")
    eval_type: List[Literal["pairwise", "single"]] = Field(["single"], description="If there is a LLM Eval, which methof") 
    # pairwise => select which of the two is better or equally good or bad

class HumanTest(BaseModel):
    type: str = Field("human", description="Type discriminator for this test")
    name: str = Field(..., description="Name of this Test Case")
    query: str = Field(..., description="Query for the Vector Database")


# === Evaluation Config

class RerankConfig(BaseModel):
    provider: Literal["huggingface", "openai"] = Field("huggingface", description="The model provider for reranking")
    model: Optional[str] = Field(..., description="Rerank model name")
    top_k: Optional[int] = Field(..., description="The Top Results returned from the reranker")


class TestConfig(BaseModel):
    default_test: Optional[str] = Field("data/tests/default.json", description="A path to a JSON will defined custom test cases for faster iteration.")
    tests:  List[Union[LLMTest, AgentTest, HumanTest]] = Field([], description="Specific Test cases we care about...")



class EvaluationConfig(BaseModel):
    top_k: Optional[int] = Field(None, description="Number of top results to retrieve")
    rerank: Optional[RerankConfig] = Field(None, description="A Reranker on top of the Retrieval")

    provider: str = Field(..., description="Generator provider")
    model: str = Field(..., description="Generator model name")

    evaluations: bool = Field(True, description="If evaluations are being used for this...")
    metrics: List[Literal["precision", "recall", "hit-rate", "mrr", "ndcg"]] = Field(
        default_factory=lambda: ["ndcg", "precision", "recall"],
        description="The metrics we care for evaluation."
    )    
    # MRR => Mean Recipricol Rank
    # NDCG => Normalized Discounted Cumaltive gain
    test: Optional[TestConfig] = Field(None, description="Specific Test cases we care about...")


# === General Config

class YamlConfig(BaseModel):
    run_id: str = Field(uuid4().hex, description="Task name")
    task: str = Field(..., description="Task name")
    
    # document processing
    preprocess: Optional[str] = Field(None, description="Preprocessing of Documents")
    parser: Optional[str] = Field(None, description="Document Parser configuration")
    embedding: Optional[EmbeddingConfig] = Field(None, description="Embedding configuration")
    
    # document saving
    storage: Optional[StorageConfig] = Field(None, description="Storage configuration")
    
    # testing 

    report: Optional[EvaluationConfig] = Field(None, description="Test Cases to report configuration")

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
    
    @field_validator('dataset_path')
    @classmethod
    def validate_dataset_path(cls, v):
        """Ensure dataset path exists or is valid"""
        if not Path(v).exists() and not v.startswith(('http://', 'https://', 's3://')):
            raise ValueError(f"Path error {v}")
        return v