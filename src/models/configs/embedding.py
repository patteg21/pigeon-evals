from typing import Optional, Literal


from pydantic import BaseModel, Field


# === Emebedding Config

class DimensionReduction(BaseModel):
    type: str = Field(..., description="Type of dimension reduction (PCA, UMAP, T-SNE)")
    dims: int = Field(..., description="Target dimensions")
    seed: int = Field(42, description="Default Seed")
    path: Optional[str] = Field(None, description='Path to pretrained Dimensitonal Reduction model')

class EmbeddingConfig(BaseModel):
    provider: Literal["huggingface", "openai"] = Field(..., description="Embedding provider (openai, huggingface, etc.)")
    model: str = Field(..., description="Model name")
    batch_size: int = Field(default=128, description="Batch size for embedding chunks at a time")
    pooling_strategy: str = Field(default="mean", description="Pooling strategy: mean, max, weighted, smooth_decay")
    dimension_reduction: Optional[DimensionReduction] = None
    use_threading: bool = Field(default=True, description="Whether to use threading")
