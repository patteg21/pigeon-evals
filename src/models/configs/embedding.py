from typing import Optional, Literal


from pydantic import BaseModel, Field


# === Emebedding Config

class DimensionReduction(BaseModel):
    type: str = Field(..., description="Type of dimension reduction (PCA, UMAP, T-SNE)")
    dims: int = Field(..., description="Target dimensions")


class EmbeddingConfig(BaseModel):
    provider: Literal["huggingface", "openai"] = Field(..., description="Embedding provider (openai, huggingface, etc.)")
    model: str = Field(..., description="Model name")
    pooling_strategy: str = Field(default="mean", description="Pooling strategy: mean, max, weighted, smooth_decay")
    dimension_reduction: Optional[DimensionReduction] = None
    use_threading: bool = Field(default=True, description="Whether to use threading")
