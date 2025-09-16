from typing import Optional, List, Union, Literal


from pydantic import BaseModel, Field


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
    api_key: Optional[str] = Field(None, description="API Key for associated model")

    evaluations: bool = Field(True, description="If evaluations are being used for this...")
    metrics: List[Literal["precision", "recall", "hit-rate", "mrr", "ndcg"]] = Field(
        default_factory=lambda: ["ndcg", "precision", "recall"],
        description="The metrics we care for evaluation."
    )    
    # MRR => Mean Recipricol Rank
    # NDCG => Normalized Discounted Cumaltive gain
    test: Optional[TestConfig] = Field(None, description="Specific Test cases we care about...")
