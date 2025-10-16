from typing import Optional, List, Union, Literal, Dict
import os

from pydantic import BaseModel, Field


# === Test Cases


class MCPStdioConfig(BaseModel):
    """Configuration for local MCP servers using stdio (similar to Claude Code format)"""
    type: Literal["stdio"] = Field("stdio", description="MCP server type")
    command: str = Field(..., description="Command to run the MCP server")
    args: Optional[List[str]] = Field(default_factory=list, description="Arguments for the command")
    env: Optional[Dict[str, str]] = Field(default_factory=dict, description="Environment variables")
    cwd: Optional[str] = Field(None, description="Working directory for the command")


class MCPSseConfig(BaseModel):
    """Configuration for remote MCP servers using SSE"""
    type: Literal["sse"] = Field("sse", description="MCP server type")
    url: str = Field(..., description="URL of the remote MCP server")
    headers: Optional[Dict[str, str]] = Field(default_factory=dict, description="HTTP headers for authentication")
    timeout: Optional[float] = Field(30.0, description="Request timeout in seconds")
    sse_read_timeout: Optional[float] = Field(300.0, description="SSE read timeout in seconds")


# Union type for MCP config
MCPConfig = Union[MCPStdioConfig, MCPSseConfig]


class AgentTest(BaseModel):
    type: str = Field("agent", description="Type discriminator for this test")
    name: str = Field(..., description="Name of this Test Case")
    query: str = Field(..., description="Query for the Vector Database")
    prompt: Optional[str] = Field(None, description="Prompt for the agent to execute")
    mcp: MCPConfig = Field(..., description="MCP server configuration (stdio or sse)")

    # Execution configuration
    timeout: Optional[int] = Field(60, description="Timeout in seconds for agent execution")
    max_turns: Optional[int] = Field(10, description="Maximum conversation turns for the agent")

    # Agent configuration
    agent_model: Optional[str] = Field(None, description="Model to use for this specific agent test (overrides global)")
    agent_instructions: Optional[str] = Field(None, description="Custom instructions for the agent")

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
    load_test: Optional[str] = Field("data/tests/default.json", description="A path to a JSON will defined custom test cases for faster iteration.")
    tests:  List[Union[LLMTest, AgentTest, HumanTest]] = Field([], description="Specific Test cases we care about...")


class LLMConfig(BaseModel):
    provider: str = Field(default="openai", description="Generator provider")
    model: str = Field(default="gpt-4o-mini", description="Generator model name")
    api_key: Optional[str] = Field(default=None, description="API Key for associated model")




class EvaluationConfig(BaseModel):
    top_k: Optional[int] = Field(None, description="Number of top results to retrieve")
    rerank: Optional[RerankConfig] = Field(None, description="A Reranker on top of the Retrieval")

    llm: Optional[LLMConfig] = Field(None, description="Configuration for the LLM")

    evaluations: bool = Field(True, description="If evaluations are being used for this...")
    metrics: List[Literal["precision", "recall", "hit-rate", "mrr", "ndcg"]] = Field(
        default_factory=lambda: ["ndcg", "precision", "recall"],
        description="The metrics we care for evaluation."
    )    
    # MRR => Mean Recipricol Rank
    # NDCG => Normalized Discounted Cumaltive gain
    test: Optional[TestConfig] = Field(None, description="Specific Test cases we care about...")
