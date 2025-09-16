from .base import LLMBaseClient
from typing import Optional
from utils.logger import logger
import boto3
import json
import os
from models.configs.eval import EvaluationConfig

class BedrockLLM(LLMBaseClient):
    """AWS Bedrock LLM client."""

    def __init__(self, config: EvaluationConfig):
        super().__init__(config)
        self.model = self.config.model or "anthropic.claude-3-haiku-20240307-v1:0"

        # For Bedrock, we'll use environment variables or IAM roles for AWS credentials
        # since EvaluationConfig doesn't have AWS-specific fields
        session_kwargs = {}
        if os.getenv("AWS_ACCESS_KEY_ID"):
            session_kwargs["aws_access_key_id"] = os.getenv("AWS_ACCESS_KEY_ID")
        if os.getenv("AWS_SECRET_ACCESS_KEY"):
            session_kwargs["aws_secret_access_key"] = os.getenv("AWS_SECRET_ACCESS_KEY")
        if os.getenv("AWS_SESSION_TOKEN"):
            session_kwargs["aws_session_token"] = os.getenv("AWS_SESSION_TOKEN")
        
        self.session = boto3.Session(**session_kwargs)
        self.client = self.session.client("bedrock-runtime", region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
        logger.info(f"Initializing Bedrock LLM with model: {self.model}")
    
    @property
    def provider_name(self) -> str:
        return "bedrock"
    
    def invoke(self, prompt: Optional[str] = None, query: Optional[str] = None, **kwargs) -> str:
        """Invoke Bedrock with a prompt or query."""
        if prompt and query:
            raise ValueError("Cannot specify both prompt and query")
        if not prompt and not query:
            raise ValueError("Must specify either prompt or query")
        
        text = prompt or query
        
        try:
            # Different models have different request formats
            if "anthropic.claude" in self.model:
                body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": kwargs.get("max_tokens", 1000),
                    "temperature": kwargs.get("temperature", 0.7),
                    "messages": [
                        {"role": "user", "content": text}
                    ]
                }
            elif "amazon.titan" in self.model:
                body = {
                    "inputText": text,
                    "textGenerationConfig": {
                        "maxTokenCount": kwargs.get("max_tokens", 1000),
                        "temperature": kwargs.get("temperature", 0.7)
                    }
                }
            elif "ai21.j2" in self.model:
                body = {
                    "prompt": text,
                    "maxTokens": kwargs.get("max_tokens", 1000),
                    "temperature": kwargs.get("temperature", 0.7)
                }
            elif "cohere.command" in self.model:
                body = {
                    "prompt": text,
                    "max_tokens": kwargs.get("max_tokens", 1000),
                    "temperature": kwargs.get("temperature", 0.7)
                }
            else:
                # Default to Claude format
                body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": kwargs.get("max_tokens", 1000),
                    "temperature": kwargs.get("temperature", 0.7),
                    "messages": [
                        {"role": "user", "content": text}
                    ]
                }
            
            response = self.client.invoke_model(
                modelId=self.model,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json"
            )
            
            response_body = json.loads(response["body"].read())
            
            # Extract result based on model type
            if "anthropic.claude" in self.model:
                result = response_body["content"][0]["text"]
            elif "amazon.titan" in self.model:
                result = response_body["results"][0]["outputText"]
            elif "ai21.j2" in self.model:
                result = response_body["completions"][0]["data"]["text"]
            elif "cohere.command" in self.model:
                result = response_body["generations"][0]["text"]
            else:
                # Default to Claude format
                result = response_body["content"][0]["text"]
            
            logger.info(f"Bedrock invoke successful, response length: {len(result)}")
            return result
            
        except Exception as e:
            logger.error(f"Bedrock API call failed: {str(e)}")
            raise