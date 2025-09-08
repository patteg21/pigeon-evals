from .base import LLMBaseClient
from typing import Dict, Any, Optional
from utils.logger import logger
import boto3
import json
import os

class BedrockLLM(LLMBaseClient):
    """AWS Bedrock LLM client."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model = self.config.get("model", "anthropic.claude-3-haiku-20240307-v1:0")
        self.region = self.config.get("region", "us-east-1")
        
        # AWS credentials can come from config, environment, or IAM role
        session_kwargs = {}
        if self.config.get("aws_access_key_id"):
            session_kwargs["aws_access_key_id"] = self.config["aws_access_key_id"]
        if self.config.get("aws_secret_access_key"):
            session_kwargs["aws_secret_access_key"] = self.config["aws_secret_access_key"]
        if self.config.get("aws_session_token"):
            session_kwargs["aws_session_token"] = self.config["aws_session_token"]
        
        self.session = boto3.Session(**session_kwargs)
        self.client = self.session.client("bedrock-runtime", region_name=self.region)
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