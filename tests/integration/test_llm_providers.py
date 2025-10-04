import pytest
import os
from unittest.mock import patch, MagicMock

from src.infra.llm.factory import LLMFactory
from src.infra.llm.openai import OpenAILLM
from src.infra.llm.anthropic import AnthropicLLM
from src.infra.llm.gemini import GeminiLLM
from src.infra.llm.bedrock import BedrockLLM


class TestOpenAILLM:
    """Integration tests for OpenAI LLM provider."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.test_config = {
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "max_tokens": 100
        }
        self.test_prompt = "What is the capital of France? Answer in one sentence."

    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not available")
    def test_openai_llm_invoke_with_prompt(self):
        """Test OpenAI LLM invocation with prompt."""
        llm = OpenAILLM(self.test_config)
        
        # Test invocation
        response = llm.invoke(prompt=self.test_prompt)
        
        # Verify response
        assert isinstance(response, str)
        assert len(response) > 0
        assert "Paris" in response or "paris" in response.lower()

    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not available")
    def test_openai_llm_invoke_with_query(self):
        """Test OpenAI LLM invocation with query parameter."""
        llm = OpenAILLM(self.test_config)
        
        # Test invocation with query instead of prompt
        response = llm.invoke(query="Name three programming languages.")
        
        # Verify response
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not available")
    def test_openai_llm_custom_parameters(self):
        """Test OpenAI LLM with custom parameters."""
        llm = OpenAILLM(self.test_config)
        
        # Test with custom temperature and max_tokens
        response = llm.invoke(
            prompt="Write a short poem about AI.",
            temperature=0.9,
            max_tokens=50
        )
        
        # Verify response
        assert isinstance(response, str)
        assert len(response) > 0

    def test_openai_llm_invalid_parameters(self):
        """Test OpenAI LLM with invalid parameters."""
        llm = OpenAILLM(self.test_config)
        
        # Test with both prompt and query (should raise error)
        with pytest.raises(ValueError, match="Cannot specify both prompt and query"):
            llm.invoke(prompt="Test prompt", query="Test query")
        
        # Test with neither prompt nor query (should raise error)
        with pytest.raises(ValueError, match="Must specify either prompt or query"):
            llm.invoke()

    def test_openai_llm_invalid_api_key(self):
        """Test OpenAI LLM with invalid API key."""
        config = {
            "model": "gpt-4o-mini",
            "api_key": "invalid_key"
        }
        
        llm = OpenAILLM(config)
        
        # Should raise an exception for invalid API key
        with pytest.raises(Exception):
            llm.invoke(prompt="Test prompt")

    def test_openai_llm_provider_name(self):
        """Test OpenAI LLM provider name."""
        llm = OpenAILLM(self.test_config)
        assert llm.provider_name == "openai"


class TestAnthropicLLM:
    """Integration tests for Anthropic LLM provider."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.test_config = {
            "model": "claude-3-haiku-20240307",
            "temperature": 0.7,
            "max_tokens": 100
        }
        self.test_prompt = "What is the capital of France? Answer in one sentence."

    @pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="Anthropic API key not available")
    def test_anthropic_llm_invoke_with_prompt(self):
        """Test Anthropic LLM invocation with prompt."""
        llm = AnthropicLLM(self.test_config)
        
        # Test invocation
        response = llm.invoke(prompt=self.test_prompt)
        
        # Verify response
        assert isinstance(response, str)
        assert len(response) > 0
        assert "Paris" in response or "paris" in response.lower()

    @pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="Anthropic API key not available")
    def test_anthropic_llm_invoke_with_query(self):
        """Test Anthropic LLM invocation with query parameter."""
        llm = AnthropicLLM(self.test_config)
        
        # Test invocation with query instead of prompt
        response = llm.invoke(query="Name three programming languages.")
        
        # Verify response
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="Anthropic API key not available")
    def test_anthropic_llm_custom_parameters(self):
        """Test Anthropic LLM with custom parameters."""
        llm = AnthropicLLM(self.test_config)
        
        # Test with custom temperature and max_tokens
        response = llm.invoke(
            prompt="Write a short poem about AI.",
            temperature=0.9,
            max_tokens=50
        )
        
        # Verify response
        assert isinstance(response, str)
        assert len(response) > 0

    def test_anthropic_llm_invalid_parameters(self):
        """Test Anthropic LLM with invalid parameters."""
        llm = AnthropicLLM(self.test_config)
        
        # Test with both prompt and query (should raise error)
        with pytest.raises(ValueError, match="Cannot specify both prompt and query"):
            llm.invoke(prompt="Test prompt", query="Test query")
        
        # Test with neither prompt nor query (should raise error)
        with pytest.raises(ValueError, match="Must specify either prompt or query"):
            llm.invoke()

    def test_anthropic_llm_invalid_api_key(self):
        """Test Anthropic LLM with invalid API key."""
        config = {
            "model": "claude-3-haiku-20240307",
            "api_key": "invalid_key"
        }
        
        llm = AnthropicLLM(config)
        
        # Should raise an exception for invalid API key
        with pytest.raises(Exception):
            llm.invoke(prompt="Test prompt")

    def test_anthropic_llm_provider_name(self):
        """Test Anthropic LLM provider name."""
        llm = AnthropicLLM(self.test_config)
        assert llm.provider_name == "anthropic"


class TestGeminiLLM:
    """Integration tests for Gemini LLM provider."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.test_config = {
            "model": "gemini-pro",
            "temperature": 0.7,
            "max_tokens": 100
        }
        self.test_prompt = "What is the capital of France? Answer in one sentence."

    @pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="Google API key not available")
    def test_gemini_llm_invoke_with_prompt(self):
        """Test Gemini LLM invocation with prompt."""
        llm = GeminiLLM(self.test_config)
        
        # Test invocation
        response = llm.invoke(prompt=self.test_prompt)
        
        # Verify response
        assert isinstance(response, str)
        assert len(response) > 0
        assert "Paris" in response or "paris" in response.lower()

    @pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="Google API key not available")
    def test_gemini_llm_invoke_with_query(self):
        """Test Gemini LLM invocation with query parameter."""
        llm = GeminiLLM(self.test_config)
        
        # Test invocation with query instead of prompt
        response = llm.invoke(query="Name three programming languages.")
        
        # Verify response
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="Google API key not available")
    def test_gemini_llm_custom_parameters(self):
        """Test Gemini LLM with custom parameters."""
        llm = GeminiLLM(self.test_config)
        
        # Test with custom temperature and max_tokens
        response = llm.invoke(
            prompt="Write a short poem about AI.",
            temperature=0.9,
            max_tokens=50
        )
        
        # Verify response
        assert isinstance(response, str)
        assert len(response) > 0

    def test_gemini_llm_invalid_parameters(self):
        """Test Gemini LLM with invalid parameters."""
        llm = GeminiLLM(self.test_config)
        
        # Test with both prompt and query (should raise error)
        with pytest.raises(ValueError, match="Cannot specify both prompt and query"):
            llm.invoke(prompt="Test prompt", query="Test query")
        
        # Test with neither prompt nor query (should raise error)
        with pytest.raises(ValueError, match="Must specify either prompt or query"):
            llm.invoke()

    def test_gemini_llm_invalid_api_key(self):
        """Test Gemini LLM with invalid API key."""
        config = {
            "model": "gemini-pro",
            "api_key": "invalid_key"
        }
        
        llm = GeminiLLM(config)
        
        # Should raise an exception for invalid API key
        with pytest.raises(Exception):
            llm.invoke(prompt="Test prompt")

    def test_gemini_llm_provider_name(self):
        """Test Gemini LLM provider name."""
        llm = GeminiLLM(self.test_config)
        assert llm.provider_name == "gemini"


class TestBedrockLLM:
    """Integration tests for AWS Bedrock LLM provider."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.test_config = {
            "model": "anthropic.claude-3-haiku-20240307-v1:0",
            "region": "us-east-1",
            "temperature": 0.7,
            "max_tokens": 100
        }
        self.test_prompt = "What is the capital of France? Answer in one sentence."

    @pytest.mark.skipif(
        not (os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY")),
        reason="AWS credentials not available"
    )
    def test_bedrock_llm_invoke_with_prompt_claude(self):
        """Test Bedrock LLM invocation with Claude model."""
        llm = BedrockLLM(self.test_config)
        
        # Test invocation
        response = llm.invoke(prompt=self.test_prompt)
        
        # Verify response
        assert isinstance(response, str)
        assert len(response) > 0
        assert "Paris" in response or "paris" in response.lower()

    @pytest.mark.skipif(
        not (os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY")),
        reason="AWS credentials not available"
    )
    def test_bedrock_llm_invoke_with_titan_model(self):
        """Test Bedrock LLM with Amazon Titan model."""
        config = self.test_config.copy()
        config["model"] = "amazon.titan-text-express-v1"
        
        llm = BedrockLLM(config)
        
        # Test invocation
        response = llm.invoke(prompt=self.test_prompt)
        
        # Verify response
        assert isinstance(response, str)
        assert len(response) > 0

    def test_bedrock_llm_with_mocked_client(self):
        """Test Bedrock LLM with mocked AWS client."""
        llm = BedrockLLM(self.test_config)
        
        # Mock the Bedrock response
        mock_response = {
            'body': MagicMock(),
            'contentType': 'application/json',
            'status': 200
        }
        
        # Mock the response body
        mock_body_content = {
            "content": [{"text": "Paris is the capital of France."}]
        }
        mock_response['body'].read.return_value = str(mock_body_content).encode()
        
        # Patch the invoke_model method
        with patch.object(llm.client, 'invoke_model', return_value=mock_response):
            with patch('json.loads', return_value=mock_body_content):
                response = llm.invoke(prompt=self.test_prompt)
                assert isinstance(response, str)
                assert "Paris" in response

    def test_bedrock_llm_invalid_parameters(self):
        """Test Bedrock LLM with invalid parameters."""
        llm = BedrockLLM(self.test_config)
        
        # Test with both prompt and query (should raise error)
        with pytest.raises(ValueError, match="Cannot specify both prompt and query"):
            llm.invoke(prompt="Test prompt", query="Test query")
        
        # Test with neither prompt nor query (should raise error)
        with pytest.raises(ValueError, match="Must specify either prompt or query"):
            llm.invoke()

    def test_bedrock_llm_provider_name(self):
        """Test Bedrock LLM provider name."""
        llm = BedrockLLM(self.test_config)
        assert llm.provider_name == "bedrock"


# TODO: UPDATE THESE
class TestLLMFactory:
    """Integration tests for LLMFactory."""
    
    def test_create_openai_llm(self):
        """Test creating OpenAI LLM through factory."""
        config = {
            "model": "gpt-4o-mini"
        }
        
        llm = LLMFactory.create("openai", config)
        
        # Verify LLM type
        assert isinstance(llm, OpenAILLM)
        assert llm.provider_name == "openai"

    def test_create_anthropic_llm(self):
        """Test creating Anthropic LLM through factory."""
        config = {
            "model": "claude-3-haiku-20240307"
        }
        
        llm = LLMFactory.create("anthropic", config)
        
        # Verify LLM type
        assert isinstance(llm, AnthropicLLM)
        assert llm.provider_name == "anthropic"

    def test_create_gemini_llm(self):
        """Test creating Gemini LLM through factory."""
        config = {
            "model": "gemini-pro"
        }
        
        llm = LLMFactory.create("gemini", config)
        
        # Verify LLM type
        assert isinstance(llm, GeminiLLM)
        assert llm.provider_name == "gemini"

    def test_create_bedrock_llm(self):
        """Test creating Bedrock LLM through factory."""
        config = {
            "model": "anthropic.claude-3-haiku-20240307-v1:0",
            "region": "us-east-1"
        }
        
        llm = LLMFactory.create("bedrock", config)
        
        # Verify LLM type
        assert isinstance(llm, BedrockLLM)
        assert llm.provider_name == "bedrock"

    def test_create_unknown_provider_fallback(self):
        """Test creating LLM with unknown provider (should fallback to OpenAI)."""
        config = {
            "model": "gpt-4o-mini"
        }
        
        llm = LLMFactory.create("unknown_provider", config)
        
        # Should fallback to OpenAI
        assert isinstance(llm, OpenAILLM)
        assert llm.provider_name == "openai"

    @patch('pathlib.Path.exists')
    def test_create_from_config_file_not_found(self, mock_exists):
        """Test creating LLM when config file doesn't exist."""
        mock_exists.return_value = False
        
        llm = LLMFactory.create_from_config()
        
        # Should create default OpenAI LLM
        assert isinstance(llm, OpenAILLM)
        assert llm.provider_name == "openai"


class TestLLMProviderComparison:
    """Integration tests comparing different LLM providers."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.test_prompt = "What is 2 + 2? Answer with just the number."

    @pytest.mark.skipif(
        not (os.getenv("OPENAI_API_KEY") and os.getenv("ANTHROPIC_API_KEY")),
        reason="Both OpenAI and Anthropic API keys required"
    )
    def test_llm_consistency_across_providers(self):
        """Test that different LLM providers can handle the same simple prompt."""
        # Create LLMs
        openai_config = {"model": "gpt-4o-mini", "temperature": 0.1}
        anthropic_config = {"model": "claude-3-haiku-20240307", "temperature": 0.1}
        
        openai_llm = OpenAILLM(openai_config)
        anthropic_llm = AnthropicLLM(anthropic_config)
        
        # Get responses from both providers
        openai_response = openai_llm.invoke(prompt=self.test_prompt)
        anthropic_response = anthropic_llm.invoke(prompt=self.test_prompt)
        
        # Both should produce valid responses
        assert isinstance(openai_response, str)
        assert isinstance(anthropic_response, str)
        assert len(openai_response.strip()) > 0
        assert len(anthropic_response.strip()) > 0
        
        # Both should contain "4" in the response for this simple math question
        assert "4" in openai_response
        assert "4" in anthropic_response

    def test_llm_error_handling(self):
        """Test that all LLM providers handle errors consistently."""
        providers_configs = [
            ("openai", {"model": "gpt-4o-mini", "api_key": "invalid"}),
            ("anthropic", {"model": "claude-3-haiku-20240307", "api_key": "invalid"}),
            ("gemini", {"model": "gemini-pro", "api_key": "invalid"}),
            ("bedrock", {"model": "anthropic.claude-3-haiku-20240307-v1:0", "aws_access_key_id": "invalid"})
        ]
        
        for provider_name, config in providers_configs:
            llm = LLMFactory.create(provider_name, config)
            
            # All providers should raise an exception with invalid credentials
            with pytest.raises(Exception):
                llm.invoke(prompt="Test prompt")