# Integration Tests

This directory contains integration tests for the embedding and LLM providers in the pigeon-evals project.

## Overview

The integration tests are designed to test real API calls to verify that each provider works correctly with a single API call. These tests cover:

### Embedding Providers
- **OpenAI Embeddings** (`text-embedding-3-small`)
- **HuggingFace Embeddings** (sentence-transformers models)

### LLM Providers  
- **OpenAI** (GPT models)
- **Anthropic** (Claude models)
- **Google Gemini** (Gemini models) 
- **AWS Bedrock** (Various foundation models)

## Test Structure

```
tests/
├── integration/
│   ├── test_embedding_providers.py  # Tests for embedding providers
│   └── test_llm_providers.py        # Tests for LLM providers
└── README.md                        # This file
```

## Running Tests

### Prerequisites

Install test dependencies:
```bash
pip install pytest pytest-asyncio
```

Set up API keys in your environment:
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key" 
export GOOGLE_API_KEY="your-google-key"
export AWS_ACCESS_KEY_ID="your-aws-access-key"
export AWS_SECRET_ACCESS_KEY="your-aws-secret-key"
```

### Running All Tests

```bash
# Run all integration tests
python run_integration_tests.py

# Or use pytest directly
pytest tests/integration/ -v --asyncio-mode=auto
```

### Running Specific Tests

```bash
# Run only embedding tests
pytest tests/integration/test_embedding_providers.py -v

# Run only LLM tests  
pytest tests/integration/test_llm_providers.py -v

# Run tests for specific provider
pytest tests/integration/ -k "openai" -v

# Run tests that don't require API keys
pytest tests/integration/ -k "not (skipif)" -v
```

### Running with Docker

If you prefer to run tests in isolation:

```bash
# Build test container
docker build -t pigeon-evals-tests .

# Run tests with environment variables
docker run --rm \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  pigeon-evals-tests
```

## Test Coverage

### Embedding Provider Tests

Each embedding provider is tested for:

1. **Single Chunk Embedding**: Test embedding a single text chunk
2. **Multiple Chunk Embedding**: Test batch embedding multiple chunks  
3. **Large Text Handling**: Test handling of text that exceeds token limits
4. **Error Handling**: Test behavior with invalid API keys
5. **Factory Creation**: Test creation through the EmbedderFactory
6. **Provider-Specific Features**: Test unique features of each provider

### LLM Provider Tests

Each LLM provider is tested for:

1. **Basic Invocation**: Test with simple prompts and queries
2. **Custom Parameters**: Test with custom temperature, max_tokens, etc.
3. **Parameter Validation**: Test error handling for invalid parameters
4. **Error Handling**: Test behavior with invalid API keys
5. **Factory Creation**: Test creation through the LLMFactory
6. **Model-Specific Formats**: Test different request/response formats

### Cross-Provider Tests

Additional tests verify:

1. **Consistency**: Compare outputs from different providers for same input
2. **Reproducibility**: Verify embeddings are consistent across runs
3. **Error Handling**: Ensure all providers handle errors consistently

## Test Configuration

Tests use environment variables for configuration and will be automatically skipped if required API keys are not available. This ensures tests can run in CI/CD environments without failing due to missing credentials.

### Skipped Tests

Tests are skipped when:
- Required API key is not set in environment
- Required dependencies are not installed
- Network connectivity issues prevent API access

### Mock Tests  

Some tests include mocked versions for testing without API access:
- Bedrock tests include mocked AWS responses
- Factory tests use mocked configurations

## CI/CD Integration

These tests are designed to work in CI/CD environments:

```yaml
# Example GitHub Actions workflow
- name: Run Integration Tests
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
  run: |
    python run_integration_tests.py
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the project root directory
2. **API Key Errors**: Verify your API keys are valid and have sufficient quota
3. **Network Errors**: Check your internet connection and any firewall rules
4. **Rate Limiting**: Some tests may fail due to API rate limits - run with fewer parallel tests

### Debug Mode

Run tests with additional debugging:

```bash
pytest tests/integration/ -v -s --tb=long --log-cli-level=DEBUG
```

### Test Individual Providers

If you want to test only providers you have keys for:

```bash
# Test only OpenAI (if you have OPENAI_API_KEY)
pytest tests/integration/ -k "openai" -v

# Test only HuggingFace (no API key required)  
pytest tests/integration/ -k "huggingface" -v
```

## Contributing

When adding new providers or tests:

1. Follow the existing test patterns
2. Include both positive and negative test cases  
3. Use appropriate skip decorators for API key requirements
4. Add factory tests for new providers
5. Include comparison tests when applicable
6. Update this README with new provider information