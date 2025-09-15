import pytest
import os
from unittest.mock import patch

from src.infra.embedding.factory import EmbedderFactory
from src.infra.embedding.openai_embedder import OpenAIEmbedder
from src.infra.embedding.huggingface_embedder import HuggingFaceEmbedder
from src.models import DocumentChunk
from src.models.configs.embedding import EmbeddingConfig


class TestOpenAIEmbedder:
    """Integration tests for OpenAI embedding provider."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.test_config = EmbeddingConfig(
            model="text-embedding-3-small",
            pooling_strategy="mean"
        )
        self.test_chunk = DocumentChunk(
            id="test_chunk_1",
            text="This is a test document chunk for embedding.",
            type_chunk="text",
            document="test_document"
        )

    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not available")
    async def test_openai_embedder_single_chunk(self):
        """Test OpenAI embedder with a single chunk."""
        embedder = OpenAIEmbedder(self.test_config)
        
        # Test single chunk embedding
        result = await embedder.embed_chunk(self.test_chunk)
        
        # Verify result structure
        assert isinstance(result, DocumentChunk)
        assert result.id == self.test_chunk.id
        assert result.text == self.test_chunk.text
        assert result.embeddding is not None  # Note: keeping original typo
        assert isinstance(result.embeddding, list)
        assert len(result.embeddding) > 0
        assert all(isinstance(x, float) for x in result.embeddding)

    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not available")
    async def test_openai_embedder_multiple_chunks(self):
        """Test OpenAI embedder with multiple chunks."""
        embedder = OpenAIEmbedder(self.test_config)
        
        chunks = [
            DocumentChunk(id="chunk_1", text="First test chunk", type_chunk="text", document="doc1"),
            DocumentChunk(id="chunk_2", text="Second test chunk", type_chunk="text", document="doc2"),
            DocumentChunk(id="chunk_3", text="Third test chunk", type_chunk="text", document="doc3")
        ]
        
        # Test batch embedding
        results = await embedder.embed_chunks(chunks)
        
        # Verify results
        assert len(results) == 3
        for i, result in enumerate(results):
            assert isinstance(result, DocumentChunk)
            assert result.id == chunks[i].id
            assert result.embeddding is not None
            assert isinstance(result.embeddding, list)
            assert len(result.embeddding) > 0

    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not available")
    async def test_openai_embedder_large_text(self):
        """Test OpenAI embedder with text that exceeds token limits."""
        embedder = OpenAIEmbedder(self.test_config)
        
        # Create a chunk with very long text
        long_text = "This is a test sentence. " * 1000  # Long text that will exceed token limits
        large_chunk = DocumentChunk(
            id="large_chunk",
            text=long_text,
            type_chunk="text",
            document="large_doc"
        )
        
        # Test embedding with large text (should use chunking strategy)
        result = await embedder.embed_chunk(large_chunk)
        
        # Verify result
        assert isinstance(result, DocumentChunk)
        assert result.embeddding is not None
        assert isinstance(result.embeddding, list)
        assert len(result.embeddding) > 0

    @pytest.mark.asyncio
    async def test_openai_embedder_invalid_api_key(self):
        """Test OpenAI embedder with invalid API key."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "invalid_key"}):
            embedder = OpenAIEmbedder(self.test_config)
            
            # Should raise an exception for invalid API key
            with pytest.raises(Exception):
                await embedder.embed_chunk(self.test_chunk)


class TestHuggingFaceEmbedder:
    """Integration tests for HuggingFace embedding provider."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.test_config = EmbeddingConfig(
            model="sentence-transformers/all-MiniLM-L6-v2",
            pooling_strategy="mean"
        )
        self.test_chunk = DocumentChunk(
            id="test_chunk_1",
            text="This is a test document chunk for embedding.",
            type_chunk="text",
            document="test_document"
        )

    @pytest.mark.asyncio
    async def test_huggingface_embedder_single_chunk(self):
        """Test HuggingFace embedder with a single chunk."""
        embedder = HuggingFaceEmbedder(self.test_config)
        
        # Test single chunk embedding
        result = await embedder.embed_chunk(self.test_chunk)
        
        # Verify result structure
        assert isinstance(result, DocumentChunk)
        assert result.id == self.test_chunk.id
        assert result.text == self.test_chunk.text
        assert result.embeddding is not None
        assert isinstance(result.embeddding, list)
        assert len(result.embeddding) > 0
        assert all(isinstance(x, float) for x in result.embeddding)

    @pytest.mark.asyncio
    async def test_huggingface_embedder_multiple_chunks(self):
        """Test HuggingFace embedder with multiple chunks."""
        embedder = HuggingFaceEmbedder(self.test_config)
        
        chunks = [
            DocumentChunk(id="chunk_1", text="First test chunk", type_chunk="text", document="doc1"),
            DocumentChunk(id="chunk_2", text="Second test chunk", type_chunk="text", document="doc2"),
            DocumentChunk(id="chunk_3", text="Third test chunk", type_chunk="text", document="doc3")
        ]
        
        # Test batch embedding
        results = await embedder.embed_chunks(chunks)
        
        # Verify results
        assert len(results) == 3
        for i, result in enumerate(results):
            assert isinstance(result, DocumentChunk)
            assert result.id == chunks[i].id
            assert result.embeddding is not None
            assert isinstance(result.embeddding, list)
            assert len(result.embeddding) > 0

    @pytest.mark.asyncio
    async def test_huggingface_embedder_model_info(self):
        """Test HuggingFace embedder model info retrieval."""
        embedder = HuggingFaceEmbedder(self.test_config)
        
        model_info = embedder.get_model_info()
        
        # Verify model info structure
        assert isinstance(model_info, dict)
        assert "embedding_dimension" in model_info
        assert "max_seq_length" in model_info
        assert "device" in model_info
        assert isinstance(model_info["embedding_dimension"], int)
        assert model_info["embedding_dimension"] > 0

    @pytest.mark.asyncio
    async def test_huggingface_embedder_different_model(self):
        """Test HuggingFace embedder with a different model."""
        config = EmbeddingConfig(
            model="sentence-transformers/paraphrase-MiniLM-L3-v2",
            pooling_strategy="mean"
        )
        embedder = HuggingFaceEmbedder(config)
        
        # Test embedding
        result = await embedder.embed_chunk(self.test_chunk)
        
        # Verify result
        assert isinstance(result, DocumentChunk)
        assert result.embeddding is not None
        assert isinstance(result.embeddding, list)
        assert len(result.embeddding) > 0


class TestEmbedderFactory:
    """Integration tests for EmbedderFactory."""
    
    def test_create_openai_embedder(self):
        """Test creating OpenAI embedder through factory."""
        config = EmbeddingConfig(
            model="text-embedding-3-small",
            pooling_strategy="mean"
        )
        
        embedder = EmbedderFactory.create("openai", config)
        
        # Verify embedder type
        assert isinstance(embedder, OpenAIEmbedder)
        assert embedder.provider_name == "openai"

    def test_create_huggingface_embedder(self):
        """Test creating HuggingFace embedder through factory."""
        config = EmbeddingConfig(
            model="sentence-transformers/all-MiniLM-L6-v2",
            pooling_strategy="mean"
        )
        
        embedder = EmbedderFactory.create("huggingface", config)
        
        # Verify embedder type
        assert isinstance(embedder, HuggingFaceEmbedder)
        assert embedder.provider_name == "huggingface"

    def test_create_unknown_provider_fallback(self):
        """Test creating embedder with unknown provider (should fallback to HuggingFace)."""
        config = EmbeddingConfig(
            model="sentence-transformers/all-MiniLM-L6-v2",
            pooling_strategy="mean"
        )
        
        embedder = EmbedderFactory.create("unknown_provider", config)
        
        # Should fallback to HuggingFace
        assert isinstance(embedder, HuggingFaceEmbedder)
        assert embedder.provider_name == "huggingface"

    @patch('pathlib.Path.exists')
    def test_create_from_config_file_not_found(self, mock_exists):
        """Test creating embedder when config file doesn't exist."""
        mock_exists.return_value = False
        
        embedder = EmbedderFactory.create_from_config()
        
        # Should create default HuggingFace embedder
        assert isinstance(embedder, HuggingFaceEmbedder)
        assert embedder.provider_name == "huggingface"


class TestEmbeddingProviderComparison:
    """Integration tests comparing different embedding providers."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.test_chunk = DocumentChunk(
            id="comparison_chunk",
            text="This is a test document for comparing embedding providers.",
            type_chunk="text",
            document="comparison_doc"
        )

    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not available")
    async def test_embedding_consistency_across_providers(self):
        """Test that both providers produce consistent embeddings for same text."""
        # Create embedders
        openai_config = EmbeddingConfig(
            model="text-embedding-3-small",
            pooling_strategy="mean"
        )
        hf_config = EmbeddingConfig(
            model="sentence-transformers/all-MiniLM-L6-v2",
            pooling_strategy="mean"
        )
        
        openai_embedder = OpenAIEmbedder(openai_config)
        hf_embedder = HuggingFaceEmbedder(hf_config)
        
        # Get embeddings from both providers
        openai_result = await openai_embedder.embed_chunk(self.test_chunk)
        hf_result = await hf_embedder.embed_chunk(self.test_chunk)
        
        # Both should produce valid embeddings (dimensions will differ)
        assert openai_result.embeddding is not None
        assert hf_result.embeddding is not None
        assert len(openai_result.embeddding) > 0
        assert len(hf_result.embeddding) > 0
        
        # Both embeddings should be normalized (values between -1 and 1)
        assert all(-1.1 <= x <= 1.1 for x in openai_result.embeddding)
        assert all(-1.1 <= x <= 1.1 for x in hf_result.embeddding)

    @pytest.mark.asyncio
    async def test_embedding_reproducibility(self):
        """Test that embeddings are reproducible for same input."""
        config = EmbeddingConfig(
            model="sentence-transformers/all-MiniLM-L6-v2",
            pooling_strategy="mean"
        )
        embedder = HuggingFaceEmbedder(config)
        
        # Get embedding twice for same text
        result1 = await embedder.embed_chunk(self.test_chunk)
        result2 = await embedder.embed_chunk(self.test_chunk)
        
        # Embeddings should be identical
        assert len(result1.embeddding) == len(result2.embeddding)
        for v1, v2 in zip(result1.embeddding, result2.embeddding):
            assert abs(v1 - v2) < 1e-6  # Allow for small floating point differences