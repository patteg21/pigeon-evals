import pytest
import numpy as np
import os
import tempfile
from unittest.mock import AsyncMock, patch

from clients import EmbeddingModel
from utils.pca import PCALoader


class TestPCALoader:
    """Test cases for PCALoader with embedding vector generation"""

    @pytest.fixture
    def sample_embeddings(self):
        """Generate sample embeddings for testing"""
        np.random.seed(42)
        return np.random.rand(1000, 1536).tolist()  # 1000 samples, 1536 dims (OpenAI embedding size)

    @pytest.fixture
    def sample_sentence(self):
        """Sample sentence for embedding generation"""
        return "Apple Inc. reported strong quarterly earnings with revenue growth of 15% year-over-year."

    @pytest.fixture
    def temp_pca_path(self):
        """Create temporary path for PCA model"""
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            temp_path = f.name
        yield temp_path
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.fixture
    def pca_loader(self, temp_pca_path):
        """Create PCALoader instance with temporary path"""
        return PCALoader(path=temp_pca_path, target_dim=512, seed=42)

    @pytest.fixture
    async def embedding_model(self):
        """Create EmbeddingModel instance"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            return EmbeddingModel()

    def test_pca_loader_initialization(self, pca_loader):
        """Test PCALoader initialization"""
        assert pca_loader.target_dim == 512
        assert pca_loader.seed == 42
        assert pca_loader.model is None

    def test_pca_fit_and_transform(self, pca_loader, sample_embeddings):
        """Test PCA fitting and transformation"""
        # Fit the PCA model
        pca_loader.fit(sample_embeddings)
        
        assert pca_loader.model is not None
        assert pca_loader.model.n_components_ == 512
        
        # Transform embeddings
        transformed = pca_loader.transform(sample_embeddings[:10])
        
        assert transformed.shape == (10, 512)
        # Check L2 normalization
        norms = np.linalg.norm(transformed, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)

    def test_transform_one(self, pca_loader, sample_embeddings):
        """Test single vector transformation"""
        # Fit first
        pca_loader.fit(sample_embeddings)
        
        # Transform single vector
        single_vec = sample_embeddings[0]
        transformed = pca_loader.transform_one(single_vec)
        
        assert len(transformed) == 512
        assert isinstance(transformed, list)
        # Check L2 normalization
        norm = np.linalg.norm(transformed)
        assert abs(norm - 1.0) < 1e-5

    def test_save_and_load(self, pca_loader, sample_embeddings, temp_pca_path):
        """Test saving and loading PCA model"""
        # Fit and save
        pca_loader.fit(sample_embeddings)
        pca_loader.save()
        
        assert os.path.exists(temp_pca_path)
        
        # Load in new instance
        new_loader = PCALoader(path=temp_pca_path)
        new_loader.load()
        
        assert new_loader.model is not None
        
        # Test consistency
        vec = sample_embeddings[0]
        original_transform = pca_loader.transform_one(vec)
        loaded_transform = new_loader.transform_one(vec)
        
        np.testing.assert_allclose(original_transform, loaded_transform, rtol=1e-3)

    @pytest.mark.asyncio
    async def test_sentence_embedding_with_pca(self, sample_sentence, temp_pca_path):
        """
        Integration test: Generate embedding for sentence and apply PCA transformation.
        This test requires both EmbeddingModel and PCALoader working together.
        """
        # Mock the OpenAI API call to avoid real API usage
        mock_embedding = np.random.rand(1536).tolist()
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            embedding_model = EmbeddingModel()
            
            # Mock the embedding creation
            with patch.object(embedding_model, '_embeddings', return_value=mock_embedding):
                # Generate embedding for sentence
                sentence_embedding = await embedding_model.create_embedding(sample_sentence)
                
                # Create sample embeddings for PCA training (including our sentence embedding)
                np.random.seed(42)
                training_embeddings = np.random.rand(999, 1536).tolist()
                training_embeddings.append(sentence_embedding)
                
                # Initialize and train PCA
                pca_loader = PCALoader(path=temp_pca_path, target_dim=512, seed=42)
                pca_loader.fit(training_embeddings)
                
                # Transform the sentence embedding
                reduced_embedding = pca_loader.transform_one(sentence_embedding)
                
                # Assertions
                assert len(sentence_embedding) == 1536  # Original OpenAI embedding size
                assert len(reduced_embedding) == 512    # PCA reduced size
                assert isinstance(reduced_embedding, list)
                
                # Check L2 normalization
                norm = np.linalg.norm(reduced_embedding)
                assert abs(norm - 1.0) < 1e-5
                
                # Verify the reduced embedding is different from original
                # (truncated comparison since dimensions differ)
                original_first_512 = sentence_embedding[:512]
                assert not np.allclose(original_first_512, reduced_embedding, rtol=1e-3)

    def test_error_cases(self, pca_loader, sample_embeddings):
        """Test error handling"""
        # Test transform without fitting
        with pytest.raises(RuntimeError, match="PCA model not loaded/fitted"):
            pca_loader.transform(sample_embeddings)
        
        with pytest.raises(RuntimeError, match="PCA model not loaded"):
            pca_loader.transform_one(sample_embeddings[0])
        
        # Test save without fitting
        with pytest.raises(RuntimeError, match="Nothing to save"):
            pca_loader.save()

    def test_nan_inf_handling(self, pca_loader):
        """Test handling of NaN/Inf values in embeddings"""
        bad_embeddings = [[1.0, 2.0, np.nan], [1.0, np.inf, 3.0]]
        
        with pytest.raises(ValueError, match="Embeddings contain NaN/Inf"):
            pca_loader.fit(bad_embeddings)

    @pytest.mark.asyncio
    async def test_load_pretrained_pca_from_artifacts(self, sample_sentence):
        """
        Test loading the actual pre-trained PCA model from artifacts directory
        and using it to transform sentence embeddings (like in tools.py).
        This test MUST fail if the artifacts PCA model cannot be loaded.
        """
        artifacts_pca_path = "artifacts/sec_pca_512.joblib"
        
        mock_embedding = np.random.rand(1536).tolist()
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            embedding_model = EmbeddingModel()
            
            with patch.object(embedding_model, '_embeddings', return_value=mock_embedding):
                # Generate embedding for sentence
                sentence_embedding = await embedding_model.create_embedding(sample_sentence)
                
                # Load pre-trained PCA model from artifacts - MUST succeed or test fails
                try:
                    pca_loader = PCALoader(path=artifacts_pca_path).load()
                except Exception as e:
                    pytest.fail(f"Failed to load pre-trained PCA model from {artifacts_pca_path}: {e}")
                
                # Transform the sentence embedding using pre-trained model
                reduced_embedding = pca_loader.transform_one(sentence_embedding)
                
                # Assertions
                assert len(sentence_embedding) == 1536  # Original OpenAI embedding size
                assert len(reduced_embedding) == 512    # PCA reduced size
                assert isinstance(reduced_embedding, list)
                
                # Check L2 normalization
                norm = np.linalg.norm(reduced_embedding)
                assert abs(norm - 1.0) < 1e-5
                
                # Verify the model was loaded successfully
                assert pca_loader.model is not None
                assert pca_loader.model.n_components_ == 512

    @pytest.mark.asyncio
    async def test_tools_workflow_with_artifacts_pca(self, sample_sentence):
        """
        Test the exact workflow from tools.py: load artifacts PCA and
        use it for query-time dimensionality reduction.
        This test MUST fail if PCA loading fails.
        """
        mock_embedding = np.random.rand(1536).tolist()
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            embedding_model = EmbeddingModel()
            
            with patch.object(embedding_model, '_embeddings', return_value=mock_embedding):
                # Simulate the tools.py workflow - PCA MUST load successfully
                try:
                    reducer = PCALoader(path="artifacts/sec_pca_512.joblib").load()
                    print("PCA reducer loaded for query-time dimensionality reduction.")
                except Exception as e:
                    pytest.fail(f"PCA loading failed in tools.py workflow: {e}")
                
                def _reduce(vec: list[float]) -> list[float]:
                    # Since we require PCA to load, this should always use PCA transform
                    return reducer.transform_one(vec)
                
                # Test vector search workflow
                query = sample_sentence
                vec = await embedding_model.create_embedding(query)
                reduced_vec = _reduce(vec)
                
                # Assertions - PCA must be working
                assert len(vec) == 1536  # Original embedding size
                assert len(reduced_vec) == 512  # PCA reduced size
                assert reducer.model.n_components_ == 512
                
                # Check L2 normalization
                norm = np.linalg.norm(reduced_vec)
                assert abs(norm - 1.0) < 1e-5

    @pytest.mark.asyncio
    async def test_integration_with_processing_pipeline(self, sample_sentence, temp_pca_path):
        """
        Test PCALoader integration similar to how it's used in processing.py
        """
        mock_embedding = np.random.rand(1536).tolist()
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            embedding_model = EmbeddingModel()
            
            with patch.object(embedding_model, '_embeddings', return_value=mock_embedding):
                # Simulate the workflow from processing.py
                
                # 1. Generate embeddings for multiple sentences
                sentences = [
                    sample_sentence,
                    "Microsoft Azure cloud services showed significant growth.",
                    "The Federal Reserve announced interest rate changes."
                ]
                
                embeddings = []
                for sentence in sentences:
                    emb = await embedding_model.create_embedding(sentence)
                    embeddings.append(emb)
                
                # Add more sample embeddings to have enough for PCA
                np.random.seed(42)
                additional_embeddings = np.random.rand(1000, 1536).tolist()
                all_embeddings = embeddings + additional_embeddings
                
                # 2. Train PCA on the embeddings (simulate train_pca_and_reduce_in_place)
                pca_loader = PCALoader(path=temp_pca_path, target_dim=512, seed=42)
                pca_loader.fit(all_embeddings)
                pca_loader.save()
                
                # 3. Transform embeddings in place
                transformed_embeddings = pca_loader.transform(embeddings)
                
                # 4. Verify results
                assert transformed_embeddings.shape == (3, 512)
                
                # Check L2 normalization for all vectors
                norms = np.linalg.norm(transformed_embeddings, axis=1)
                np.testing.assert_allclose(norms, 1.0, rtol=1e-5)
                
                # 5. Test loading and using the saved model (simulate tools.py workflow)
                query_loader = PCALoader(path=temp_pca_path).load()
                query_embedding = await embedding_model.create_embedding("What is Apple's revenue?")
                reduced_query = query_loader.transform_one(query_embedding)
                
                assert len(reduced_query) == 512
                assert abs(np.linalg.norm(reduced_query) - 1.0) < 1e-5