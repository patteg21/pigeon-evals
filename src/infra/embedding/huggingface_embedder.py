from typing import Dict, Any, List
import torch
from sentence_transformers import SentenceTransformer

from models import DocumentChunk
from models.configs import EmbeddingConfig
from utils import logger

from .base import BaseEmbedder


class HuggingFaceEmbedder(BaseEmbedder):
    """Hugging Face embedding provider using sentence-transformers."""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        
        # Get model configuration
        model_name = config.model

        # TODO: Allow users to switch these 
        self.device = "auto"
        self.batch_size = 32
        self.max_seq_length = None  # Use model default if None
        
        # Auto-detect device if not specified
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"  # Apple Silicon
            else:
                self.device = "cpu"
        
        logger.info(f"Initializing HuggingFace embedder with model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        try:
            # Load the sentence transformer model
            self.model = SentenceTransformer(model_name, device=self.device)
            
            # Log the model's default max sequence length
            model_default_length = self.model.max_seq_length
            logger.info(f"Model default max_seq_length: {model_default_length}")
            
            # Set max sequence length if specified and different from default
            if self.max_seq_length is not None:
                if self.max_seq_length != model_default_length:
                    # Update the tokenizer's max length for each module
                    for module in self.model._modules.values():
                        if hasattr(module, 'max_seq_length'):
                            module.max_seq_length = self.max_seq_length
                        if hasattr(module, 'tokenizer') and hasattr(module.tokenizer, 'model_max_length'):
                            # Only increase if we're going beyond model's default, otherwise respect model limits
                            if self.max_seq_length > module.tokenizer.model_max_length:
                                logger.warning(f"Requested max_seq_length ({self.max_seq_length}) exceeds model's max ({module.tokenizer.model_max_length}). Using model max.")
                                self.max_seq_length = module.tokenizer.model_max_length
                            else:
                                module.tokenizer.model_max_length = self.max_seq_length
                    
                    # Set on the main model
                    self.model.max_seq_length = self.max_seq_length
                    logger.info(f"Updated max_seq_length to: {self.max_seq_length}")
                else:
                    logger.info(f"Using model default max_seq_length: {model_default_length}")
                    self.max_seq_length = model_default_length
            else:
                self.max_seq_length = model_default_length
                logger.info(f"Using model default max_seq_length: {model_default_length}")
                
            logger.info(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
            
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model '{model_name}': {e}")
            raise
    
    @property
    def provider_name(self) -> str:
        return "huggingface"
    
    async def _embed_chunk_raw(self, chunk: DocumentChunk) -> List[float]:
        """Get raw HuggingFace embeddings for a single chunk."""
        try:
            # Get embedding for single text
            embedding = self.model.encode(
                chunk.text, 
                convert_to_tensor=False,  # Return numpy array
                normalize_embeddings=self.config.get("normalize", True)
            )
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Failed to embed chunk {chunk.id}: {e}")
            raise
    
    async def _embed_chunks_raw(self, chunks: List[DocumentChunk]) -> List[List[float]]:
        """Get raw HuggingFace embeddings for multiple chunks (batch optimized)."""
        try:
            texts = [chunk.text for chunk in chunks]
            logger.info(f"Embedding {len(texts)} chunks in batches of {self.batch_size}")
            
            # Process in batches for memory efficiency
            all_embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                
                # Get batch embeddings
                batch_embeddings = self.model.encode(
                    batch_texts,
                    convert_to_tensor=False,  # Return numpy arrays
                    normalize_embeddings=self.config.get("normalize", True),
                    show_progress_bar=False  # Disable progress bar for batches
                )
                
                # Convert to list of lists
                batch_embeddings_list = [emb.tolist() for emb in batch_embeddings]
                all_embeddings.extend(batch_embeddings_list)
                
                if (i // self.batch_size + 1) % 10 == 0:
                    logger.info(f"Processed {min(i + self.batch_size, len(texts))}/{len(texts)} chunks")
            
            logger.info(f"Successfully embedded all {len(all_embeddings)} chunks")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Failed to embed chunks: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model._modules_config[0].get('model_name', 'Unknown'),
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "max_seq_length": self.model.max_seq_length,
            "device": str(self.model.device),
            "normalize_embeddings": self.config.get("normalize", True)
        }