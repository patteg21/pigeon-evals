from typing import Dict, Any, List
import torch
from sentence_transformers import CrossEncoder
from utils import logger
from storage.vector.reranker.base import RerankerBase

from utils.types.configs import RerankConfig

class HuggingFaceReranker(RerankerBase):
    """Hugging Face cross-encoder reranker using sentence-transformers."""
    
    def __init__(self, rerank_config: RerankConfig):
        super().__init__()
        
        # Set default config if none provided

        # Get model configuration

        self.model_name = rerank_config.model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.top_k: int = rerank_config.top_k or 10
        self.device = "auto" # TODO: Add this to the config
        self.max_length = 512 # TODO: Adjust to handle larger inputs
        

        # Auto-detect device if not specified
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"  # Apple Silicon
            else:
                self.device = "cpu"
        
        logger.info(f"Initializing HuggingFace reranker with model: {self.model_name}")
        logger.info(f"Using device: {self.device}")
        
        try:
            # Load the cross-encoder model
            self.model = CrossEncoder(self.model_name, device=self.device, max_length=self.max_length)
            logger.info("Cross-encoder model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load HuggingFace cross-encoder model '{self.model_name}': {e}")
            raise

    def rerank(self, documents: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """ Reranks a set of documents based on relevance to the query. """
        if not documents:
            return documents
            
        # Extract text content from documents (assuming 'text' field contains the document content)
        doc_texts = []
        for doc in documents:
            text = doc.get('text', '')
            if not text:
                # Fallback to other possible text fields
                text = doc.get('content', '') or doc.get('body', '') or str(doc)
            doc_texts.append(text)
        
        # Prepare query-document pairs for cross-encoder
        query_doc_pairs = [(query, doc_text) for doc_text in doc_texts]
        
        try:
            # Get relevance scores from cross-encoder
            scores = self.model.predict(query_doc_pairs)
            
            # Create pairs of (document, score) and sort by score descending
            doc_score_pairs = list(zip(documents, scores))
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Return top_k reranked documents with scores
            reranked_docs = []
            for doc, score in doc_score_pairs[:self.top_k]:
                doc_with_score = doc.copy()
                doc_with_score['rerank_score'] = float(score)
                reranked_docs.append(doc_with_score)
                
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            # Return original documents if reranking fails
            return documents[:self.top_k]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": getattr(self.model, 'model_name', 'Unknown'),
            "device": str(self.model.device) if hasattr(self.model, 'device') else self.device,
            "max_length": self.max_length
        }
