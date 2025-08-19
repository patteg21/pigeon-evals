from typing import Dict, Any, List
from utils import logger
from utils.typing.chunks import DocumentChunk
from embedder import OpenAIEmbedder


class EmbedderRunner:
    """Runner for executing embedding operations."""
    
    def __init__(self):
        self.embedder_map = {
            "openai": OpenAIEmbedder,
        }
    
    async def run_embedder(self, chunks: List[DocumentChunk], embedding_config: Dict[str, Any]) -> List[DocumentChunk]:
        """Run embedding on chunks based on config."""
        provider = embedding_config.get("provider", "openai")
        
        if provider not in self.embedder_map:
            available = ", ".join(self.embedder_map.keys())
            raise NotImplementedError(
                f"Embedding provider '{provider}' is not implemented. "
                f"Available providers: {available}"
            )
        
        logger.info(f"Running {provider} embedder on {len(chunks)} chunks")
        embedder_class = self.embedder_map[provider]
        embedder = embedder_class(embedding_config)
        
        # Embed all chunks
        embedded_chunks = await embedder.embed_chunks(chunks)
        
        logger.info(f"Successfully embedded {len(embedded_chunks)} chunks")
        return embedded_chunks
    
    def get_available_providers(self) -> List[str]:
        """Get list of available embedding providers."""
        return list(self.embedder_map.keys())