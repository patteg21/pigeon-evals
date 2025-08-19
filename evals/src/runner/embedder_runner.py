from typing import Dict, Any, List
import asyncio
import concurrent.futures
from utils import logger
from utils.typing.chunks import DocumentChunk
from ..embedder import OpenAIEmbedder, HuggingFaceEmbedder


class EmbedderRunner:
    """Runner for executing embedding operations."""
    
    def __init__(self, max_workers: int = 4):
        self.embedder_map = {
            "openai": OpenAIEmbedder,
            "huggingface": HuggingFaceEmbedder,
        }
        self.max_workers = max_workers
    
    async def run_embedder(self, chunks: List[DocumentChunk], embedding_config: Dict[str, Any]) -> List[DocumentChunk]:
        """Run embedding on chunks based on config."""
        provider = embedding_config.get("provider", "openai")
        
        if provider not in self.embedder_map:
            available = ", ".join(self.embedder_map.keys())
            raise NotImplementedError(
                f"Embedding provider '{provider}' is not implemented. "
                f"Available providers: {available}"
            )
        
        # Check if threading is enabled
        use_threading = embedding_config.get("use_threading", True)
        max_workers = embedding_config.get("max_workers", self.max_workers)
        
        logger.info(f"Running {provider} embedder on {len(chunks)} chunks with threading={'enabled' if use_threading else 'disabled'}")
        
        if use_threading and len(chunks) > 1:
            embedded_chunks = await self._run_embedder_threaded(chunks, embedding_config, max_workers)
        else:
            embedded_chunks = await self._run_embedder_sequential(chunks, embedding_config)
        
        logger.info(f"Successfully embedded {len(embedded_chunks)} chunks")
        return embedded_chunks
    
    async def _run_embedder_sequential(self, chunks: List[DocumentChunk], embedding_config: Dict[str, Any]) -> List[DocumentChunk]:
        """Run embedder sequentially without threading."""
        provider = embedding_config.get("provider", "openai")
        embedder_class = self.embedder_map[provider]
        embedder: OpenAIEmbedder | HuggingFaceEmbedder = embedder_class(embedding_config)
        
        return await embedder.embed_chunks(chunks)
    
    async def _run_embedder_threaded(self, chunks: List[DocumentChunk], embedding_config: Dict[str, Any], max_workers: int) -> List[DocumentChunk]:
        """Run embedder with threading for concurrent processing."""
        
        # Calculate optimal chunk size per thread
        chunk_size = max(1, len(chunks) // max_workers)
        if len(chunks) % max_workers != 0:
            chunk_size += 1
        
        # Split chunks into batches for each thread
        chunk_batches = [chunks[i:i + chunk_size] for i in range(0, len(chunks), chunk_size)]
        
        logger.info(f"Processing {len(chunks)} chunks across {len(chunk_batches)} threads (max_workers={max_workers})")
        
        # Create tasks for concurrent processing
        tasks = []
        for batch in chunk_batches:
            if batch:  # Only create task if batch is not empty
                task = self._process_chunk_batch(batch, embedding_config)
                tasks.append(task)
        
        # Run all tasks concurrently
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect and flatten results
        embedded_chunks = []
        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"Thread failed with error: {result}")
                raise result
            embedded_chunks.extend(result)
        
        return embedded_chunks
    
    async def _process_chunk_batch(self, chunk_batch: List[DocumentChunk], embedding_config: Dict[str, Any]) -> List[DocumentChunk]:
        """Process a batch of chunks in a single thread."""
        try:
            provider = embedding_config.get("provider", "openai")
            embedder_class = self.embedder_map[provider]
            embedder: OpenAIEmbedder | HuggingFaceEmbedder = embedder_class(embedding_config)
            
            logger.debug(f"Thread processing {len(chunk_batch)} chunks")
            return await embedder.embed_chunks(chunk_batch)
            
        except Exception as e:
            logger.error(f"Error processing chunk batch: {e}")
            raise
    
    def get_available_providers(self) -> List[str]:
        """Get list of available embedding providers."""
        return list(self.embedder_map.keys())