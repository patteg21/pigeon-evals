from typing import List, Tuple
import asyncio

from evals.src.utils import logger
from evals.src.utils.types import DocumentChunk, Embedding

from evals.src.embedder import OpenAIEmbedder, HuggingFaceEmbedder


class EmbedderRunner:
    """Runner for executing embedding operations."""
    
    def __init__(self, max_workers: int = 4):
        self.embedder_map = {
            "openai": OpenAIEmbedder,
            "huggingface": HuggingFaceEmbedder,
        }
        self.max_workers = max_workers
    
    async def run_embedder(self, chunks: List[DocumentChunk], embedding_config: Embedding) -> List[DocumentChunk]:
        """Run embedding on chunks based on config."""
        provider = embedding_config.provider
        
        if provider not in self.embedder_map:
            available = ", ".join(self.embedder_map.keys())
            raise NotImplementedError(
                f"Embedding provider '{provider}' is not implemented. "
                f"Available providers: {available}"
            )
        
        # Check if threading is enabled
        use_threading = embedding_config.use_threading
        max_workers = embedding_config.max_workers
        
        logger.info(f"Running {provider} embedder on {len(chunks)} chunks with threading={'enabled' if use_threading else 'disabled'}")
        
        if use_threading and len(chunks) > 1:
            embedded_chunks = await self._run_embedder_threaded(chunks, embedding_config, max_workers)
        else:
            embedded_chunks = await self._run_embedder_sequential(chunks, embedding_config)
        
        logger.info(f"Successfully embedded {len(embedded_chunks)} chunks")
        return embedded_chunks
    
    async def _run_embedder_sequential(self, chunks: List[DocumentChunk], embedding_config: Embedding) -> List[DocumentChunk]:
        """Run embedder sequentially without threading."""
        provider = embedding_config.provider
        embedder_class = self.embedder_map[provider]
        embedder: OpenAIEmbedder | HuggingFaceEmbedder = embedder_class(embedding_config.model_dump())
        
        return await embedder.embed_chunks(chunks)
    
    async def _run_embedder_threaded(self, chunks: List[DocumentChunk], embedding_config: Embedding, max_workers: int) -> List[DocumentChunk]:
        """Run embedder with threading for concurrent processing."""
        
        # Calculate optimal chunk size per thread
        chunk_size = max(1, len(chunks) // max_workers)
        if len(chunks) % max_workers != 0:
            chunk_size += 1
        
        # Split chunks into batches for each thread
        chunk_batches = [chunks[i:i + chunk_size] for i in range(0, len(chunks), chunk_size)]
        
        logger.info(f"Processing {len(chunks)} chunks across {len(chunk_batches)} threads (max_workers={max_workers})")
        
        # Create tasks for concurrent processing - get RAW embeddings only
        tasks = []
        for batch in chunk_batches:
            if batch:  # Only create task if batch is not empty
                task = self._process_chunk_batch_raw(batch, embedding_config)
                tasks.append(task)
        
        # Run all tasks concurrently to get raw embeddings
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect and flatten raw embeddings with their chunk info
        all_raw_embeddings = []
        chunks_in_order = []
        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"Thread failed with error: {result}")
                raise result
            for chunk, raw_embedding in result:
                chunks_in_order.append(chunk)
                all_raw_embeddings.append(raw_embedding)
        
        # Now apply PCA once on ALL raw embeddings if configured
        provider = embedding_config.provider
        embedder_class = self.embedder_map[provider]
        embedder = embedder_class(embedding_config.model_dump())
        
        if embedder.reducer:
            logger.info(f"Applying {embedder.reducer.name} dimensional reduction to all {len(all_raw_embeddings)} embeddings")
            # Train PCA on ALL raw embeddings, then transform them
            reduced_embeddings = embedder.reducer.fit_transform(all_raw_embeddings)
            # Save trained PCA model to disk for later use
            embedder.reducer.save()
        else:
            reduced_embeddings = all_raw_embeddings
        
        # Create embedded chunks with the processed embeddings
        embedded_chunks = []
        for chunk, embedding in zip(chunks_in_order, reduced_embeddings):
            embedded_chunk = DocumentChunk(
                id=chunk.id,
                text=chunk.text,
                type_chunk=chunk.type_chunk,
                document=chunk.document,
                embeddding=embedding  # Note: keeping original typo for compatibility
            )
            embedded_chunks.append(embedded_chunk)
        
        return embedded_chunks
    
    async def _process_chunk_batch_raw(self, chunk_batch: List[DocumentChunk], embedding_config: Embedding) -> List[Tuple[DocumentChunk, List[float]]]:
        """Process a batch of chunks in a single thread, returning raw embeddings only."""
        try:
            provider = embedding_config.provider
            embedder_class = self.embedder_map[provider]
            embedder: OpenAIEmbedder | HuggingFaceEmbedder = embedder_class(embedding_config.model_dump())
            
            logger.debug(f"Thread processing {len(chunk_batch)} chunks for raw embeddings")
            
            # Get raw embeddings without PCA reduction
            raw_embeddings = await embedder._embed_chunks_raw(chunk_batch)
            
            # Return pairs of (chunk, raw_embedding)
            return list(zip(chunk_batch, raw_embeddings))
            
        except Exception as e:
            logger.error(f"Error processing chunk batch: {e}")
            raise
    
    async def _process_chunk_batch(self, chunk_batch: List[DocumentChunk], embedding_config: Embedding) -> List[DocumentChunk]:
        """Process a batch of chunks in a single thread."""
        try:
            provider = embedding_config.provider
            embedder_class = self.embedder_map[provider]
            embedder: OpenAIEmbedder | HuggingFaceEmbedder = embedder_class(embedding_config.model_dump())
            
            logger.debug(f"Thread processing {len(chunk_batch)} chunks")
            return await embedder.embed_chunks(chunk_batch)
            
        except Exception as e:
            logger.error(f"Error processing chunk batch: {e}")
            raise
    
    def get_available_providers(self) -> List[str]:
        """Get list of available embedding providers."""
        return list(self.embedder_map.keys())