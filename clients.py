from typing import List, Dict, Iterable
import os

import tiktoken
import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pinecone import (
    Pinecone
)

from utils.typing import (
    VectorObject,
    Pooling,
)

load_dotenv()


class EmbeddingModel:
    TOKEN_LIMITS: Dict[str, int] = {
        "text-embedding-3-small": 8191,
        # add models as needed
    }

    def __init__(self, model="text-embedding-3-small"):
        self.model: str = model
        if model not in self.TOKEN_LIMITS:
            raise ValueError(f"Unsupported model '{model}'. Provide max_tokens explicitly or add to TOKEN_LIMITS.")

        self.max_tokens: int = self.TOKEN_LIMITS[model]
        self.client: AsyncOpenAI = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.encoding = tiktoken.encoding_for_model(model)

    
    async def _embeddings(self, text: str) -> List[float]:
        """Create embedding for a single text"""
        response = await self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding
    
    async def create_embeddings_batch(self, texts: List[str]) ->  List[List[float]]:
        """Create embeddings for multiple texts"""
        response = await self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        return [data.embedding for data in response.data]
    
    async def count_tokens(self, text: str) -> int:
        """Count tokens in a text for the current model"""
        encoding = tiktoken.encoding_for_model(self.model)
        return len(encoding.encode(text))
    
    async def _is_too_large(self, tokens):
        return True if tokens > self.max_tokens else False

    @staticmethod
    def _l2n(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """L2 normalization to put between units between 0-1 """
        n = np.linalg.norm(v) + eps
        return v / n
    
    @staticmethod
    def _pool(
        vecs: np.ndarray, 
        strategy: Pooling, 
        weights: Iterable[float] | None = None
    ) -> np.ndarray:
        if strategy == "mean":
            return vecs.mean(axis=0)
        if strategy == "max":
            return vecs.max(axis=0)
        if strategy == "weighted":
            w = np.asarray(list(weights) if weights is not None else [1.0]*len(vecs), dtype=np.float32)
            w = w / (w.sum() if w.sum() > 0 else 1.0)
            return (vecs * w[:, None]).sum(axis=0)
        if strategy == "smooth_decay":
            # Exponential decay by chunk index (earlier chunks weigh slightly more)
            idx = np.arange(len(vecs), dtype=np.float32)
            # decay factor ~0.9 per step; adjust if needed
            w = np.power(0.9, idx)
            w = w / w.sum()
            return (vecs * w[:, None]).sum(axis=0)
        raise ValueError(f"Unknown pooling strategy: {strategy}")


    def _chunk_by_tokens(self, text: str, max_tokens: int, overlap: int) -> List[str]:
        """ Chunking based on the overlap and max_tokens """
        ids = self.encoding.encode(text)
        if len(ids) <= max_tokens:
            return [text]
        chunks, start = [], 0
        while start < len(ids):
            end = min(start + max_tokens, len(ids))
            chunks.append(self.encoding.decode(ids[start:end]))
            if end == len(ids): 
                break
            start = max(0, end - overlap)
        return chunks

        

    async def create_embedding(
        self,
        text: str,
        strategy: Pooling = "mean",
        *,
        chunk_max_tokens: int = 2048,
        overlap_tokens: int = 128,
        batch_size: int = 64,
        normalize_chunks: bool = True,
        normalize_output: bool = True,
        weighted_by_length: bool = True,
    ) -> List[float]:
        """
        Return a single pooled embedding vector for the given text.

        If the text exceeds the model's token limit, it is split into chunks
        (`chunk_max_tokens` with `overlap_tokens`) and each chunk is embedded
        separately. The resulting embeddings are then combined using the
        specified pooling strategy:

        - "mean": average across all chunk embeddings
        - "max": elementwise maximum
        - "weighted": weighted average by chunk token length
        - "smooth_decay": exponentially decayed weights (earlier chunks weighted more)

        Args:
            text (str): Input text to embed.
            strategy (str, optional): Pooling strategy to combine chunk embeddings.
                One of {"mean", "max", "weighted", "smooth_decay"}. Default is "mean".
            chunk_max_tokens (int, optional): Maximum tokens per chunk. Defaults to
                a safe limit under the model's max capacity. Default is 2048
            overlap_tokens (int, optional): Overlap tokens between adjacent chunks.
                Helps preserve context across boundaries. Default is 128.
            batch_size (int, optional): Number of chunks sent per API request. Default 64.
            normalize_chunks (bool, optional): Whether to L2-normalize chunk embeddings
                before pooling. Default True.
            normalize_output (bool, optional): Whether to L2-normalize the final pooled
                embedding. Default True.
            weighted_by_length (bool, optional): When using "weighted" pooling, use
                chunk token length as weights. Default True.

        Returns:
            List[float]: A single pooled embedding vector.
        """
        total = await self.count_tokens(text)
        if not await self._is_too_large(total):
            vec = np.asarray(await self._embeddings(text), dtype=np.float32)
            return self._l2n(vec).tolist() if normalize_output else vec.tolist()

        if chunk_max_tokens > self.max_tokens:
            raise ValueError(
                f"`chunk_max_tokens` ({chunk_max_tokens}) cannot exceed model limit ({self.max_tokens})."
            )
        cap = chunk_max_tokens # rename for better naming internally
        chunks = self._chunk_by_tokens(text, cap, overlap_tokens) # chunk the text

        embs: List[List[float]] = []
        for i in range(0, len(chunks), batch_size): # batch embeddigs
            embs.extend(await self.create_embeddings_batch(chunks[i:i+batch_size]))

        vecs = np.asarray(embs, dtype=np.float32)
        if normalize_chunks:
            vecs = np.vstack([self._l2n(v) for v in vecs])

        weights = [await self.count_tokens(c) for c in chunks] if (strategy == "weighted" and weighted_by_length) else None
        pooled = self._pool(vecs, strategy=strategy, weights=weights)
        if normalize_output:
            pooled = self._l2n(pooled)
        return pooled.astype(np.float32).tolist()
    

class VectorDB:
    def __init__(self, index_name="sec-embeddings"):
        self.client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.client.Index(index_name)

    def retrieve_from_id(self, vector_id: str):
        """Retrieve a vector by its ID"""
        response = self.index.fetch(ids=[vector_id])
        return response.vectors.get(vector_id)

    def upload(self, vector_object: VectorObject):
        metadata: Dict = vector_object.model_dump(exclude={'id', 'embeddings'})
        
        # Prepare vector for upsert
        vector_data = {
            "id": vector_object.id,
            "values": vector_object.embeddings,
            "metadata": metadata
        }
        
        return self.index.upsert(vectors=[vector_data])
    
    
    def query(self, vector, top_k=10, include_metadata=True):
        """Query the index for similar vectors"""
        return self.index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=include_metadata
        )
    
    def delete(self, ids: List[str]):
        """Delete vectors by IDs"""
        return self.index.delete(ids=ids)
    
    def clear(self):
        """Completely wipe all vectors from the index"""
        return self.index.delete(delete_all=True)