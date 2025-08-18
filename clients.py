from typing import List, Dict
import os

import tiktoken
from openai import AsyncOpenAI
from dotenv import load_dotenv
from pinecone import (
    Pinecone
)

from utils.typing import VectorObject

load_dotenv()

class EmbeddingModel:
    def __init__(self, model="text-embedding-3-small"):
        self.model = model
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    async def create_embedding(self, text: str) -> List[float]:
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
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text for the current model"""
        encoding = tiktoken.encoding_for_model(self.model)
        return len(encoding.encode(text))
    

class VectorDB:
    def __init__(self, index_name="sec-embeddings"):
        self.client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.client.Index(index_name)

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