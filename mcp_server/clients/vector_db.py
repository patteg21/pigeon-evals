from typing import List
import os

from dotenv import load_dotenv
from pinecone import (
    Pinecone
)

from utils.typing import EntityType
from utils.typing.chunks import DocumentChunk


class VectorDBError(Exception):
    """Base exception for VectorDB operations"""
    pass


class MetadataFieldError(VectorDBError):
    """Error raised when required metadata fields are missing or invalid"""
    
    def __init__(self, field_name: str, message: str = None):
        self.field_name = field_name
        if message is None:
            message = f"Missing or invalid metadata field: {field_name}"
        super().__init__(message)


class VectorNotFoundError(VectorDBError):
    """Error raised when a vector is not found in the database"""
    
    def __init__(self, vector_id: str):
        self.vector_id = vector_id
        super().__init__(f"Vector not found: {vector_id}")


class InvalidFilterError(VectorDBError):
    """Error raised when filter parameters are invalid"""
    
    def __init__(self, filter_param: str, value, message: str = None):
        self.filter_param = filter_param
        self.value = value
        if message is None:
            message = f"Invalid filter parameter '{filter_param}': {value}"
        super().__init__(message)


class UploadError(VectorDBError):
    """Error raised during vector upload operations"""
    pass

load_dotenv()



class VectorDB:
    def __init__(self, index_name="sec-embeddings"):
        self.client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.client.Index(index_name)

    def retrieve_from_id(self, vector_id: str):
        """Retrieve a vector by its ID"""
        if not vector_id or not isinstance(vector_id, str):
            raise VectorNotFoundError(vector_id)
            
        response = self.index.fetch(ids=[vector_id])
        vector = response.vectors.get(vector_id)
        
        if vector is None:
            raise VectorNotFoundError(vector_id)
            
        return vector
    
    
    def retrieve_by_metadata(
        self,
        vector: list[float],
        *,
        entity_type: EntityType | None = None,
        ticker: str | None = None,
        year: str | None = None,
        top_k: int = 10
    ):
        """Search only vectors with metadata.entity_type == entity_type (and optional ticker)."""
        
        if not vector or not isinstance(vector, list):
            raise InvalidFilterError("vector", vector, "Vector must be a non-empty list of floats")
            
        if top_k <= 0:
            raise InvalidFilterError("top_k", top_k, "top_k must be a positive integer")
        
        filt = {}
        if entity_type:
            if not isinstance(entity_type, EntityType):
                raise MetadataFieldError("entity_type", f"entity_type must be an EntityType, got {type(entity_type)}")
            filt["entity_type"] = {"$eq": entity_type.value}
            
        if ticker:
            if not isinstance(ticker, str) or not ticker.strip():
                raise MetadataFieldError("ticker", "ticker must be a non-empty string")
            filt["ticker"] = {"$eq": ticker.strip().upper()}
            
        if year:
            try:
                year_int = int(year)
                if year_int < 1900 or year_int > 2100:
                    raise InvalidFilterError("year", year, "Year must be between 1900 and 2100")
                filt["year"] = {"$eq": year_int}
            except (ValueError, TypeError):
                raise InvalidFilterError("year", year, "Year must be a valid integer")

        try:
            return self.index.query(
                vector=vector,
                top_k=top_k,
                include_metadata=True,
                filter=filt
            )
        except Exception as e:
            raise VectorDBError(f"Query failed: {str(e)}")


    def upload(self, chunk: DocumentChunk):
        """Uploads a DocumentChunk directly to the vector database"""
        
        if not isinstance(chunk, DocumentChunk):
            raise UploadError("chunk must be a DocumentChunk instance")
            
        if not chunk.id:
            raise MetadataFieldError("id", "DocumentChunk must have a valid id")
            
        if not chunk.embeddding:
            raise MetadataFieldError("embeddings", "DocumentChunk must have embeddings")
            
        if not isinstance(chunk.embeddding, list):
            raise MetadataFieldError("embeddings", "embeddings must be a list of floats")

        try:

            metadata = {
                "entity_type": chunk.type_chunk,
                "text_preview": chunk.text[:100] if chunk.text else "",
            }
            
            if chunk.document:
                if hasattr(chunk.document, 'ticker') and chunk.document.ticker:
                    metadata["ticker"] = str(chunk.document.ticker).upper()
                if hasattr(chunk.document, 'year') and chunk.document.year:
                    metadata["year"] = int(chunk.document.year) if isinstance(chunk.document.year, str) and chunk.document.year.isdigit() else chunk.document.year
                if hasattr(chunk.document, 'form_type') and chunk.document.form_type:
                    metadata["form_type"] = str(chunk.document.form_type)

                metadata["document_id"] = chunk.document.id

            # Prepare vector for upsert
            vector_data = {
                "id": chunk.id,
                "values": chunk.embeddding,
                "metadata": metadata
            }
            
            return self.index.upsert(vectors=[vector_data])
        except Exception as e:
            raise UploadError(f"Failed to upload chunk: {str(e)}")
    
    
    def query(self, vector, top_k=10, include_metadata=True):
        """Query the index for similar vectors"""
        if not vector or not isinstance(vector, list):
            raise InvalidFilterError("vector", vector, "Vector must be a non-empty list of floats")
            
        if top_k <= 0:
            raise InvalidFilterError("top_k", top_k, "top_k must be a positive integer")
            
        try:
            return self.index.query(
                vector=vector,
                top_k=top_k,
                include_metadata=include_metadata
            )
        except Exception as e:
            raise VectorDBError(f"Query failed: {str(e)}")
    
    def delete(self, ids: List[str]):
        """Delete vectors by IDs"""
        if not ids or not isinstance(ids, list):
            raise InvalidFilterError("ids", ids, "ids must be a non-empty list of strings")
            
        for vector_id in ids:
            if not isinstance(vector_id, str) or not vector_id.strip():
                raise InvalidFilterError("id", vector_id, "Each id must be a non-empty string")
                
        try:
            return self.index.delete(ids=ids)
        except Exception as e:
            raise VectorDBError(f"Delete failed: {str(e)}")
    
    def clear(self):
        """Completely wipe all vectors from the index"""
        try:
            return self.index.delete(delete_all=True)
        except Exception as e:
            raise VectorDBError(f"Clear operation failed: {str(e)}")