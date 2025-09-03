
import json
from typing import List
from pathlib import Path

from .base import FileStorageBase, FileStorageError
from src.utils.types import DocumentChunk, SECDocument
from src.utils import logger


class LocalFileStore(FileStorageBase):
    """Local file system storage implementation"""
    
    def __init__(self, base_path: str = "evals/reports/outputs"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    
    def export_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """Export chunks as JSON objects"""
        try:
            objects_data = []
            for chunk in chunks:
                obj = {
                    "id": chunk.id,
                    "text": chunk.text,
                    "type_chunk": chunk.type_chunk,
                    "document": chunk.document.path,
                    "embeddings": [] if chunk.embeddding else None,
                    "date": chunk.document.date,
                    "ticker": chunk.document.ticker
                }
                objects_data.append(obj)
            
            self.base_path.mkdir(parents=True, exist_ok=True)
            output_file = self.base_path / "chunks_objects.json"
            with open(output_file, 'w') as f:
                json.dump(objects_data, f, indent=2)
            
            logger.info(f"Exported {len(objects_data)} chunk objects to {output_file}")
            return True
        except Exception as e:
            raise FileStorageError(f"Failed to export chunks: {str(e)}")

    def export_documents(self, documents: List[SECDocument]) -> bool:
        """Export documents as JSON objects"""
        try:
            documents_data = []
            for doc in documents:
                doc_obj = {
                    "ticker": doc.ticker,
                    "company": doc.company,
                    "year": doc.year,
                    "date": doc.date,
                    "path": doc.path,
                    "form_type": doc.form_type,
                    "text": doc.text[:100] + "...",
                    "sec_data": doc.sec_data,
                    "sec_metadata": doc.sec_metadata.model_dump() if doc.sec_metadata else None
                }
                documents_data.append(doc_obj)
            
            self.base_path.mkdir(parents=True, exist_ok=True)
            output_file = self.base_path / "documents.json"
            with open(output_file, 'w') as f:
                json.dump(documents_data, f, indent=2)
            
            logger.info(f"Exported {len(documents_data)} documents to {output_file}")
            return True
        except Exception as e:
            raise FileStorageError(f"Failed to export documents: {str(e)}")