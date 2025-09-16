
import json
from typing import Optional, List, Dict, Any
from pathlib import Path

from .base import TextStorageBase, TextStorageError
from models.documents import DocumentChunk
from models.configs.storage import TextStoreConfig


class FileStoreError(TextStorageError):
    """File store-specific exception for operations"""
    pass


class FileStore(TextStorageBase):
    """Local file system storage implementation"""
    
    def __init__(self, config: TextStoreConfig):
        """Initialize file storage with base path"""
        super().__init__(config)
        self.base_path = Path(self.config.base_path, 'data/documents')
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    @property
    def provider_name(self) -> str:
        return "file"

    def _get_file_path(self, doc_id: str) -> Path:
        """Get file path for document ID"""
        return self.base_path / f"{doc_id}.json"

    def store_document(self, doc_id: str, doc_data: Dict[str, Any]) -> bool:
        """Store document data as JSON file"""
        try:
            file_path = self._get_file_path(doc_id)
            document = {
                'id': doc_id,
                'text': doc_data.get('text'),
                'created_at': doc_data.get('created_at')
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(document, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            raise FileStoreError(f"Failed to store document {doc_id}: {str(e)}")

    def store_document_chunk(self, chunk: DocumentChunk) -> bool:
        """Store DocumentChunk as JSON file"""
        try:
            file_path = self._get_file_path(chunk.id)
            chunk_data = {
                'id': chunk.id,
                'text': chunk.text,
                'document': {
                    'id': chunk.document.id,
                    'name': chunk.document.name,
                    'path': chunk.document.path,
                    'text': chunk.document.text
                },
                'embedding': chunk.embedding
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(chunk_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            raise FileStoreError(f"Failed to store document chunk {chunk.id}: {str(e)}")

    def retrieve_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve document by ID"""
        try:
            file_path = self._get_file_path(doc_id)
            if not file_path.exists():
                return None
                
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise FileStoreError(f"Failed to retrieve document {doc_id}: {str(e)}")

    def retrieve_documents(self, doc_ids: List[str]) -> List[Dict[str, Any]]:
        """Retrieve multiple documents by IDs"""
        if not doc_ids:
            return []
            
        documents = []
        for doc_id in doc_ids:
            doc = self.retrieve_document(doc_id)
            if doc:
                documents.append(doc)
        return documents

    def delete_document(self, doc_id: str) -> bool:
        """Delete document by ID"""
        try:
            file_path = self._get_file_path(doc_id)
            if file_path.exists():
                file_path.unlink()
                return True
            return False
        except Exception as e:
            raise FileStoreError(f"Failed to delete document {doc_id}: {str(e)}")

    def get_document_count(self) -> int:
        """Get total number of documents"""
        try:
            json_files = list(self.base_path.glob('*.json'))
            return len(json_files)
        except Exception as e:
            raise FileStoreError(f"Failed to get document count: {str(e)}")

    def clear_all(self) -> bool:
        """Clear all documents from storage"""
        try:
            for json_file in self.base_path.glob('*.json'):
                json_file.unlink()
            return True
        except Exception as e:
            raise FileStoreError(f"Failed to clear documents: {str(e)}")