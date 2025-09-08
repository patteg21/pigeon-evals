
import sqlite3
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from pathlib import Path
import json

from .base import TextStorageBase, TextStorageError
from models.documents import DocumentChunk
from models.configs.storage import SqliteConfig


class SQLiteError(TextStorageError):
    """SQLite-specific exception for operations"""
    pass


class SQLiteDB(TextStorageBase):
    def __init__(self, config: SqliteConfig):
        """Initialize SQLite client with database path"""
        super().__init__(config)
        self.db_path = self.config.path or "data/.sql/chunks.db"
        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize_db()
    
    @property
    def provider_name(self) -> str:
        return "sqlite"
    

    def _initialize_db(self):
        """Initialize database with required tables"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Create documents table to store text content
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    document_data TEXT,
                    embedding TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for faster retrieval
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_id ON documents(id)")
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access to rows
        try:
            yield conn
        except Exception as e:
            conn.rollback()
            raise SQLiteError(f"Database operation failed: {str(e)}")
        finally:
            conn.close()
    
    def store_document(self, doc_id: str, doc_data: Dict[str, Any]) -> bool:
        """Store document data in SQLite database"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO documents (id, text, document_data, embedding) VALUES (?, ?, ?, ?)
                """, (doc_id, doc_data.get('text'), json.dumps(doc_data.get('document_data')), json.dumps(doc_data.get('embedding'))))
                
                conn.commit()
                return True
                
        except Exception as e:
            raise SQLiteError(f"Failed to store document {doc_id}: {str(e)}")

    def store_document_chunk(self, chunk: DocumentChunk) -> bool:
        """Store DocumentChunk in SQLite database"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                document_data = {
                    'id': chunk.document.id,
                    'name': chunk.document.name,
                    'path': chunk.document.path,
                    'text': chunk.document.text
                }
                
                cursor.execute("""
                    INSERT OR REPLACE INTO documents (id, text, document_data, embedding) VALUES (?, ?, ?, ?)
                """, (chunk.id, chunk.text, json.dumps(document_data), json.dumps(chunk.embeddding)))
                
                conn.commit()
                return True
                
        except Exception as e:
            raise SQLiteError(f"Failed to store document chunk {chunk.id}: {str(e)}")
    

    def retrieve_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve document by ID"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
                row = cursor.fetchone()
                
                if row:
                    return dict(row)
                return None
                
        except Exception as e:
            raise SQLiteError(f"Failed to retrieve document {doc_id}: {str(e)}")
    

    def retrieve_documents(self, doc_ids: List[str]) -> List[Dict[str, Any]]:
        """Retrieve multiple documents by IDs"""
        if not doc_ids:
            return []
            
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                placeholders = ','.join(['?'] * len(doc_ids))
                cursor.execute(f"SELECT * FROM documents WHERE id IN ({placeholders})", doc_ids)
                rows = cursor.fetchall()
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            raise SQLiteError(f"Failed to retrieve documents: {str(e)}")
    
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete document by ID"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
                conn.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            raise SQLiteError(f"Failed to delete document {doc_id}: {str(e)}")
    

    def get_document_count(self) -> int:
        """Get total number of documents"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM documents")
                return cursor.fetchone()[0]
                
        except Exception as e:
            raise SQLiteError(f"Failed to get document count: {str(e)}")
    

    def clear_all(self) -> bool:
        """Clear all documents from database"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM documents")
                conn.commit()
                return True
                
        except Exception as e:
            raise SQLiteError(f"Failed to clear documents: {str(e)}")