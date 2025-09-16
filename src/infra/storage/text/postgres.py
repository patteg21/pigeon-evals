import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
import json

from .base import TextStorageBase, TextStorageError
from models.documents import DocumentChunk
from models.configs.storage import TextStoreConfig


class PostgresError(TextStorageError):
    """PostgreSQL-specific exception for operations"""
    pass


class PostgresDB(TextStorageBase):
    def __init__(self, config: TextStoreConfig):
        """Initialize PostgreSQL client with connection parameters"""
        super().__init__(config)
        self.connection_params = {
            'host': self.config.host or  'localhost',
            'port': self.config.port or  5432,
            'database': self.config.database or 'pigeon_evals',
            'user': self.config.user or 'postgres',
            'password': self.config.password or  ''
        }
        self._initialize_db()
    
    @property
    def provider_name(self) -> str:
        return "postgres"

    def _initialize_db(self):
        """Initialize database with required tables"""
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id TEXT PRIMARY KEY,
                        text TEXT NOT NULL,
                        document_data JSONB,
                        embedding JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_id ON documents(id)")
                conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = psycopg2.connect(**self.connection_params)
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise PostgresError(f"Database operation failed: {str(e)}")
        finally:
            if conn:
                conn.close()

    def store_document(self, doc_id: str, doc_data: Dict[str, Any]) -> bool:
        """Store document data in PostgreSQL database"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO documents (id, text, document_data, embedding) VALUES (%s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET 
                            text = EXCLUDED.text,
                            document_data = EXCLUDED.document_data,
                            embedding = EXCLUDED.embedding
                    """, (doc_id, doc_data.get('text'), json.dumps(doc_data.get('document_data')), json.dumps(doc_data.get('embedding'))))
                    conn.commit()
                    return True
        except Exception as e:
            raise PostgresError(f"Failed to store document {doc_id}: {str(e)}")

    def store_document_chunk(self, chunk: DocumentChunk) -> bool:
        """Store DocumentChunk in PostgreSQL database"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    document_data = {
                        'id': chunk.document.id,
                        'name': chunk.document.name,
                        'path': chunk.document.path,
                        'text': chunk.document.text
                    }
                    
                    cursor.execute("""
                        INSERT INTO documents (id, text, document_data, embedding) VALUES (%s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET 
                            text = EXCLUDED.text,
                            document_data = EXCLUDED.document_data,
                            embedding = EXCLUDED.embedding
                    """, (chunk.id, chunk.text, json.dumps(document_data), json.dumps(chunk.embedding)))
                    conn.commit()
                    return True
        except Exception as e:
            raise PostgresError(f"Failed to store document chunk {chunk.id}: {str(e)}")

    def retrieve_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve document by ID"""
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("SELECT * FROM documents WHERE id = %s", (doc_id,))
                    row = cursor.fetchone()
                    return dict(row) if row else None
        except Exception as e:
            raise PostgresError(f"Failed to retrieve document {doc_id}: {str(e)}")

    def retrieve_documents(self, doc_ids: List[str]) -> List[Dict[str, Any]]:
        """Retrieve multiple documents by IDs"""
        if not doc_ids:
            return []
            
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("SELECT * FROM documents WHERE id = ANY(%s)", (doc_ids,))
                    rows = cursor.fetchall()
                    return [dict(row) for row in rows]
        except Exception as e:
            raise PostgresError(f"Failed to retrieve documents: {str(e)}")

    def delete_document(self, doc_id: str) -> bool:
        """Delete document by ID"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
                    conn.commit()
                    return cursor.rowcount > 0
        except Exception as e:
            raise PostgresError(f"Failed to delete document {doc_id}: {str(e)}")

    def get_document_count(self) -> int:
        """Get total number of documents"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT COUNT(*) FROM documents")
                    return cursor.fetchone()[0]
        except Exception as e:
            raise PostgresError(f"Failed to get document count: {str(e)}")

    def clear_all(self) -> bool:
        """Clear all documents from database"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("DELETE FROM documents")
                    conn.commit()
                    return True
        except Exception as e:
            raise PostgresError(f"Failed to clear documents: {str(e)}")