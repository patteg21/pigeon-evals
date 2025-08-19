
import sqlite3
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from pathlib import Path


class SQLiteError(Exception):
    """Base exception for SQLite operations"""
    pass


class SQLClient:
    def __init__(self, db_path: str = ".sql/chunks.db"):
        """Initialize SQLite client with database path"""
        self.db_path = db_path
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize database with required tables"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Create documents table to store text content
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
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
                    INSERT OR REPLACE INTO documents (id, text) VALUES (?, ?)
                """, (doc_id, doc_data.get('text')))
                
                conn.commit()
                return True
                
        except Exception as e:
            raise SQLiteError(f"Failed to store document {doc_id}: {str(e)}")
    
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