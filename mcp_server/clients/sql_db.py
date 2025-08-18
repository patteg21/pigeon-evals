
import sqlite3
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from pathlib import Path


class SQLiteError(Exception):
    """Base exception for SQLite operations"""
    pass


class SQLClient:
    def __init__(self, db_path: str = "data/documents.db"):
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
                    ticker TEXT NOT NULL,
                    date TEXT NOT NULL,
                    form_type TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    text TEXT NOT NULL,
                    title TEXT,
                    section TEXT,
                    subsection TEXT,
                    page_number INTEGER,
                    document_path TEXT,
                    commission_number TEXT,
                    period_end TEXT,
                    prev_chunk_id TEXT,
                    next_chunk_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for faster retrieval
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_id ON documents(id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_ticker ON documents(ticker)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_date ON documents(date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_entity_type ON documents(entity_type)")
            
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
                    INSERT OR REPLACE INTO documents (
                        id, ticker, date, form_type, entity_type, text,
                        title, section, subsection, page_number, document_path,
                        commission_number, period_end, prev_chunk_id, next_chunk_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    doc_id,
                    doc_data.get('ticker'),
                    doc_data.get('date'),
                    doc_data.get('form_type'),
                    doc_data.get('entity_type'),
                    doc_data.get('text'),
                    doc_data.get('title'),
                    doc_data.get('section'),
                    doc_data.get('subsection'),
                    doc_data.get('page_number'),
                    doc_data.get('document_path'),
                    doc_data.get('commission_number'),
                    doc_data.get('period_end'),
                    doc_data.get('prev_chunk_id'),
                    doc_data.get('next_chunk_id')
                ))
                
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
    
    def search_documents(
        self, 
        ticker: Optional[str] = None,
        date: Optional[str] = None,
        form_type: Optional[str] = None,
        entity_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search documents by metadata"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                conditions = []
                params = []
                
                if ticker:
                    conditions.append("ticker = ?")
                    params.append(ticker)
                if date:
                    conditions.append("date = ?")
                    params.append(date)
                if form_type:
                    conditions.append("form_type = ?")
                    params.append(form_type)
                if entity_type:
                    conditions.append("entity_type = ?")
                    params.append(entity_type)
                
                where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
                query = f"SELECT * FROM documents{where_clause} ORDER BY created_at DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            raise SQLiteError(f"Failed to search documents: {str(e)}")
    
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