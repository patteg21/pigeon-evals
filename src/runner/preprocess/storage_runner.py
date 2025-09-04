from typing import List, Dict, Any


from utils import logger
from utils.types import DocumentChunk, Document, Storage

from storage.vector import VectorStorageBase, PineconeDB
from storage.text import TextStorageBase, SQLiteDB
from storage.file_store import FileStorageBase, LocalFileStore 


class StorageRunner:
    """Runner for storing chunks in VectorDB and SQLite following existing patterns"""
    
    def __init__(self):
        pass
    
    async def run_storage(
            self, 
        ):

        pass