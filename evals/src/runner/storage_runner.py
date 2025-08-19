from typing import List, Dict, Any
from pathlib import Path

from utils import logger
from utils.typing import DocumentChunk, SECDocument
from evals.src.config_types import Storage
from mcp_server.clients.vector_db import VectorDB
from mcp_server.clients.sql_db import SQLClient


class StorageRunner:
    """Runner for storing chunks in VectorDB and SQLite following existing patterns"""
    
    def __init__(self):
        self.vector_db = None
        self.sql_client = None
        
        self.storage_map = {
            "vector": {
                "pinecone"
            },
            "text" : {
                "sqlite"
            },
            "file" : {
                "local"
            }
        }
    
    async def run_storage(
            self, 
            chunks: List[DocumentChunk],
            documents: List[SECDocument], 
            storage_config: Storage
        ) -> Dict[str, Any]:
        """
        Store chunks based on storage configuration
        
        Args:
            chunks: List of DocumentChunk objects with embeddings
            documents: List of SECDocument objects
            storage_config: Storage configuration from YAML
            
        Returns:
            Dictionary with storage results and metadata
        """
        logger.info(f"Starting storage for {len(chunks)} chunks")
        
        results = {
            "total_chunks": len(chunks),
            "stored_vector": 0,
            "stored_text": 0,
            "errors": []
        }
        
        # Initialize storage clients based on config
        await self._initialize_storage(storage_config)
        
        # Filter chunks with embeddings for vector storage
        embedded_chunks = [chunk for chunk in chunks if chunk.embeddding is not None]
        logger.info(f"Found {len(embedded_chunks)} chunks with embeddings for vector storage")
        
        # Clear vector database if configured
        vector_config = storage_config.vector
        if vector_config and vector_config.clear and self.vector_db:
            try:
                logger.info("Clearing existing vectors from VectorDB")
                self.vector_db.clear()
                logger.info("VectorDB cleared successfully")
            except Exception as e:
                error_msg = f"Failed to clear VectorDB: {str(e)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)

        # Store in vector database if configured and upload enabled
        if vector_config and vector_config.upload and self.vector_db and embedded_chunks:
            vector_results = await self._store_in_vector_db(embedded_chunks, storage_config)
            results["stored_vector"] = vector_results["stored"]
            results["errors"].extend(vector_results["errors"])
        
        # Store in SQLite if text_store is configured
        text_store = storage_config.text_store
        if text_store and text_store == "sqlite" and self.sql_client:
            sql_results = await self._store_in_sqlite(chunks, storage_config)
            results["stored_text"] = sql_results["stored"]
            results["errors"].extend(sql_results["errors"])
        
        # Generate outputs if configured
        outputs = storage_config.outputs
        if outputs:
            await self._generate_outputs(chunks, documents, outputs, storage_config)
        
        logger.info(f"Storage complete: {results['stored_vector']} vectors, {results['stored_text']} text chunks")
        if results["errors"]:
            logger.warning(f"Encountered {len(results['errors'])} errors during storage")
        
        return results
    
    async def _initialize_storage(self, storage_config: Storage):
        """Initialize storage clients based on configuration"""
        
        # Initialize vector DB if upload is enabled
        vector_config = storage_config.vector
        if vector_config and vector_config.upload:
            try:
                # Get index name from config or use default
                index_name = (vector_config.index_name or vector_config.index or 
                             (storage_config.vector_db.get("index_name") if storage_config.vector_db else None) or 
                             "sec-embeddings")
                self.vector_db = VectorDB(index_name=index_name)
                logger.info(f"Initialized VectorDB with index: {index_name}")
            except Exception as e:
                logger.error(f"Failed to initialize VectorDB: {e}")
                self.vector_db = None
        
        # Initialize SQLite if text_store is sqlite
        if storage_config.text_store and storage_config.text_store == "sqlite":
            try:
                db_path = storage_config.sqlite_path or ".sql/chunks.db"
                self.sql_client = SQLClient(db_path=db_path)
                logger.info(f"Initialized SQLite client with path: {db_path}")
            except Exception as e:
                logger.error(f"Failed to initialize SQLite: {e}")
                self.sql_client = None
    
    async def _store_in_vector_db(self, chunks: List[DocumentChunk], storage_config: Storage) -> Dict[str, Any]:
        """Store chunks in vector database"""
        logger.info(f"Storing {len(chunks)} chunks in VectorDB")
        
        results = {"stored": 0, "errors": []}
        
        for chunk in chunks:
            try:
                # Upload DocumentChunk directly to vector database
                self.vector_db.upload(chunk)
                results["stored"] += 1
                
                if results["stored"] % 100 == 0:
                    logger.info(f"Stored {results['stored']}/{len(chunks)} vectors")
                    
            except Exception as e:
                error_msg = f"Failed to store chunk {chunk.id}: {str(e)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
        
        return results
    
    async def _store_in_sqlite(self, chunks: List[DocumentChunk], storage_config: Storage) -> Dict[str, Any]:
        """Store chunks in SQLite database"""
        logger.info(f"Storing {len(chunks)} chunks in SQLite")
        
        results = {"stored": 0, "errors": []}
        
        for chunk in chunks:
            try:
                # Convert chunk to document data
                doc_data = {
                    "text": chunk.text,
                    "type_chunk": chunk.type_chunk,
                    "document_id": chunk.id if chunk.id else None,
                    "embedding_dims": len(chunk.embeddding) if chunk.embeddding else 0
                }
                
                # Store in SQLite
                success = self.sql_client.store_document(chunk.id, doc_data)
                if success:
                    results["stored"] += 1
                    
                if results["stored"] % 100 == 0:
                    logger.info(f"Stored {results['stored']}/{len(chunks)} text chunks")
                    
            except Exception as e:
                error_msg = f"Failed to store chunk {chunk.id} in SQLite: {str(e)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
        
        return results
    
    
    async def _generate_outputs(self, chunks: List[DocumentChunk], documents: List[SECDocument], outputs: List[str], storage_config: Storage):
        """Generate output files based on configuration"""
        logger.info(f"Generating outputs: {outputs}")
        
        # Get output directory from report config or use default
        output_dir = Path("evals/reports/outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            if "chunks" in outputs:
                await self._export_chunks(chunks, output_dir)
            
            if "documents" in outputs:
                await self._export_documents(documents, output_dir)
            
        except Exception as e:
            logger.error(f"Failed to generate outputs: {e}")
    
    async def _export_chunks(self, chunks: List[DocumentChunk], output_dir: Path):
        """Export chunks as JSON objects"""
        import json
        
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
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "chunks_objects.json"
        with open(output_file, 'w') as f:
            json.dump(objects_data, f, indent=2)
        
        logger.info(f"Exported {len(objects_data)} chunk objects to {output_file}")

    async def _export_documents(self, documents: List[SECDocument], output_dir: Path):
        """Export documents as JSON objects"""
        import json
        
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
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "documents.json"
        with open(output_file, 'w') as f:
            json.dump(documents_data, f, indent=2)
        
        logger.info(f"Exported {len(documents_data)} documents to {output_file}")
