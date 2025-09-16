from typing import List

from tqdm import tqdm

from models import DocumentChunk
from utils.dry_run import dry_response
from utils.logger import logger


from infra.storage.text import TextStorageFactory
from infra.storage.vector import VectorStorageFactory
from runner.base import Runner


class StorageRunner(Runner):

    def __init__(self):
        super().__init__()
        self.text_storage = TextStorageFactory.create_from_config()
        self.vector_storage = VectorStorageFactory.create_from_config()
    
    async def run(
            self,
            chunks: List[DocumentChunk]
        ) -> List[DocumentChunk]:
        """Store document chunks in both text and vector storage."""

        logger.info(f"Storing {len(chunks)} chunks in text and vector storage")

        # Store in text storage
        if self.text_storage and self.text_storage.config.upload:
            logger.info(f"Storing chunks in {self.text_storage.provider_name} text storage")
            for chunk in tqdm(chunks, desc="Text storage", unit="chunk"):
                success = self.text_storage.store_document_chunk(chunk)
                if not success:
                    logger.warning(f"Failed to store chunk {chunk.id} in text storage")

        # Store in vector storage
        if self.vector_storage and self.vector_storage.config.upload:
            logger.info(f"Storing chunks in {self.vector_storage.provider_name} vector storage")
            for chunk in tqdm(chunks, desc="Vector storage", unit="chunk"):
                try:
                    self.vector_storage.upload(chunk)
                except Exception as e:
                    logger.warning(f"Failed to store chunk {chunk.id} in vector storage: {e}")
                    logger.exception(f"Full traceback for chunk {chunk.id}:")

        logger.info("Storage operations completed")
        return chunks
