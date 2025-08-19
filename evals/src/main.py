import argparse
import asyncio
from pathlib import Path
from typing import List

from utils import logger
from utils.typing import DocumentChunk
from utils.typing.evals.config import Config

from loader.data_loader import DataLoader
from runner import ProcessorRunner, EmbedderRunner, StorageRunner
from parser.parser import SECDataParser

def load_yaml_config(config_path: str) -> List[Config]:
    """Load YAML configuration file and return list of configs."""
    return [Config.from_yaml(config_path)]




async def main():
    parser = argparse.ArgumentParser(description='Run evaluation with YAML configuration')
    parser.add_argument('--config', '-c', type=str, required=True,
                       help='Path to YAML configuration file')
    
    args = parser.parse_args()
    
    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Error: Configuration file {args.config} not found")
        return 1
    
    # Load configuration
    try:
        configs = load_yaml_config(args.config)
        logger.info(f"Loaded {len(configs)} configuration(s) from {args.config}")
        
        for i, config in enumerate(configs):
            logger.info(f"RUN ID: {config.run_id}")
            logger.info(f"\n{'='*50}")
            logger.info(f"Config {i + 1}")
            logger.info(f"{'='*50}")
            logger.info(f"  Task: {config.task}")
            logger.info(f"  Dataset Path: {config.dataset_path}")
            logger.info(f"  Processors: {config.processors}")
            logger.info(f"  Output Path: {config.report.output_path if config.report else 'Not configured'}")
            
        # Load documents using DataLoader
        dataset_path = config.dataset_path
        logger.info(f"Loading documents from: {dataset_path}")
        
        # Create and initialize DataLoader
        dataloader = await DataLoader.create(path=dataset_path)
        documents = dataloader.documents
        logger.info(f"Loaded {len(documents)} documents")

        # Parse SEC metadata if configured
        sec_metadata_config = config.sec_metadata
        if sec_metadata_config:
            logger.info(f"Parsing SEC metadata with config: {sec_metadata_config}")
            parser = SECDataParser()
            for document in documents:
                parser.process(document)
            

        # Run processors based on config
        processors_config = config.processors
        chunks: List[DocumentChunk] = []
        if processors_config:
            runner = ProcessorRunner()
            chunks = await runner.run_processors(documents, processors_config, config)
            logger.info(f"Generated {len(chunks)} chunks from {len(processors_config)} processors")
            

            logger.info(f"Total Chunks: {len(chunks) - 3}")
        
        # Run embedding based on config
        embedding_config = config.embedding
        embedded_chunks = chunks
        if embedding_config and chunks:
            embedder_runner = EmbedderRunner()
            embedded_chunks = await embedder_runner.run_embedder(chunks, embedding_config)
            logger.info(f"Embedded {len(embedded_chunks)} chunks")
            
            # Show sample embedded chunks
            for i, chunk in enumerate(embedded_chunks[:3]):
                embedding_len = len(chunk.embeddding) if chunk.embeddding else 0
                logger.info(f"  Embedded Chunk {i+1}: {chunk.type_chunk} - {embedding_len} dimensions")
            if len(embedded_chunks) > 3:
                logger.info(f"  ... and {len(embedded_chunks) - 3} more embedded chunks")

        # Run storage based on config
        storage_config = config.storage
        if storage_config and embedded_chunks:
            storage_runner = StorageRunner()
            storage_results = await storage_runner.run_storage(embedded_chunks, documents, storage_config)
            logger.info(f"Storage complete: {storage_results['stored_vector']} vectors, {storage_results['stored_text']} text chunks")
            
            if storage_results['errors']:
                logger.warning(f"Storage encountered {len(storage_results['errors'])} errors:")
                for error in storage_results['errors'][:5]:  # Show first 5 errors
                    logger.warning(f"  - {error}")


    except ValueError as e:
        logger.error(f"Error parsing YAML file: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))