import argparse
import yaml
import asyncio
from pathlib import Path
from typing import Dict, List, Any

from utils import logger
from utils.typing import DocumentChunk

from loader.data_loader import DataLoader
from runner import ProcessorRunner, EmbedderRunner

def load_yaml_config(config_path: str) -> List[Dict[str, Any]]:
    """Load YAML configuration file and return list of configs."""
    with open(config_path, 'r') as file:
        configs = list(yaml.safe_load_all(file))
    return configs




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
            logger.info(f"\n{'='*50}")
            logger.info(f"Config {i + 1}")
            logger.info(f"{'='*50}")
            logger.info(f"  Task: {config.get('task', 'Unknown')}")
            logger.info(f"  Dataset Path: {config.get('dataset_path', 'Unknown')}")
            logger.info(f"  Processors: {config.get('processors', [])}")
            logger.info(f"  Output Path: {config.get('output_path', 'Unknown')}")
            
        # Load documents using DataLoader
        dataset_path = config.get('dataset_path', 'data/')
        logger.info(f"Loading documents from: {dataset_path}")
        
        # Create and initialize DataLoader
        dataloader = await DataLoader.create(path=dataset_path)
        documents = dataloader.documents
        logger.info(f"Loaded {len(documents)} documents")

        # TODO: metadata_from SEC 

        # Run processors based on config
        processors_config = config.get('processors', ["breaks"])
        chunks: List[DocumentChunk] = []
        if processors_config:
            runner = ProcessorRunner()
            chunks = await runner.run_processors(documents, processors_config, config)
            logger.info(f"Generated {len(chunks)} chunks from {len(processors_config)} processors")
            

            logger.info(f"Total Chunks: {len(chunks) - 3}")
        
        # Run embedding based on config
        embedding_config = config.get('embedding')
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


    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))