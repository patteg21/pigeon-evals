import argparse
import asyncio
from pathlib import Path
from typing import List

from utils import logger, DataLoader
from utils.config_manager import ConfigManager
from models import YamlConfig, Document, DocumentChunk
from runner import EmbeddingRunner, StorageRunner, ReportRunner, ParserRunner
from utils.dry_run import set_dry_run_mode

def load_yaml_config(config_path: str) -> List[YamlConfig]:
    """Load YAML configuration file and return list of configs."""
    return [YamlConfig.from_yaml(config_path)]




async def main():
    parser = argparse.ArgumentParser(description='Run evaluation with YAML configuration')
    parser.add_argument('--config', '-c', type=str, required=False, default="configs/test.yml",
                       help='Path to YAML configuration file (default: configs/test.yml)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Run in dry mode with mock responses (no actual embedding/storage calls)')
    
    args = parser.parse_args()
    
    # Set dry run mode if specified
    if args.dry_run:
        set_dry_run_mode(True)
        logger.info("Running in DRY RUN mode - using mock responses")
    
    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Error: Configuration file {args.config} not found")
        return 1
    
    # Load configuration
    try:
        # Initialize the config singleton
        config_manager = ConfigManager()
        config_manager.load_config(args.config)
        config = config_manager.config

        # Process the configuration
        loader = DataLoader(config.dataset)
        documents: List[Document] = loader.load()

        logger.info(f"Processing configuration: {config.task}")

        if config.parser:
            logger.info("Parsing Data... ")
            parser_runner = ParserRunner(config.parser)
            document_chunks: List[DocumentChunk] = await parser_runner.run(documents)
            print(len(document_chunks))

        if config.embedding:
            embedding_runner = EmbeddingRunner()
            embedded_chunks: List[DocumentChunk] = await embedding_runner.run(document_chunks)

        if config.storage:
            if config.storage.text_store:
                logger.info("Text Storing Data...")
                storage_runner = StorageRunner()
                await storage_runner.run(embedded_chunks)

            if config.storage.vector:
                logger.info("Vector Storing Data...")
                pass

        if config.eval:
            logger.info("Evaluating Data...")
            pass
            
    except ValueError as e:
        logger.error(f"Error parsing YAML file: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))