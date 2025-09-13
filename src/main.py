import argparse
import asyncio
from pathlib import Path
from typing import List

from utils import logger, DataLoader
from models import YamlConfig, Document, DocumentChunk
from runner import EmebeddingRunner, StorageRunner, ReportRunner, ParserRunner

def load_yaml_config(config_path: str) -> List[YamlConfig]:
    """Load YAML configuration file and return list of configs."""
    return [YamlConfig.from_yaml(config_path)]




async def main():
    parser = argparse.ArgumentParser(description='Run evaluation with YAML configuration')
    parser.add_argument('--config', '-c', type=str, required=False, default="configs/test.yml",
                       help='Path to YAML configuration file (default: configs/test.yml)')
    
    args = parser.parse_args()
    
    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Error: Configuration file {args.config} not found")
        return 1
    
    # Load configuration
    try:
        configs: List[YamlConfig] = load_yaml_config(args.config)
        
        # Process each configuration
        for config in configs:

            loader = DataLoader(config.dataset)
            documents: List[Document] = loader.load()

            logger.info(f"Processing configuration: {config.task}")


            if config.parser:
                logger.info("Parsing Data... ")
                parser_runner = ParserRunner(config.parser)
                document_chunks: List[DocumentChunk] = await parser_runner.run(documents)
                print(len(document_chunks))

            if config.embedding:            
                embedding_runner = EmebeddingRunner(config.embedding)
                await embedding_runner.run()


            if config.storage:
                
                if config.storage.text_store:
                    logger.info("Text Storing Data...")

                    pass

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