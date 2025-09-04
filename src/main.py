import argparse
import asyncio
from pathlib import Path
from typing import List

from utils import logger
from utils.types import DocumentChunk, YamlConfig

from runner import EmbedderRunner, StorageRunner, ReportRunner

def load_yaml_config(config_path: str) -> List[YamlConfig]:
    """Load YAML configuration file and return list of configs."""
    return [YamlConfig.from_yaml(config_path)]




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
        configs: List[YamlConfig]  = load_yaml_config(args.config)

    except ValueError as e:
        logger.error(f"Error parsing YAML file: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))