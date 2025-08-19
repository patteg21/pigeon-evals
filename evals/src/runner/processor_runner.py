from typing import List, Dict, Any
from utils import logger
from utils.typing import SECDocument
from utils.typing.chunks import DocumentChunk
from ..processor import TOCProcessor, TablesProcessor, BreaksProcessor


class ProcessorRunner:
    """Runner for executing document processors."""
    
    def __init__(self):
        self.processor_map = {
            "toc": TOCProcessor,
            "tables": TablesProcessor, 
            "breaks": BreaksProcessor
        }
    
    async def run_processors(self, documents: List[SECDocument], processor_names: List[str], config: Dict[str, Any] = None) -> List[DocumentChunk]:
        """Run the specified processors on documents and return all chunks."""
        config = config or {}
        all_chunks = []
        
        for processor_name in processor_names:
            if processor_name not in self.processor_map:
                logger.warning(f"Unknown processor: {processor_name}")
                continue
                
            logger.info(f"Running {processor_name} processor")
            processor_class = self.processor_map[processor_name]
            processor = processor_class(config.get(processor_name, {}))
            
            # Process each document and collect chunks
            for document in documents:
                chunks = processor.process(document)
                all_chunks.extend(chunks)
        
        logger.info(f"Generated {len(all_chunks)} total chunks from {len(processor_names)} processors")
        return all_chunks
    
    def get_available_processors(self) -> List[str]:
        """Get list of available processor names."""
        return list(self.processor_map.keys())