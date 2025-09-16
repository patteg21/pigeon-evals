from typing import List, Dict, Optional
import re
from uuid import uuid4
from tqdm import tqdm

from models import DocumentChunk, Document
from models.configs.parser import ParserConfig, ProcessConfig, StepConfig
from utils import logger

class TextSplitterBuilder:

    def __init__(self, config: ParserConfig):

        self.config = config
        pass

    
    def process(self, document: Document) -> List[DocumentChunk]:
        document_chunks: List[DocumentChunk] = []
        logger.info(f"Processing: {document.name}")
        for process in self.config.processes:
            chunks = self._process(process, document)
            document_chunks.extend(chunks)

        return document_chunks

    def _process(
            self,
            process: ProcessConfig,
            document: Document
        ) -> List[DocumentChunk]:
        # start with a single chunk which is just the document
        original_chunk = DocumentChunk(
            id=uuid4().hex,
            text=document.text,
            document=document
        )

        chunks: List[DocumentChunk] = [original_chunk]
        for step in tqdm(process.steps, desc=f"Processing {process.name}", unit="step", leave=False):
            chunks = self._process_step(step, chunks, process)

        return chunks


    def _process_step(
            self,
            step: StepConfig,
            chunks: List[DocumentChunk],
            process: ProcessConfig
        ) -> List[DocumentChunk]:

        result_chunks: List[DocumentChunk] = []

        for chunk in tqdm(chunks, desc=f"Step: {step.strategy}", unit="chunk", leave=False):
            split_texts = self._split_text(chunk.text, step)

            for i, split_text in enumerate(split_texts):
                # Create chunk based on step configuration (empty handling is done in _split_text)
                new_chunk = DocumentChunk(
                    text=split_text,
                    document=chunk.document
                )
                result_chunks.append(new_chunk)

        return result_chunks

    def _split_text(self, text: str, step: StepConfig) -> List[str]:
        """Split text based on the step configuration."""
        if step.strategy == "regex":
            splits = self._split_by_regex(text, step.regex_pattern, step.ignore_case)
        elif step.strategy == "character":
            splits = self._split_by_character(text, step.chunk_size, step.chunk_overlap)
        elif step.strategy == "word":
            splits = self._split_by_word(text, step.chunk_size, step.chunk_overlap)
        elif step.strategy == "sentence":
            splits = self._split_by_sentence(text, step.chunk_size, step.chunk_overlap)
        elif step.strategy == "paragraph":
            splits = self._split_by_paragraph(text)
        elif step.strategy == "separator":
            splits = self._split_by_separator(text, step.separator or "\n\n")
        else:
            splits = [text]
        
        # Apply post-processing based on step configuration
        processed_splits = []
        for split in splits:
            if step.trim_whitespace:
                split = split.strip()
            
            # Keep or discard empty chunks
            if split or step.keep_empty:
                processed_splits.append(split)
        
        return processed_splits
    
    def _split_by_regex(self, text: str, pattern: str, ignore_case: bool = False) -> List[str]:
        """Split text using regex pattern."""
        if not pattern:
            return [text]
        
        flags = re.IGNORECASE if ignore_case else 0
        splits = re.split(pattern, text, flags=flags)
        return splits
    
    def _split_by_character(self, text: str, chunk_size: Optional[int], chunk_overlap: int = 0) -> List[str]:
        """Split text by character count with optional overlap."""
        if chunk_size is None:
            return [text]
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            
            if end >= len(text):
                break
                
            start = max(start + chunk_size - chunk_overlap, start + 1)
            
        return chunks
    
    def _split_by_word(self, text: str, word_count: Optional[int], chunk_overlap: int = 0) -> List[str]:
        """Split text by word count with optional overlap."""
        if word_count is None:
            return [text]
            
        words = text.split()
        if len(words) <= word_count:
            return [text]
        
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + word_count, len(words))
            chunk_words = words[start:end]
            chunk = " ".join(chunk_words)
            chunks.append(chunk)
            
            if end >= len(words):
                break
                
            start = max(start + word_count - chunk_overlap, start + 1)
            
        return chunks
    
    def _split_by_sentence(self, text: str, sentence_count: Optional[int], chunk_overlap: int = 0) -> List[str]:
        """Split text by sentence count with optional overlap."""
        if sentence_count is None:
            return [text]
            
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= sentence_count:
            return [text]
        
        chunks = []
        start = 0
        while start < len(sentences):
            end = min(start + sentence_count, len(sentences))
            chunk_sentences = sentences[start:end]
            chunk = ". ".join(chunk_sentences) + "."
            chunks.append(chunk)
            
            if end >= len(sentences):
                break
                
            start = max(start + sentence_count - chunk_overlap, start + 1)
            
        return chunks
    
    def _split_by_paragraph(self, text: str) -> List[str]:
        """Split text by paragraphs.
        
        Paragraph Detection Logic:
        - A paragraph is defined as text separated by double newlines ('\n\n')
        - This is the standard markdown/plain text convention
        - Single newlines (\n) are treated as line breaks within a paragraph
        - Double newlines (\n\n) create paragraph boundaries
        
        Examples:
        "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        Results in: ["First paragraph.", "Second paragraph.", "Third paragraph."]
        
        "Line 1\nLine 2\n\nNew paragraph\nWith multiple lines"
        Results in: ["Line 1\nLine 2", "New paragraph\nWith multiple lines"]
        """
        # Split on double newlines to separate paragraphs
        paragraphs = text.split('\n\n')
        
        # Strip whitespace and filter out empty paragraphs
        # This removes any paragraphs that are only whitespace
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _split_by_separator(self, text: str, separator: str) -> List[str]:
        """Split text by separator."""
        splits = text.split(separator)
        return [split.strip() for split in splits if split.strip()]