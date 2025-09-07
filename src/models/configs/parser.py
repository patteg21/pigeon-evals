from typing import Optional, List, Literal
from pydantic import BaseModel, Field


class StepConfig(BaseModel):
    """Configuration for a single step in a process chain."""
    strategy: Literal["character", "word", "sentence", "paragraph", "regex", "separator"] = Field("character", description="Splitting strategy")
    chunk_size: Optional[int] = Field(None, description="Maximum chunk size (None for no limit)")
    chunk_overlap: int = Field(0, description="Overlap between chunks")
    separator: Optional[str] = Field("\n\n", description="Separator for splitting")
    regex_pattern: Optional[str] = Field(None, description="Regex pattern for regex splitting")
    keep_separator: bool = Field(False, description="Whether to keep separators in chunks")
    ignore_case: bool = Field(False, description="Case-insensitive matching for regex")
    keep_empty: bool = Field(False, description="Keep empty chunks after splitting")
    trim_whitespace: bool = Field(True, description="Trim whitespace from chunks")


class ProcessConfig(BaseModel):
    """Configuration for a process containing multiple chained steps."""
    name: str = Field(..., description="Name for this process")
    steps: List[StepConfig] = Field(..., description="List of steps to apply in sequence")


class SplitStageConfig(BaseModel):
    """Configuration for a single text splitting stage (legacy)."""
    name: Optional[str] = Field(None, description="Optional name for this stage")
    strategy: Literal["character", "word", "sentence", "paragraph", "regex", "separator"] = Field("character", description="Splitting strategy")
    chunk_size: Optional[int] = Field(None, description="Maximum chunk size (None for no limit)")
    chunk_overlap: int = Field(256, description="Overlap between chunks")
    separator: Optional[str] = Field("\n\n", description="Separator for splitting")
    regex_pattern: Optional[str] = Field(None, description="Regex pattern for regex splitting")
    keep_separator: bool = Field(False, description="Whether to keep separators in chunks")


class MultiStageParserConfig(BaseModel):
    """Configuration for multi-stage text parsing."""
    stages: List[SplitStageConfig] = Field(..., description="List of parsing stages to apply in order")


class RegexParserConfig(BaseModel):
    """Legacy regex parser configuration."""
    patterns: List[str] = Field(..., description="List of regex patterns")


class ParserConfig(BaseModel):
    """Main parser configuration that supports multiple parsing types."""
    type: Literal["regex", "multistage", "simple"] = Field("simple", description="Type of parser")
    
    # For regex type (backward compatibility)
    patterns: Optional[List[str]] = Field(None, description="Regex patterns for regex type")
    
    # For multistage type with new process-based structure
    processes: Optional[List[ProcessConfig]] = Field(None, description="Independent processes with chained steps")
    
    # For multistage type (legacy)
    stages: Optional[List[SplitStageConfig]] = Field(None, description="Multi-stage parsing configuration (legacy)")
    
    # For simple type (single stage)
    strategy: Optional[Literal["character", "word", "sentence", "paragraph", "regex", "separator"]] = Field("character", description="Splitting strategy")
    chunk_size: Optional[int] = Field(None, description="Maximum chunk size")
    chunk_overlap: int = Field(256, description="Overlap between chunks")
    separator: Optional[str] = Field("\n\n", description="Separator for splitting")
    regex_pattern: Optional[str] = Field(None, description="Regex pattern for regex splitting")
    keep_separator: bool = Field(False, description="Whether to keep separators in chunks")


class RegexParser(BaseModel):
    """Legacy regex parser model."""
    pass