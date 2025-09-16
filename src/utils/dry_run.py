import os
import asyncio
from functools import wraps
from typing import Any, Callable, List
import random
from copy import deepcopy

from models import DocumentChunk

_DRY_RUN_MODE = False

def set_dry_run_mode(enabled: bool = True):
    """Enable or disable dry run mode globally."""
    global _DRY_RUN_MODE
    _DRY_RUN_MODE = enabled

def is_dry_run_mode() -> bool:
    """Check if dry run mode is enabled."""
    return _DRY_RUN_MODE or os.getenv("DRY_RUN", "false").lower() == "true"


def dry_response(mock_value: Any = None, *, mock_factory: Callable = None):
    """
    Decorator that returns a mock response when dry run mode is enabled.
    
    Args:
        mock_value: Static value to return in dry run mode (string, list, dict, etc.)
        mock_factory: Function that generates mock data dynamically (takes original args)
    
    Usage:
        @dry_response("mock string")
        @dry_response(["mock", "list"])
        @dry_response({"mock": "dict"})
        @dry_response(mock_factory=lambda chunks: generate_mock_chunks(chunks))
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if is_dry_run_mode():
                if mock_factory is not None:
                    return mock_factory(*args, **kwargs)
                elif mock_value is not None:
                    return deepcopy(mock_value)
                else:
                    # Default mock based on function name or return type hints
                    return await _generate_default_mock(func, *args, **kwargs)
            return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if is_dry_run_mode():
                if mock_factory is not None:
                    return mock_factory(*args, **kwargs)
                elif mock_value is not None:
                    return deepcopy(mock_value)
                else:
                    return _generate_default_sync_mock(func, *args, **kwargs)
            return func(*args, **kwargs)
        
        # Return appropriate wrapper based on whether function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def _generate_mock_embedding(dimensions: int = 384) -> List[float]:
    """Generate a random embedding vector with specified dimensions."""
    random.seed(42)  # Deterministic for testing
    return [random.uniform(-1.0, 1.0) for _ in range(dimensions)]


def _generate_mock_chunks(chunks: List[DocumentChunk], dimensions: int = 384) -> List[DocumentChunk]:
    """Add mock embeddings to existing DocumentChunks."""
    # Modify original chunks in place to add embeddings
    for chunk in chunks:
        if not chunk.embedding:  # Only add embedding if it doesn't exist
            mock_embedding = _generate_mock_embedding(dimensions)
            chunk.embedding = mock_embedding

    return chunks



async def _generate_default_mock(func: Callable, *args, **kwargs) -> Any:
    """Generate default mock response based on function context."""
    # Check if this looks like an embedding function
    if hasattr(args[0], '__class__') and 'embedding' in args[0].__class__.__name__.lower():
        # Assume first argument after self is chunks
        if len(args) > 1 and isinstance(args[1], list):
            chunks = args[1]
            if chunks and isinstance(chunks[0], DocumentChunk):
                return _generate_mock_chunks(chunks)
    
    return f"DRY_RUN_MOCK_RESPONSE_FOR_{func.__name__}"


def _generate_default_sync_mock(func: Callable, *args, **kwargs) -> Any:
    """Generate default sync mock response."""
    return f"DRY_RUN_MOCK_RESPONSE_FOR_{func.__name__}"


# Convenience factory functions for common mock types
def mock_embedding_chunks(dimensions: int = 384):
    """Factory function to create mock embedding chunks with custom dimensions."""
    def factory(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        return _generate_mock_chunks(chunks, dimensions)
    return factory


def mock_string(value: str):
    """Factory function to create string mock."""
    def factory(*args, **kwargs) -> str:
        return value
    return factory


def mock_list(items: List[Any]):
    """Factory function to create list mock."""
    def factory(*args, **kwargs) -> List[Any]:
        return deepcopy(items)
    return factory