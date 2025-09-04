from .base import BaseEmbedder
from .openai_embedder import  OpenAIEmbedder
from .huggingface_embedder import  HuggingFaceEmbedder


class Embedder(BaseEmbedder | OpenAIEmbedder | HuggingFaceEmbedder):
    pass



__all__ = ["BaseEmbedder", "OpenAIEmbedder", "HuggingFaceEmbedder"]