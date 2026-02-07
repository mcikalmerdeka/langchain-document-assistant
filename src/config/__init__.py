"""Configuration package for the RAG application"""

from .settings import (
    PDF_STORAGE_PATH,
    EMBEDDING_MODEL_NAME,
    MODEL_OPTIONS,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_RETRIEVAL_K,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY
)

from .models import initialize_language_model, APIKeyError
from .logging import setup_logger, get_logger

__all__ = [
    'PDF_STORAGE_PATH',
    'EMBEDDING_MODEL_NAME',
    'MODEL_OPTIONS',
    'DEFAULT_CHUNK_SIZE',
    'DEFAULT_CHUNK_OVERLAP',
    'DEFAULT_RETRIEVAL_K',
    'CHROMA_PERSIST_DIR',
    'CHROMA_COLLECTION_NAME',
    'OPENAI_API_KEY',
    'ANTHROPIC_API_KEY',
    'initialize_language_model',
    'APIKeyError',
    'setup_logger',
    'get_logger'
]
