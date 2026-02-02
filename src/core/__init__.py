"""Core business logic for RAG application"""

from .document_processor import (
    save_uploaded_file,
    load_pdf_documents,
    chunk_documents,
    format_docs,
    format_docs_with_metadata,
    get_unique_sources
)

from .vector_store import (
    ChromaVectorStore,
    InMemoryVectorStoreWrapper
)

from .rag_chain import (
    create_rag_chain,
    generate_enhanced_answer,
    format_sources_for_display
)

__all__ = [
    'save_uploaded_file',
    'load_pdf_documents',
    'chunk_documents',
    'format_docs',
    'format_docs_with_metadata',
    'get_unique_sources',
    'ChromaVectorStore',
    'InMemoryVectorStoreWrapper',
    'create_rag_chain',
    'generate_enhanced_answer',
    'format_sources_for_display'
]
