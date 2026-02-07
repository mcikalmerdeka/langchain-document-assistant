"""
Document processing utilities for loading, chunking, and formatting documents
"""
import os
from datetime import datetime
from typing import List, Dict, Any
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config.settings import PDF_STORAGE_PATH, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from config.logging import get_logger

logger = get_logger("document_processor")


def save_uploaded_file(uploaded_file) -> str:
    """
    Save the uploaded PDF file to storage
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        str: Path to the saved file
    """
    file_path = PDF_STORAGE_PATH + uploaded_file.name
    logger.info(f"Saving uploaded file: {uploaded_file.name} to {file_path}")
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    logger.info(f"Successfully saved file: {file_path}")
    return file_path


def extract_pdf_metadata(file_path: str) -> Dict[str, Any]:
    """
    Extract PDF file metadata
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Dictionary containing file metadata
    """
    try:
        file_stat = os.stat(file_path)
        return {
            "source": file_path,
            "filename": os.path.basename(file_path),
            "file_size_bytes": file_stat.st_size,
            "uploaded_at": datetime.now().isoformat(),
            "file_type": "PDF"
        }
    except Exception:
        return {
            "source": file_path,
            "filename": os.path.basename(file_path),
            "file_type": "PDF"
        }


def load_pdf_documents(file_path: str):
    """
    Load PDF documents from the uploaded file with rich metadata
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        List of loaded documents with metadata
        
    Raises:
        ValueError: If no content could be loaded from the PDF
    """
    logger.info(f"Loading PDF document from: {file_path}")
    document_loader = PyMuPDFLoader(file_path)
    docs = document_loader.load()
    
    if not docs:
        logger.error(f"Failed to load any content from PDF: {file_path}")
        raise ValueError(f"Failed to load any content from PDF: {file_path}")
    
    logger.info(f"Loaded {len(docs)} pages from PDF: {file_path}")
    
    # Extract file-level metadata
    file_metadata = extract_pdf_metadata(file_path)
    logger.debug(f"Extracted PDF metadata: {file_metadata}")
    
    # Enrich each document with metadata
    for doc in docs:
        # PyMuPDF already provides page metadata, enrich it
        doc.metadata.update(file_metadata)
        doc.metadata["page_number"] = doc.metadata.get("page", 0) + 1  # 1-based page numbers
        
    logger.info(f"Successfully enriched {len(docs)} documents with metadata")
    return docs


def chunk_documents(raw_documents, chunk_size: int = DEFAULT_CHUNK_SIZE, 
                    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
    """
    Chunk the documents into smaller parts while preserving metadata
    
    Args:
        raw_documents: List of documents to chunk
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of document chunks with preserved metadata
        
    Raises:
        ValueError: If no documents to chunk or chunking produces no results
    """
    logger.info(f"Chunking {len(raw_documents)} documents with size={chunk_size}, overlap={chunk_overlap}")
    
    if not raw_documents:
        logger.error("No documents to chunk. The document may be empty.")
        raise ValueError("No documents to chunk. The document may be empty.")
    
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", ", ", " ", ""]  # Better semantic splitting
    )
    chunks = text_processor.split_documents(raw_documents)
    
    if not chunks:
        logger.error("Chunking produced no results. The document may contain no text.")
        raise ValueError("Chunking produced no results. The document may contain no text.")
    
    logger.info(f"Created {len(chunks)} chunks from {len(raw_documents)} documents")
    
    # Add chunk index to metadata for tracking
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["total_chunks"] = len(chunks)
    
    logger.debug(f"Successfully enriched all chunks with metadata")
    return chunks


def format_docs(docs) -> str:
    """
    Format retrieved documents into a single context string
    
    Args:
        docs: List of documents to format
        
    Returns:
        str: Formatted context string
    """
    return "\n\n".join([doc.page_content for doc in docs])


def format_docs_with_metadata(docs: List[Document]) -> tuple[str, List[Dict[str, Any]]]:
    """
    Format retrieved documents into a context string with metadata tracking
    
    Args:
        docs: List of documents to format
        
    Returns:
        Tuple of (formatted context string, list of source metadata)
    """
    formatted_chunks = []
    sources = []
    
    for i, doc in enumerate(docs, 1):
        # Build source metadata
        source_info = {
            "chunk_number": i,
            "content": doc.page_content,
            "metadata": doc.metadata.copy()
        }
        sources.append(source_info)
        
        # Format with metadata header
        chunk_header = f"--- Source {i} ---"
        
        # Add metadata info if available
        meta_parts = []
        if doc.metadata.get("filename"):
            meta_parts.append(f"File: {doc.metadata['filename']}")
        if doc.metadata.get("page_number"):
            meta_parts.append(f"Page: {doc.metadata['page_number']}")
        if doc.metadata.get("chunk_index") is not None:
            meta_parts.append(f"Chunk: {doc.metadata['chunk_index'] + 1}/{doc.metadata.get('total_chunks', '?')}")
        
        if meta_parts:
            chunk_header += f"\n[{' | '.join(meta_parts)}]"
        
        formatted_chunks.append(f"{chunk_header}\n\n{doc.page_content}")
    
    return "\n\n".join(formatted_chunks), sources


def get_unique_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract unique source documents from retrieved chunks
    
    Args:
        sources: List of source metadata from format_docs_with_metadata
        
    Returns:
        List of unique source files with their page ranges
    """
    source_files = {}
    
    for source in sources:
        filename = source["metadata"].get("filename", "Unknown")
        page = source["metadata"].get("page_number")
        
        if filename not in source_files:
            source_files[filename] = {
                "filename": filename,
                "pages": set(),
                "chunks": 0
            }
        
        if page:
            source_files[filename]["pages"].add(page)
        source_files[filename]["chunks"] += 1
    
    # Convert to list and format pages
    unique_sources = []
    for filename, data in source_files.items():
        pages = sorted(data["pages"]) if data["pages"] else []
        unique_sources.append({
            "filename": filename,
            "pages": pages,
            "chunk_count": data["chunks"],
            "page_range": f"Pages {min(pages)}-{max(pages)}" if pages else "N/A"
        })
    
    return unique_sources
