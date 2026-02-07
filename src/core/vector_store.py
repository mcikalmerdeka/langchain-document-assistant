"""
Vector store management for document retrieval
"""
import os
from langchain_chroma import Chroma
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from config.settings import (
    EMBEDDING_MODEL_NAME,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    DEFAULT_RETRIEVAL_K
)
from config.logging_config import get_logger

logger = get_logger("vector_store")


def get_embedding_model():
    """Get the configured embedding model"""
    return OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=EMBEDDING_MODEL_NAME
    )


class ChromaVectorStore:
    """Wrapper for ChromaDB vector store operations"""
    
    def __init__(self):
        logger.info(f"Initializing ChromaVectorStore with collection: {CHROMA_COLLECTION_NAME}")
        self.embedding_model = get_embedding_model()
        self.vector_db = Chroma(
            embedding_function=self.embedding_model,
            collection_name=CHROMA_COLLECTION_NAME,
            persist_directory=CHROMA_PERSIST_DIR
        )
        logger.info(f"ChromaVectorStore initialized successfully at: {CHROMA_PERSIST_DIR}")
    
    def add_documents(self, document_chunks):
        """Add document chunks to the vector store"""
        logger.info(f"Adding {len(document_chunks)} document chunks to vector store")
        if not document_chunks:
            logger.error("No document chunks to index. The document may be empty or failed to load.")
            raise ValueError("No document chunks to index. The document may be empty or failed to load.")
        self.vector_db.add_documents(document_chunks)
        logger.info(f"Successfully added {len(document_chunks)} document chunks to vector store")
    
    def document_exists(self, file_name: str) -> bool:
        """Check if document already exists in vector store"""
        logger.debug(f"Checking if document exists: {file_name}")
        try:
            existing_docs = self.vector_db.get()
            if existing_docs and 'metadatas' in existing_docs:
                existing_files = [doc.get('source', '') for doc in existing_docs['metadatas'] if doc]
                exists = any(file_name in file_path for file_path in existing_files)
                if exists:
                    logger.info(f"Document already exists in vector store: {file_name}")
                else:
                    logger.debug(f"Document not found in vector store: {file_name}")
                return exists
        except Exception as e:
            logger.warning(f"Error checking document existence for {file_name}: {e}")
            pass
        return False
    
    def create_retriever(self, k: int = DEFAULT_RETRIEVAL_K):
        """Create a retriever from the vector store"""
        logger.info(f"Creating retriever with k={k}")
        return self.vector_db.as_retriever(search_kwargs={"k": k})
    
    def reset(self):
        """Reset or initialize vector store"""
        logger.warning("Resetting vector store - deleting all collections")
        try:
            self.vector_db.delete_collection()
            logger.info("Deleted existing collection")
        except Exception as e:
            logger.warning(f"Error deleting collection: {e}")
            pass
        self.vector_db = Chroma(
            embedding_function=self.embedding_model,
            collection_name=CHROMA_COLLECTION_NAME,
            persist_directory=CHROMA_PERSIST_DIR
        )
        logger.info("Vector store reset complete")
        return self.vector_db


class InMemoryVectorStoreWrapper:
    """Wrapper for InMemoryVectorStore operations"""
    
    def __init__(self):
        logger.info("Initializing InMemoryVectorStoreWrapper")
        self.embedding_model = get_embedding_model()
        self.vector_db = InMemoryVectorStore(self.embedding_model)
        logger.info("InMemoryVectorStoreWrapper initialized successfully")
    
    def add_documents(self, document_chunks):
        """Add document chunks to the vector store"""
        logger.info(f"Adding {len(document_chunks)} document chunks to InMemoryVectorStore")
        self.vector_db.add_documents(document_chunks)
        logger.info(f"Successfully added {len(document_chunks)} document chunks to InMemoryVectorStore")
    
    def document_exists(self, file_name: str) -> bool:
        """Check if document already exists in vector store"""
        logger.debug(f"Checking if document exists in InMemoryVectorStore: {file_name}")
        try:
            stored_docs = self.vector_db.similarity_search("", k=1000)
            if stored_docs:
                existing_files = [doc.metadata.get('source', '') for doc in stored_docs if hasattr(doc, 'metadata')]
                exists = any(file_name in file_path for file_path in existing_files)
                if exists:
                    logger.info(f"Document already exists in InMemoryVectorStore: {file_name}")
                return exists
        except Exception as e:
            logger.warning(f"Error checking document existence in InMemoryVectorStore for {file_name}: {e}")
            pass
        return False
    
    def create_retriever(self, k: int = DEFAULT_RETRIEVAL_K):
        """Create a retriever from the vector store"""
        logger.info(f"Creating InMemoryVectorStore retriever with k={k}")
        return self.vector_db.as_retriever(search_kwargs={"k": k})
