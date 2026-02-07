"""
Streamlit application using DeepSeek R1 local LLM with Ollama
Uses InMemoryVectorStore and PDFPlumberLoader for better local processing
"""
import streamlit as st
import sys
import os
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_openai import OpenAIEmbeddings

# Import configuration
from config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, PDF_STORAGE_PATH, setup_logger

# Setup application logger
logger = setup_logger("docuschat_deepseek")

# Import core functionality
from core import (
    chunk_documents,
    InMemoryVectorStoreWrapper,
    create_rag_chain,
    generate_enhanced_answer,
    format_sources_for_display
)

# Import UI components
from styles import apply_custom_theme
from components import (
    render_app_header,
    render_app_info_expander,
    render_deepseek_flow_expander,
    render_external_search_toggle,
    render_clear_chat_button,
    render_file_uploader,
    display_chat_history,
    render_status_message
)

# Check external search availability
try:
    from agents.external_sources_lookup_agent import lookup
    EXTERNAL_SEARCH_AVAILABLE = True
    logger.info("External search agent loaded successfully (DeepSeek)")
except ImportError as e:
    logger.warning(f"External search not available (DeepSeek): {e}")
    st.warning(f"External search not available: {e}")
    EXTERNAL_SEARCH_AVAILABLE = False

# Load environment variables
load_dotenv()

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Apply custom theme
apply_custom_theme()

# DeepSeek-specific configurations
DEEPSEEK_MODEL = "deepseek-r1:1.5b"
logger.info(f"DeepSeek model configured: {DEEPSEEK_MODEL}")

# Choose embedding model (OpenAI recommended for better quality)
USE_OPENAI_EMBEDDINGS = True
logger.info(f"Using OpenAI embeddings: {USE_OPENAI_EMBEDDINGS}")

if USE_OPENAI_EMBEDDINGS:
    EMBEDDING_MODEL = OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="text-embedding-3-large"
    )
    logger.info("OpenAI embeddings initialized")
else:
    EMBEDDING_MODEL = OllamaEmbeddings(model=DEEPSEEK_MODEL)
    logger.info("Ollama embeddings initialized")

# Initialize vector store with chosen embeddings
if 'vector_store' not in st.session_state:
    from langchain_core.vectorstores import InMemoryVectorStore
    st.session_state.vector_store = InMemoryVectorStore(EMBEDDING_MODEL)
    logger.info("InMemoryVectorStore initialized for DeepSeek")

# Initialize DeepSeek language model
LANGUAGE_MODEL = OllamaLLM(model=DEEPSEEK_MODEL)
logger.info(f"DeepSeek language model initialized: {DEEPSEEK_MODEL}")


def save_uploaded_file(uploaded_file):
    """Save uploaded PDF file"""
    file_path = PDF_STORAGE_PATH + uploaded_file.name
    logger.info(f"Saving uploaded file (DeepSeek): {uploaded_file.name}")
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    logger.info(f"Successfully saved file (DeepSeek): {file_path}")
    return file_path


def extract_pdf_metadata(file_path: str) -> dict:
    """Extract PDF file metadata"""
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


def load_pdf_documents(file_path):
    """Load PDF documents using PDFPlumberLoader with metadata enrichment"""
    logger.info(f"Loading PDF document (DeepSeek): {file_path}")
    document_loader = PDFPlumberLoader(file_path)
    docs = document_loader.load()
    
    if not docs:
        logger.error(f"Failed to load any content from PDF (DeepSeek): {file_path}")
        raise ValueError(f"Failed to load any content from PDF: {file_path}")
    
    logger.info(f"Loaded {len(docs)} pages from PDF (DeepSeek): {file_path}")
    
    # Extract file-level metadata
    file_metadata = extract_pdf_metadata(file_path)
    
    # Enrich each document with metadata
    for doc in docs:
        doc.metadata.update(file_metadata)
        doc.metadata["page_number"] = doc.metadata.get("page", 0) + 1  # 1-based page numbers
    
    logger.info(f"Successfully enriched {len(docs)} documents with metadata (DeepSeek)")
    return docs


# Streamlit UI Configuration
render_app_header("📘 DocuChat AI - DeepSeek R1", "Local LLM with Ollama Integration")

# Render information expanders
render_app_info_expander()
render_deepseek_flow_expander()

# Sidebar components
render_clear_chat_button()
external_search_enabled = render_external_search_toggle(EXTERNAL_SEARCH_AVAILABLE)

# File Upload Section
uploaded_pdf = render_file_uploader()

# Main App Logic
if uploaded_pdf:
    logger.info(f"Processing uploaded PDF (DeepSeek): {uploaded_pdf.name}")
    saved_path = save_uploaded_file(uploaded_pdf)
    vector_store = st.session_state.vector_store
    
    # Always process (no duplicate check for InMemory store in DeepSeek version)
    raw_docs = load_pdf_documents(saved_path)
    processed_chunks = chunk_documents(raw_docs)
    vector_store.add_documents(processed_chunks)
    
    logger.info(f"Document processed successfully (DeepSeek) with {len(processed_chunks)} chunks")
    
    # Create the RAG chain
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    rag_chain = create_rag_chain(LANGUAGE_MODEL, retriever, external_search_enabled)
    
    # Store in session state
    st.session_state.retriever = retriever
    st.session_state.rag_chain = rag_chain
    
    # Display success message
    mode_info = "with External Search" if external_search_enabled else "Document Only Mode"
    render_status_message("success", "Document processed successfully! Ask your questions below", 
                        model_name="DeepSeek R1", mode_info=mode_info)

    # Display existing chat history
    display_chat_history()
    
    # Handle new user input
    user_input = st.chat_input("Enter your question about the document...")
    
    if user_input:
        logger.info(f"User query received (DeepSeek): {user_input[:50]}...")
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Dynamic spinner message based on mode
        spinner_message = "Analyzing document with DeepSeek R1 (external search enabled)..." if external_search_enabled else "Analyzing document with DeepSeek R1 (document only mode)..."
        
        with st.spinner(spinner_message):
            logger.debug("Generating AI response (DeepSeek)...")
            # Use the enhanced answer generation (now returns tuple of answer and sources)
            ai_response, sources = generate_enhanced_answer(
                user_input, 
                st.session_state.rag_chain, 
                LANGUAGE_MODEL,
                st.session_state.retriever,
                external_search_enabled,
                EXTERNAL_SEARCH_AVAILABLE
            )
            
            # Format sources for display
            formatted_sources = format_sources_for_display(sources)
            
            logger.info(f"AI response generated successfully (DeepSeek), length: {len(ai_response)} characters")
            
            # Add assistant response to chat history with sources
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": ai_response,
                "sources": formatted_sources
            })
        
        # Rerun to display the updated chat history
        st.rerun()
