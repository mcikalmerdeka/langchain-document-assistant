"""
RAG chain creation and answer generation
"""
import streamlit as st
from typing import Dict, List, Any, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from config.prompts import (
    PROMPT_TEMPLATE,
    DOCUMENT_ONLY_PROMPT_TEMPLATE,
    ENHANCED_PROMPT_TEMPLATE
)
from .document_processor import format_docs_with_metadata, get_unique_sources
from config.logging_config import get_logger

logger = get_logger("rag_chain")


def create_rag_chain(language_model, retriever, external_search_enabled: bool = True):
    """
    Create a comprehensive RAG chain using LangChain Expression Language
    
    Args:
        language_model: The LLM to use for generation
        retriever: Document retriever
        external_search_enabled: Whether external search is enabled
        
    Returns:
        RAG chain
    """
    logger.info(f"Creating RAG chain with external_search_enabled={external_search_enabled}")
    
    # Choose prompt template based on external search setting
    if external_search_enabled:
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        logger.debug("Using PROMPT_TEMPLATE with external search capability")
    else:
        prompt = ChatPromptTemplate.from_template(DOCUMENT_ONLY_PROMPT_TEMPLATE)
        logger.debug("Using DOCUMENT_ONLY_PROMPT_TEMPLATE")
    
    # Create the RAG chain using LCEL syntax with metadata-enhanced formatting
    def format_with_metadata(docs):
        """Helper to format docs and keep sources for later use"""
        formatted_context, sources = format_docs_with_metadata(docs)
        # Store sources in a way they can be retrieved later
        format_with_metadata._last_sources = sources
        logger.debug(f"Formatted {len(sources)} documents with metadata")
        return formatted_context
    
    rag_chain = (
        {
            "document_context": retriever | RunnableLambda(format_with_metadata),
            "user_query": RunnablePassthrough()
        }
        | prompt
        | language_model
        | StrOutputParser()
    )
    
    # Attach sources retrieval method to chain
    rag_chain._format_with_metadata = format_with_metadata
    
    logger.info("RAG chain created successfully")
    return rag_chain


def generate_enhanced_answer(user_query: str, rag_chain, language_model, 
                            retriever, external_search_enabled: bool = True,
                            external_search_available: bool = True) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Generate answer using RAG chain with optional external search fallback
    
    Args:
        user_query: User's question
        rag_chain: The RAG chain to use
        language_model: The LLM to use
        retriever: Document retriever
        external_search_enabled: Whether external search is enabled
        external_search_available: Whether external search is available
        
    Returns:
        Tuple of (generated answer, list of source metadata)
    """
    logger.info(f"Generating enhanced answer for query: '{user_query[:50]}...' "
                f"external_search_enabled={external_search_enabled}, "
                f"external_search_available={external_search_available}")
    
    sources = []
    
    try:
        # Step 1: Get document context with metadata
        logger.debug("Step 1: Retrieving documents from vector store")
        docs = retriever.invoke(user_query)
        document_context, sources = format_docs_with_metadata(docs)
        logger.info(f"Retrieved {len(docs)} documents, created {len(sources)} source entries")
        
        # Step 2: Create prompt based on external search setting
        if external_search_enabled:
            prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        else:
            prompt = ChatPromptTemplate.from_template(DOCUMENT_ONLY_PROMPT_TEMPLATE)
        
        # Step 3: Create and invoke the chain
        logger.debug("Step 3: Creating and invoking initial chain")
        initial_chain = (
            {
                "document_context": RunnableLambda(lambda _: document_context),
                "user_query": RunnablePassthrough()
            }
            | prompt
            | language_model
            | StrOutputParser()
        )
        
        initial_response = initial_chain.invoke(user_query)
        logger.debug(f"Initial response length: {len(initial_response)} characters")
        
        # Step 4: Check if external search is needed and enabled
        if (external_search_enabled and 
            "[EXTERNAL_SEARCH_NEEDED]" in initial_response and 
            external_search_available):
            
            logger.info("External search needed and available - initiating external lookup")
            st.info("🔍 Document context insufficient. Searching external sources...")
            
            try:
                # Import here to avoid circular dependency
                from agents.external_sources_lookup_agent import lookup
                
                # Step 5: Perform external search
                logger.info("Step 5: Performing external search")
                external_results = lookup(user_query)
                logger.info(f"External search completed, results length: {len(external_results)} characters")
                
                # Step 6: Create enhanced prompt with both contexts
                enhanced_prompt = ChatPromptTemplate.from_template(ENHANCED_PROMPT_TEMPLATE)
                
                # Step 7: Generate enhanced response
                logger.info("Step 7: Generating enhanced response with external context")
                enhanced_chain = enhanced_prompt | language_model | StrOutputParser()
                
                final_response = enhanced_chain.invoke({
                    "user_query": user_query,
                    "document_context": document_context,
                    "external_context": external_results
                })
                
                # Clean up and return enhanced response
                cleaned_response = final_response.strip()
                cleaned_response = '\n'.join(line.strip() for line in cleaned_response.split('\n') if line.strip())
                
                logger.info("Enhanced answer generated successfully with external search")
                return cleaned_response, sources
                
            except Exception as e:
                logger.error(f"External search failed: {e}", exc_info=True)
                st.error(f"External search failed: {str(e)}")
                # Fall back to original response without the search indicator
                fallback_response = initial_response.replace("[EXTERNAL_SEARCH_NEEDED]", "").strip()
                cleaned_response = fallback_response if fallback_response else "I don't have sufficient information to answer this query."
                logger.info("Returning fallback response without external search")
                return cleaned_response, sources
        
        else:
            # Step 8: Return normal response (either external search disabled or not needed)
            logger.info("No external search needed or disabled - returning document-based response")
            if not external_search_enabled:
                # Clean response for document-only mode
                cleaned_response = initial_response.strip()
            else:
                # Clean response and remove external search indicator if present
                cleaned_response = initial_response.replace("[EXTERNAL_SEARCH_NEEDED]", "").strip()
            
            cleaned_response = '\n'.join(line.strip() for line in cleaned_response.split('\n') if line.strip())
            cleaned_response = cleaned_response if cleaned_response else "I don't have sufficient information to answer this query."
            
            return cleaned_response, sources
            
    except Exception as e:
        logger.error(f"Error generating response: {e}", exc_info=True)
        return f"Error generating response: {str(e)}", sources


def format_sources_for_display(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format sources for display in the UI
    
    Args:
        sources: Raw sources from document retrieval
        
    Returns:
        List of formatted source information
    """
    if not sources:
        return []
    
    # Get unique source files with their information
    unique_sources = get_unique_sources(sources)
    
    # Format each source for display
    formatted = []
    for source_info in unique_sources:
        formatted.append({
            "filename": source_info["filename"],
            "pages": source_info["pages"],
            "page_range": source_info["page_range"],
            "chunks_used": source_info["chunk_count"]
        })
    
    return formatted
