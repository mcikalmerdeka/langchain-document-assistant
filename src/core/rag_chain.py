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
    # Choose prompt template based on external search setting
    if external_search_enabled:
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    else:
        prompt = ChatPromptTemplate.from_template(DOCUMENT_ONLY_PROMPT_TEMPLATE)
    
    # Create the RAG chain using LCEL syntax with metadata-enhanced formatting
    def format_with_metadata(docs):
        """Helper to format docs and keep sources for later use"""
        formatted_context, sources = format_docs_with_metadata(docs)
        # Store sources in a way they can be retrieved later
        format_with_metadata._last_sources = sources
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
    sources = []
    
    try:
        # Step 1: Get document context with metadata
        docs = retriever.invoke(user_query)
        document_context, sources = format_docs_with_metadata(docs)
        
        # Step 2: Create prompt based on external search setting
        if external_search_enabled:
            prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        else:
            prompt = ChatPromptTemplate.from_template(DOCUMENT_ONLY_PROMPT_TEMPLATE)
        
        # Step 3: Create and invoke the chain
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
        
        # Step 4: Check if external search is needed and enabled
        if (external_search_enabled and 
            "[EXTERNAL_SEARCH_NEEDED]" in initial_response and 
            external_search_available):
            
            st.info("🔍 Document context insufficient. Searching external sources...")
            
            try:
                # Import here to avoid circular dependency
                from agents.external_sources_lookup_agent import lookup
                
                # Step 5: Perform external search
                external_results = lookup(user_query)
                
                # Step 6: Create enhanced prompt with both contexts
                enhanced_prompt = ChatPromptTemplate.from_template(ENHANCED_PROMPT_TEMPLATE)
                
                # Step 7: Generate enhanced response
                enhanced_chain = enhanced_prompt | language_model | StrOutputParser()
                
                final_response = enhanced_chain.invoke({
                    "user_query": user_query,
                    "document_context": document_context,
                    "external_context": external_results
                })
                
                # Clean up and return enhanced response
                cleaned_response = final_response.strip()
                cleaned_response = '\n'.join(line.strip() for line in cleaned_response.split('\n') if line.strip())
                
                return cleaned_response, sources
                
            except Exception as e:
                st.error(f"External search failed: {str(e)}")
                # Fall back to original response without the search indicator
                fallback_response = initial_response.replace("[EXTERNAL_SEARCH_NEEDED]", "").strip()
                cleaned_response = fallback_response if fallback_response else "I don't have sufficient information to answer this query."
                return cleaned_response, sources
        
        else:
            # Step 8: Return normal response (either external search disabled or not needed)
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
