# """This file is going to contain all of the tools that we are going to use in the project"""

import os
from langchain_tavily import TavilySearch
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.logging import get_logger

logger = get_logger("tools")

def search_external_resources(query: str) -> str:
    """
    Searches for external resources about the user's query
    Useful when you need to find up-to-date information about a person, company, or any other topic that is not available in the database
    And want to compare it with the information in the database.
    """
    logger.info(f"Searching external resources for query: {query[:50]}...")
    
    # Check if Tavily API key is available
    if os.getenv("TAVILY_API_KEY"):
        logger.debug("TAVILY_API_KEY found, executing Tavily search")
        search = TavilySearch(max_results=2)
        results = search.run(query)
        logger.info(f"Tavily search completed, returned {len(results)} characters")
        return results
    else:
        # Fallback when no API key is available
        logger.warning("TAVILY_API_KEY not found, returning fallback message")
        return f"External search functionality is not available. Please add TAVILY_API_KEY to your .env file to search for: {query}"