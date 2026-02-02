"""Language model initialization"""

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from .settings import OPENAI_API_KEY, ANTHROPIC_API_KEY, LLM_TEMPERATURE, LLM_MAX_TOKENS


class APIKeyError(Exception):
    """Raised when required API key is not configured"""
    pass


def initialize_language_model(model_choice: str):
    """
    Initialize the chosen language model
    
    Args:
        model_choice: Model name from MODEL_OPTIONS keys
        
    Returns:
        Initialized language model instance
        
    Raises:
        APIKeyError: If the required API key is not configured
    """
    if model_choice == "GPT-4.1 mini":
        if not OPENAI_API_KEY:
            raise APIKeyError(
                "OpenAI API key is not configured. Please set OPENAI_API_KEY in your .env file."
            )
        return ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model="gpt-4.1-mini",
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS
        )
    else:  # Claude Haiku 4.5
        if not ANTHROPIC_API_KEY:
            raise APIKeyError(
                "Anthropic API key is not configured. Please set ANTHROPIC_API_KEY in your .env file."
            )
        return ChatAnthropic(
            api_key=ANTHROPIC_API_KEY,
            model="claude-haiku-4-5-20251001",
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS
        )
