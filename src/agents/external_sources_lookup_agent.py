import os
import sys

# Add the parent directory to the Python path so we can import from tools
# This must be done BEFORE importing tools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from tools.tools import search_external_resources
from config.logging import get_logger

load_dotenv()

logger = get_logger("external_lookup")


# Create a function to create a tool for searching external resources
def lookup(information_to_lookup: str) -> str:
    logger.info(f"Starting external search for: {information_to_lookup[:50]}...")
    
    # Initialize the LLM
    logger.debug("Initializing ChatOpenAI with gpt-4.1 for external search")
    llm = ChatOpenAI(model="gpt-4.1", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

    # Create a tool using decorator
    @tool
    def search_external_resources_tool(query: str) -> str:
        """Search for up-to-date information about a person, company, or any topic.
        
        Args:
            query: The information to search for
        """
        logger.debug(f"Executing search_external_resources tool for query: {query[:50]}...")
        result = search_external_resources(query)
        logger.debug(f"Search tool returned result with length: {len(result)}")
        return result

    tools = [search_external_resources_tool]

    # Create agent with system prompt
    logger.info("Creating agent for external search")
    agent = create_agent(
        llm,
        tools,
        system_prompt=f"""Given the information about {information_to_lookup}, search for external resources about it.
Your answer should only contain the results of the search and nothing else."""
    )

    # Run the agent
    logger.info("Running external search agent")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": information_to_lookup}]}
    )
    
    final_content = result["messages"][-1].content
    logger.info(f"External search completed. Result length: {len(final_content)} characters")

    return final_content

# Only use this if you are running the script directly
if __name__ == "__main__":
    # Run the lookup function
    result = lookup("Iran israel war updates")
    print(result)
