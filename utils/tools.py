import logging
from langchain.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from utils.config import Config

logger = logging.getLogger(__name__)
config = Config()

def create_web_search_tool(api_key):
    try:
        logger.info("Creating web search tool")
        tool = Tool(
            name="Web Search",
            func=TavilySearchResults(api_key=api_key, k=config.WEB_SEARCH_K),
            description="Useful for searching the web for current information."
        )
        logger.info("Web search tool created successfully")
        return tool
    except Exception as e:
        logger.error(f"Error creating web search tool: {str(e)}")
        raise