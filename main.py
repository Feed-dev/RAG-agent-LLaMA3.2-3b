import logging
from agent import create_agent
from utils.retriever import create_retriever
from utils.tools import create_web_search_tool
from utils.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # Load configuration
        config = Config()
        logger.info(f"Loaded configuration: {config}")

        # Initialize components
        retriever = create_retriever(config.ELASTICSEARCH_INDEX, config.EMBEDDING_MODEL)
        web_search_tool = create_web_search_tool(config.TAVILY_API_KEY)
        
        # Create agent
        agent = create_agent(retriever, web_search_tool)
        
        # Run agent
        question = 'What are the latest developments in AI?'
        logger.info(f"Running agent with question: {question}")
        result = agent.invoke({'question': question})
        logger.info(f"Agent result: {result}")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()