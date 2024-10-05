import logging
from agent import create_agent
from utils.retriever import create_retriever
from utils.tools import create_web_search_tool
from utils.config import Config
from utils.state import GraphState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # Load configuration
        config = Config()
        logger.info(f"Loaded configuration: {config}")

        # Prompt user for question and namespace
        question = input("Please enter your question: ")
        namespace = input("Please enter the namespace (press Enter for default): ").strip() or config.EMBEDDING_MODEL

        # Initialize components
        retriever = create_retriever(config.INDEX_NAME, namespace)
        web_search_tool = create_web_search_tool(config.TAVILY_API_KEY)

        # Create agent
        agent = create_agent(retriever, web_search_tool)

        logger.info(f"Running agent with question: {question}")
        logger.info(f"Using namespace: {namespace}")

        # Initialize the GraphState with all required fields
        initial_state = GraphState(
            question=question,
            context=[],
            current_step="",
            final_answer="",
            retriever=retriever,
            web_search_tool=web_search_tool,
            error=None
        )

        result = agent.invoke(initial_state)
        logger.info(f"Agent result: {result}")

        # Print the final answer
        if result.get("final_answer"):
            print(f"\nAnswer: {result['final_answer']}")
        elif result.get("error"):
            print(f"\nError occurred: {result['error']}")
        else:
            print("\nNo answer or error was returned.")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()
