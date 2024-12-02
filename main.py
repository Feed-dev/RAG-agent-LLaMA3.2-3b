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

        # Initialize components
        retriever = create_retriever(config.INDEX_NAME)
        web_search_tool = create_web_search_tool(config.TAVILY_API_KEY)

        # Create agent
        agent = create_agent(retriever, web_search_tool)

        while True:
            # Prompt user for question
            question = input("Please enter your question (or type 'exit' to quit): ")

            if question.lower() == 'exit':
                print("Exiting the program. Goodbye!")
                break

            logger.info(f"Running agent with question: {question}")

            # Initialize the GraphState with all required fields
            initial_state = GraphState(
                question=question,
                context=[],
                current_step="",
                final_answer="",
                retriever=retriever,
                web_search_tool=web_search_tool,
                error=None,
                selected_namespaces=[],
                web_search_results=[]
            )

            result = agent.invoke(initial_state)
            logger.info(f"Agent result: {result}")

            # Print final answer
            if result.get("final_answer"):
                print(f"\nAnswer: {result['final_answer']}")
            elif result.get("error"):
                print(f"\nError occurred: {result['error']}")
            else:
                print("\nNo answer or error was returned.")

            print("\n" + "-" * 50 + "\n")  # Add a separator between questions

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")


if __name__ == '__main__':
    main()
