import logging
import json
from typing import List, Tuple
from langgraph.graph import StateGraph, END
from langchain.prompts import ChatPromptTemplate
from utils.llm import OllamaLLM, OllamaLLMForJson
from utils.retriever import initialize_pinecone, get_pinecone_index, create_retriever
from utils.tools import create_web_search_tool
from utils.state import GraphState
from utils.config import Config
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import HumanMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = Config()


def create_agent(retriever, web_search_tool):
    workflow = StateGraph(GraphState)

    # Define nodes
    workflow.add_node("web_search", web_search)
    workflow.add_node("determine_namespace", determine_namespace)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate_answer", generate_answer)

    # Define entry
    workflow.set_entry_point("web_search")

    # Define edges and logic
    workflow.add_edge("web_search", "determine_namespace")
    workflow.add_edge("determine_namespace", "retrieve")
    workflow.add_edge("retrieve", "generate_answer")
    workflow.add_edge("generate_answer", END)

    # Initialize the state
    initial_state = GraphState(
        question="",
        context=[],
        current_step="",
        final_answer="",
        retriever=retriever,
        web_search_tool=web_search_tool,
        error=None,
        selected_namespaces=[],
        web_search_results=[]
    )

    return workflow.compile()


def retrieve(state: GraphState) -> GraphState:
    try:
        logger.info("Starting retrieval process")
        query = state["question"]

        all_docs = []
        for namespace in state["selected_namespaces"]:
            # Create a retriever for each selected namespace
            retriever = create_retriever(config.INDEX_NAME, namespace)

            # Use the retriever to get relevant documents
            docs = retriever.get_relevant_documents(query)
            all_docs.extend(docs)

        # Extract page content from documents
        state["context"].extend([doc.page_content for doc in all_docs])
        state["context"].extend(state["web_search_results"])  # Add web search results to context
        state["current_step"] = "retrieve"
        logger.info(f"Retrieved {len(all_docs)} documents from {len(state['selected_namespaces'])} namespaces")
        return state
    except Exception as e:
        logger.error(f"Error in retrieve function: {str(e)}")
        state["error"] = str(e)
    finally:
        return state


def web_search(state: GraphState) -> GraphState:
    try:
        logger.info("Starting web search process")
        query = state["question"]
        search_results = state["web_search_tool"].run(query)
        state["web_search_results"] = search_results
        state["current_step"] = "web_search"
        logger.info(f"Web search completed, found {len(search_results)} results")
        return state
    except Exception as e:
        logger.error(f"Error in web_search function: {str(e)}")
        state["error"] = str(e)
    finally:
        return state


def determine_namespace(state: GraphState) -> GraphState:
    try:
        logger.info("Starting namespace determination process")
        query = state["question"]
        search_results = state["web_search_results"]

        # Get list of available namespaces
        available_namespaces = get_available_namespaces(state["retriever"])

        if len(available_namespaces) == 1:
            # If there's only one namespace (which would be the default/index name), use it
            state["selected_namespaces"] = available_namespaces
            logger.info(f"Using default namespace: {available_namespaces[0]}")
        else:
            # Use LLM to determine relevant namespaces
            relevant_namespaces = llm_determine_namespaces(state, query, search_results, available_namespaces)
            state["selected_namespaces"] = relevant_namespaces

        state["current_step"] = "determine_namespace"
        logger.info(f"Selected namespaces: {state['selected_namespaces']}")
        return state
    except Exception as e:
        logger.error(f"Error in determine_namespace function: {str(e)}")
        state["error"] = str(e)
        return state


def get_available_namespaces(retriever):
    try:
        # Initialize Pinecone
        pc = initialize_pinecone()

        # Get the Pinecone index
        pinecone_index = get_pinecone_index(pc)

        # Get the index stats
        stats = pinecone_index.describe_index_stats()

        # Extract and return the list of namespaces
        namespaces = list(stats['namespaces'].keys())
        logger.info(f"Available namespaces: {namespaces}")
        return namespaces
    except AttributeError as e:
        logger.error(f"Error accessing Pinecone index: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Error retrieving namespaces: {str(e)}")
        return []


def llm_determine_namespaces(state: GraphState, query: str, search_results: List[any],
                             available_namespaces: List[str]) -> List[str]:
    llm = OllamaLLMForJson(model=config.OLLAMA_MODEL)

    # Define the response schema
    response_schemas = [
        ResponseSchema(name="selected_namespaces",
                       description="A list of 1 to 3 most relevant namespace names"),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    # Convert search results to strings
    search_results_str = []
    for result in search_results:
        if isinstance(result, dict):
            search_results_str.append(json.dumps(result))
        elif isinstance(result, str):
            search_results_str.append(result)
        else:
            search_results_str.append(str(result))

    # Create the prompt template
    prompt = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template(
                """Based on the following information, determine the most relevant namespaces for the given query.
                Select between 1 and 3 namespaces that best match the context of the query and search results.

                Query: {query}

                Search Results:
                {search_results}

                Available Namespaces:
                {namespaces}

                {format_instructions}

                Your response:"""
            )
        ],
        input_variables=["query", "search_results", "namespaces"],
        partial_variables={"format_instructions": format_instructions}
    )

    # Create the chain
    chain = prompt | llm | output_parser

    # Invoke the chain
    response = chain.invoke({
        "query": query,
        "search_results": "\n".join(search_results_str[:5]),  # Limit to first 5 results for brevity
        "namespaces": ", ".join(available_namespaces)
    })

    # Extract the selected namespaces from the structured output
    selected_namespaces = response.get("selected_namespaces", [])

    # Ensure we return at most 3 namespaces
    return selected_namespaces[:3]


def generate_answer(state: GraphState) -> GraphState:
    try:
        logger.info("Starting answer generation process")
        llm = OllamaLLM(model=config.OLLAMA_MODEL)

        # Convert all context items to strings
        context_strings = []
        for item in state["context"]:
            if isinstance(item, dict):
                # If the item is a dictionary, convert it to a string representation
                context_strings.append(str(item))
            elif isinstance(item, str):
                context_strings.append(item)
            else:
                # For any other type, convert to string
                context_strings.append(str(item))

        prompt = ChatPromptTemplate.from_template(
            "Based on the following context, answer the question: {question}\n\nContext: {context}"
        )

        chain = prompt | llm
        response = chain.invoke({
            "question": state["question"],
            "context": "\n".join(context_strings)
        })

        state["final_answer"] = response
        state["current_step"] = "generate_answer"
        logger.info("Answer generation completed")
        return state
    except Exception as e:
        logger.error(f"Error in generate_answer function: {str(e)}")
        state["error"] = str(e)
    finally:
        return state
