# --- Import Section ---
# Standard library imports
import os
import getpass
import json
import operator

# Typing extensions and typing imports
from typing_extensions import TypedDict
from typing import List, Annotated

# Third-party library imports for Langchain, Ollama, IPython, and Tavily
from langchain_ollama import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from langgraph.graph import END, StateGraph
from IPython.display import Image, display

# --- LLM Setup ---
# Initialize the local language model and its JSON mode
local_llm = 'llama3.2:3b-instruct-fp16'
llm = ChatOllama(model=local_llm, temperature=0)
llm_json_mode = ChatOllama(model=local_llm, temperature=0, format='json')

# --- Environment Setup ---
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

# Set necessary environment variables
_set_env("TAVILY_API_KEY")
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
_set_env("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "local-llama32-rag"
_set_env("USER_AGENT")  # Added to prompt for USER_AGENT variable

# --- Data Preparation ---
# Define URLs to load documents
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Load documents from URLs
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Split documents using RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=200
)
doc_splits = text_splitter.split_documents(docs_list)

# --- Vectorstore Creation ---
# Add documents to vectorstore using Nomic Embeddings
vectorstore = SKLearnVectorStore.from_documents(
    documents=doc_splits,
    embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local"),
)

# Create a retriever
retriever = vectorstore.as_retriever(k=3)

# --- Router Logic ---
router_instructions = """You are an expert at routing a user question to a vectorstore or web search.

The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.

Use the vectorstore for questions on these topics. For all else, and especially for current events, use web-search.

Return JSON with a single key, datasource, that is 'websearch' or 'vectorstore' depending on the question."""

# --- Grader Logic ---
doc_grader_instructions = """You are a grader assessing relevance of a retrieved document to a user question.

If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."""
doc_grader_prompt = """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}. 
This carefully and objectively assess whether the document contains at least some information that is relevant to the question.
Return JSON with a single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains at least some information that is relevant to the question."""

# --- Answer Generation Logic ---
rag_prompt = """You are an assistant for question-answering tasks. 

Here is the context to use to answer the question:

{context} 

Think carefully about the above context. 

Now, review the user question:

{question}

Provide an answer to this questions using only the above context. 

Use three sentences maximum and keep the answer concise.

Answer:"""

# --- Post-Processing ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- Search Tool ---
web_search_tool = TavilySearchResults(k=3)

# --- Graph Setup ---
class GraphState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """
    question : str
    generation : str
    web_search : str
    max_retries : int
    answers : int
    loop_step: Annotated[int, operator.add]
    documents : List[str]

### Nodes and Functions

def retrieve(state):
    """Retrieve documents from vectorstore."""
    print("---RETRIEVE---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents}

def generate(state):
    """Generate answer using RAG on retrieved documents."""
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    loop_step = state.get("loop_step", 0)
    docs_txt = format_docs(documents)
    rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    return {"generation": generation, "loop_step": loop_step + 1}

# Similar formatting would follow for other nodes and workflow construction.

### Nodes and Functions (continued)

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.
    Filters out irrelevant documents and updates the web_search flag.
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        doc_grader_prompt_formatted = doc_grader_prompt.format(document=d.page_content, question=question)
        result = llm_json_mode.invoke(
            [SystemMessage(content=doc_grader_instructions)] + [HumanMessage(content=doc_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)['binary_score']

        # Document relevance check
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "web_search": web_search}

def web_search(state):
    """
    Perform a web search based on the question and append the results to the documents.
    """
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])

    # Web search using Tavily
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)
    return {"documents": documents}

### Routing Logic

def route_question(state):
    """
    Route the question to either web search or the vectorstore for RAG.
    """
    print("---ROUTE QUESTION---")
    route_question = llm_json_mode.invoke(
        [SystemMessage(content=router_instructions)] + [HumanMessage(content=state["question"])]
    )
    source = json.loads(route_question.content)['datasource']

    if source == 'websearch':
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source == 'vectorstore':
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"

def decide_to_generate(state):
    """
    Decide whether to generate an answer based on the graded documents or add a web search.
    """
    print("---ASSESS GRADED DOCUMENTS---")
    web_search = state["web_search"]

    if web_search == "Yes":
        print("---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
        return "websearch"
    else:
        print("---DECISION: GENERATE---")
        return "generate"

def grade_generation_v_documents_and_question(state):
    """
    Grade the generated answer by comparing it to the documents and the question.
    """
    print("---CHECK HALLUCINATIONS---")
    documents = state["documents"]
    generation = state["generation"]
    max_retries = state.get("max_retries", 3)

    hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
        documents=format_docs(documents), generation=generation.content
    )
    result = llm_json_mode.invoke(
        [SystemMessage(content=hallucination_grader_instructions)] + [
            HumanMessage(content=hallucination_grader_prompt_formatted)]
    )
    grade = json.loads(result.content)['binary_score']

    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        answer_grader_prompt_formatted = answer_grader_prompt.format(question=state["question"],
                                                                     generation=generation.content)
        result = llm_json_mode.invoke(
            [SystemMessage(content=answer_grader_instructions)] + [HumanMessage(content=answer_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)['binary_score']

        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        elif state["loop_step"] <= max_retries:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
        else:
            print("---DECISION: MAX RETRIES REACHED---")
            return "max retries"
    elif state["loop_step"] <= max_retries:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
    else:
        print("---DECISION: MAX RETRIES REACHED---")
        return "max retries"

### Workflow Setup

workflow = StateGraph(GraphState)

# Define nodes in the workflow
workflow.add_node("websearch", web_search)  # Web search
workflow.add_node("retrieve", retrieve)  # Retrieve documents
workflow.add_node("grade_documents", grade_documents)  # Grade documents
workflow.add_node("generate", generate)  # Generate answer

# Define workflow edges
workflow.set_conditional_entry_point(
    route_question,
    {
        "websearch": "websearch",
        "vectorstore": "retrieve",
    }
)
workflow.add_edge("websearch", "generate")  # Connect web search to generation
workflow.add_edge("retrieve", "grade_documents")  # Connect retrieval to grading documents
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    }
)
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "websearch",
        "max retries": END,
    }
)

# Compile and Display Graph
graph = workflow.compile()
display(Image(graph.get_graph().draw_mermaid_png()))

# Example inputs to test the workflow
'''
inputs = {"question": "What are the types of agent memory?", "max_retries": 3}
for event in graph.stream(inputs, stream_mode="values"):
    print(event)
'''
