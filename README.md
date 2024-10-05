# Langgraph RAG Agent with Ollama and Llama3.2

This project implements a Retrieval-Augmented Generation (RAG) agent using Langgraph, Ollama, and Llama3.2. The agent can perform document retrieval, web searches, and generate answers based on the retrieved information.

## Prerequisites

- Python 3.8+
- Ollama (for running Llama3.2 locally)
- Pinecone (for vector storage and retrieval)
- Tavily API key (for web search functionality)

## Installation

1. Clone this repository:

   ```
   git clone https://github.com/yourusername/langgraph_rag_agent.git
   cd langgraph_rag_agent
   ```

2. Create a virtual environment and activate it:

   ```
   python -m venv .venv
   source .venv/bin/activate # On Windows, use `.venv\Scripts\activate`
   ```

3. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

4. Install Ollama:

   ```
   curl -fsSL https://ollama.com/install.sh | sh
   ```

5. Pull the Llama3.2 model:

   ```
   ollama pull llama3.2:3b
   ```

## Configuration

1. Create a `.env` file in the project root directory.

2. Set the following environment variables in the `.env` file:

   - `PINECONE_API_KEY`: Your Pinecone API key
   - `OLLAMA_MODEL`: llama3.2:3b
   - `PINECONE_INDEX_NAME`: Name of your Pinecone index
   - `COHERE_API_KEY`: Your Cohere API key for embeddings
   - `TAVILY_API_KEY`: Your Tavily API key for web search functionality
   - `EMBEDDING_MODEL`: embed-multilingual-v3.0
   - `RETRIEVER_K`: set the amount of docs u want to retrieve per query
   - `WEB_SEARCH_K`: set the amount of web searches for one query

## Usage

To run the Langgraph RAG agent, execute the following command:

``` 
python main.py 
```

The agent will prompt you to enter a question and a namespace for the Pinecone index.

## Project Structure

- `main.py`: Entry point of the application
- `agent.py`: Implements the Langgraph agent logic
- `utils/`:
  - `config.py`: Configuration management
  - `llm.py`: Ollama LLM integration
  - `retriever.py`: Document retrieval using Pinecone
  - `tools.py`: Web search tool implementation
  - `state.py`: State management for Langgraph

## Key Components

- **Retriever**: Uses Pinecone for vector storage and retrieval, with Cohere embeddings.
- **Web Search**: Utilizes Tavily for web search functionality.
- **LLM**: Integrates Ollama to run Llama3.2 locally for answer generation.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.