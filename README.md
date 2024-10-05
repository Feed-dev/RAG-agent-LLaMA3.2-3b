# Langgraph RAG Agent with Ollama and Llama3.2

This project implements a Retrieval-Augmented Generation (RAG) agent using Langgraph, Ollama, and Llama3.2. The agent can perform document retrieval, web searches, and generate answers based on the retrieved information.

## Prerequisites

- Python 3.8+
- Ollama (for running Llama3.2 locally)
- Elasticsearch (for document storage and retrieval)
- Tavily API key (for web search functionality)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/langgraph_rag_agent.git
   cd langgraph_rag_agent
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
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
   ollama pull llama3
   ```

## Configuration

1. Copy the `.env.example` file to `.env`:
   ```
   cp .env.example .env
   ```

2. Edit the `.env` file and set the following variables:
   - `OLLAMA_MODEL`: The Ollama model to use (default: 'llama3')
   - `ELASTICSEARCH_URL`: URL of your Elasticsearch instance
   - `ELASTICSEARCH_INDEX`: Name of the Elasticsearch index to use
   - `TAVILY_API_KEY`: Your Tavily API key for web search functionality
   - `EMBEDDING_MODEL`: The embedding model to use (default: 'all-MiniLM-L6-v2')
   - `RETRIEVER_K`: Number of documents to retrieve (default: 3)
   - `WEB_SEARCH_K`: Number of web search results to retrieve (default: 3)

## Usage

To run the Langgraph RAG agent, execute the following command:

```
python main.py
```

The agent will process the default question: 'What are the latest developments in AI?'

To use the agent with a different question, modify the `question` variable in the `main.py` file.

## Project Structure

- `main.py`: Entry point of the application
- `agent.py`: Implements the Langgraph agent logic
- `utils/`:
  - `config.py`: Configuration management
  - `llm.py`: Ollama LLM integration
  - `retriever.py`: Document retrieval using Elasticsearch
  - `tools.py`: Web search tool implementation
  - `state.py`: State management for Langgraph

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.