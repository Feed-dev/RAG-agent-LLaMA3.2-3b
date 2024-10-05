import os
from dotenv import load_dotenv


class Config:
    def __init__(self):
        load_dotenv()  # Load environment variables from .env file

        # Index settings
        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        self.INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

        # Cohere settings
        self.COHERE_API_KEY = os.getenv("COHERE_API_KEY")
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embed-multilingual-v3.0")

        # Ollama settings
        self.OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

        # Retriever settings
        self.RETRIEVER_K = int(os.getenv("RETRIEVER_K", "7"))

        # Optional: Keep these if you still need them for other parts of your application
        self.TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "your_tavily_api_key")
        self.WEB_SEARCH_K = int(os.getenv("WEB_SEARCH_K", "3"))

    def __repr__(self):
        return (f"Config("
                f"PINECONE_INDEX_NAME={self.INDEX_NAME}, "
                f"COHERE_EMBEDDING_MODEL={self.EMBEDDING_MODEL}, "
                f"OLLAMA_MODEL={self.OLLAMA_MODEL}, "
                f"RETRIEVER_K={self.RETRIEVER_K})")

    def get_llm(self):
        from langchain_community.llms import Ollama
        from langchain.callbacks.manager import CallbackManager
        from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

        return Ollama(
            model=self.OLLAMA_MODEL,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
        )
