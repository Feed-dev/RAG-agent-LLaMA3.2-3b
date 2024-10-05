import os

class Config:
    def __init__(self):
        self.OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3')
        self.ELASTICSEARCH_URL = os.getenv('ELASTICSEARCH_URL', 'http://localhost:9200')
        self.ELASTICSEARCH_INDEX = os.getenv('ELASTICSEARCH_INDEX', 'your_index_name')
        self.TAVILY_API_KEY = os.getenv('TAVILY_API_KEY', 'your_tavily_api_key')
        self.EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        self.RETRIEVER_K = int(os.getenv('RETRIEVER_K', '3'))
        self.WEB_SEARCH_K = int(os.getenv('WEB_SEARCH_K', '3'))

    def __repr__(self):
        return f"Config(OLLAMA_MODEL={self.OLLAMA_MODEL}, ELASTICSEARCH_URL={self.ELASTICSEARCH_URL}, ELASTICSEARCH_INDEX={self.ELASTICSEARCH_INDEX}, EMBEDDING_MODEL={self.EMBEDDING_MODEL}, RETRIEVER_K={self.RETRIEVER_K}, WEB_SEARCH_K={self.WEB_SEARCH_K})"
