import logging
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import CohereEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from utils.config import Config
from typing import List

logger = logging.getLogger(__name__)
config = Config()


class CustomCohereEmbeddings(CohereEmbeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = [str(text) for text in texts]
        return super().embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return super().embed_query(str(text))


def initialize_pinecone():
    return Pinecone(api_key=config.PINECONE_API_KEY)


def get_pinecone_index(pc):
    return pc.Index(config.INDEX_NAME)


def get_embeddings():
    return CustomCohereEmbeddings(
        model="embed-multilingual-v3.0",
        cohere_api_key=config.COHERE_API_KEY
    )


def create_retriever(index_name, namespace=None):
    try:
        logger.info(f"Creating retriever with index: {index_name}")

        # Initialize Pinecone
        pc = initialize_pinecone()
        index = get_pinecone_index(pc)

        # Create embeddings
        embeddings = get_embeddings()

        # Create vector store
        vector_store = PineconeVectorStore(index, embeddings, text_key="text")

        # Set up search parameters
        search_kwargs = {"k": config.RETRIEVER_K}
        if namespace:
            search_kwargs["namespace"] = namespace

        # Create base retriever
        base_retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
        logger.info(f"Created base retriever with search_kwargs: {search_kwargs}")

        # Set up LLM for contextual compression
        llm = config.get_llm()  # You'll need to implement this method in your Config class
        compressor = LLMChainExtractor.from_llm(llm)

        # Create contextual compression retriever
        retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)

        logger.info("Retriever created successfully")
        return retriever
    except Exception as e:
        logger.error(f"Error creating retriever: {str(e)}")
        raise
