import logging
from langchain_elasticsearch import ElasticsearchStore
from langchain.embeddings import HuggingFaceEmbeddings
from utils.config import Config

logger = logging.getLogger(__name__)
config = Config()

def create_retriever(index_name, embedding_model):
    try:
        logger.info(f"Creating retriever with index: {index_name}, embedding model: {embedding_model}")
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        vector_store = ElasticsearchStore(
            es_url=config.ELASTICSEARCH_URL,
            index_name=index_name,
            embedding=embeddings
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": config.RETRIEVER_K})
        logger.info("Retriever created successfully")
        return retriever
    except Exception as e:
        logger.error(f"Error creating retriever: {str(e)}")
        raise