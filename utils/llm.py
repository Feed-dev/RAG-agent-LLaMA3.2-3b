import logging
from langchain.llms.base import LLM
import requests
from utils.config import Config

logger = logging.getLogger(__name__)
config = Config()

class OllamaLLM(LLM):
    def __init__(self, model=None):
        self.model = model or config.OLLAMA_MODEL
        self.endpoint = "http://localhost:11434/api/generate"
        logger.info(f"Initialized OllamaLLM with model: {self.model}")

    def invoke(self, prompt, **kwargs):
        try:
            response = requests.post(self.endpoint, json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            })
            response.raise_for_status()
            return response.json()["response"]
        except requests.RequestException as e:
            logger.error(f"Error invoking Ollama LLM: {str(e)}")
            raise

class OllamaLLMForJson(OllamaLLM):
    def invoke(self, prompt, **kwargs):
        try:
            response = super().invoke(prompt, **kwargs)
            # TODO: Add logic to ensure JSON output if needed
            return response
        except Exception as e:
            logger.error(f"Error in OllamaLLMForJson: {str(e)}")
            raise