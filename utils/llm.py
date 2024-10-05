import logging
from typing import Any, List, Optional, Mapping
from langchain.llms.base import LLM
import requests
from pydantic import Field
from utils.config import Config

logger = logging.getLogger(__name__)
config = Config()

class OllamaLLM(LLM):
    model: str = Field(..., description="The Ollama model to use")
    endpoint: str = Field(default="http://localhost:11434/api/generate")

    def __init__(self, model: Optional[str] = None, **kwargs):
        model = model or config.OLLAMA_MODEL
        super().__init__(model=model, **kwargs)
        logger.info(f"Initialized OllamaLLM with model: {self.model}")

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[Any] = None, **kwargs: Any) -> str:
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

    @property
    def _llm_type(self) -> str:
        return "ollama"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.model}

class OllamaLLMForJson(OllamaLLM):
    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[Any] = None, **kwargs: Any) -> str:
        try:
            response = super()._call(prompt, stop, run_manager, **kwargs)
            # TODO: Add logic to ensure JSON output if needed
            return response
        except Exception as e:
            logger.error(f"Error in OllamaLLMForJson: {str(e)}")
            raise
