from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict

class GraphState(TypedDict):
    question: str
    context: List[str]
    current_step: str
    final_answer: Optional[str]
    retriever: Any
    web_search_tool: Any
    error: Optional[str]
