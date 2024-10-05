from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict

class GraphState(TypedDict):
    question: str
    context: List[str]
    current_step: str
    final_answer: str
    retriever: Any
    web_search_tool: Any
    error: Optional[str]
    selected_namespaces: List[str]
    web_search_results: List[str]
