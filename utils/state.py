from typing import List, Dict, Any
from typing_extensions import TypedDict

class GraphState(TypedDict):
    question: str
    context: List[str]
    current_step: str
    final_answer: str