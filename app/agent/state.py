"""LangGraph AgentState definition."""
from typing import TypedDict, Annotated, Sequence, List, Dict, Any
from langchain_core.messages import BaseMessage
import operator


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]   # conversation history
    user_id: str
    session_id: str
    intent: str                   # "knowledge_query" | "task_execution" | "unclear"
    retrieved_docs: List[Dict]    # RAG results
    tool_calls: List[Dict]        # tool execution history
    final_response: str
    needs_human: bool
    memory_context: Dict[str, Any]
