"""
LangGraph StateGraph definition.
Wires together all nodes with conditional routing.
"""
from functools import partial
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from app.agent.state import AgentState
from app.agent.nodes import (
    intent_classifier_node,
    rag_node,
    agent_executor_node,
    human_handoff_node,
    response_generator_node,
    route_by_intent,
)
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


def build_graph(db_session):
    """
    Build and compile the LangGraph StateGraph.
    db_session: SQLAlchemy AsyncSession injected at request time.
    """
    graph = StateGraph(AgentState)

    # Bind db session to nodes that need it
    rag_with_db = partial(rag_node, db=db_session)

    # Add nodes
    graph.add_node("intent_classifier", intent_classifier_node)
    graph.add_node("rag_pipeline", rag_with_db)
    graph.add_node("agent_executor", agent_executor_node)
    graph.add_node("human_handoff", human_handoff_node)
    graph.add_node("response_generator", response_generator_node)

    # Entry point
    graph.set_entry_point("intent_classifier")

    # Conditional routing from intent classifier
    graph.add_conditional_edges(
        "intent_classifier",
        route_by_intent,
        {
            "knowledge_query": "rag_pipeline",
            "task_execution": "agent_executor",
            "unclear": "human_handoff",
        },
    )

    # RAG → response generator
    graph.add_edge("rag_pipeline", "response_generator")

    # Agent executor → response generator (agent already generated final message)
    graph.add_edge("agent_executor", "response_generator")

    # Human handoff → END (response is the handoff message)
    graph.add_edge("human_handoff", END)

    # Response generator → END
    graph.add_edge("response_generator", END)

    return graph.compile()


async def get_checkpointer():
    """PostgreSQL-backed LangGraph checkpointer for cross-session memory."""
    return AsyncPostgresSaver.from_conn_string(settings.database_url)
