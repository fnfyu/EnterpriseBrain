"""
Advanced RAG pipeline node for LangGraph.
Multi-path retrieval: Vector + BM25 + Graph → RRF → Rerank
"""
import asyncio
from typing import List, Dict, Any, TYPE_CHECKING
from sqlalchemy.ext.asyncio import AsyncSession
from app.rag.vector_store import vector_search
from app.rag.keyword_search import keyword_search
from app.rag.reranker import reciprocal_rank_fusion, rerank
from app.core.graph_db import GraphStore
from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.agent.state import AgentState

logger = get_logger(__name__)


async def _graph_search(query: str) -> List[Dict[str, Any]]:
    """Entity extraction + graph neighbourhood search."""
    # Simple entity extraction: capitalized words / nouns as entities
    words = query.split()
    entity_names = [w.strip("，。？！") for w in words if len(w) > 1]

    store = GraphStore()
    entities = await store.search_by_entities(entity_names, limit=10)
    results = []
    for i, ent in enumerate(entities):
        results.append({
            "id": f"graph_{ent['id']}",
            "content": f"{ent['name']} ({ent['type']}): " + "; ".join(
                f"{r['rel']} → {r['target']}" for r in (ent.get("relations") or []) if r.get("target")
            ),
            "metadata": {"source": "graph", "entity_type": ent["type"]},
            "score": 1.0 / (i + 1),
            "source": "graph",
        })
    return results


async def _rewrite_queries(query: str, llm) -> List[str]:
    """HyDE-style query rewriting: generate 3 reformulations."""
    prompt = (
        f"请将以下问题改写为3个不同角度的搜索查询，每行一个，不要编号：\n{query}"
    )
    response = await llm.ainvoke(prompt)
    lines = [l.strip() for l in response.content.strip().split("\n") if l.strip()]
    return [query] + lines[:3]


async def rag_pipeline_node(state: "AgentState", db: AsyncSession, llm) -> Dict:
    """LangGraph node: Advanced RAG retrieval."""
    query = state["messages"][-1].content
    logger.info("RAG pipeline started", query=query[:80])

    # 1. Query rewriting
    try:
        queries = await _rewrite_queries(query, llm)
    except Exception:
        queries = [query]

    primary_query = queries[0]

    # 2. Parallel retrieval
    vector_task = vector_search(db, primary_query, top_k=20)
    keyword_task = keyword_search(db, primary_query, top_k=20)
    graph_task = _graph_search(query)

    vector_results, keyword_results, graph_results = await asyncio.gather(
        vector_task, keyword_task, graph_task, return_exceptions=True
    )

    # Filter out exceptions
    all_lists = []
    for res in (vector_results, keyword_results, graph_results):
        if isinstance(res, list):
            all_lists.append(res)
        else:
            logger.warning("Retrieval path failed", error=str(res))

    # 3. RRF fusion
    fused = reciprocal_rank_fusion(all_lists)

    # 4. Cross-encoder rerank
    reranked = rerank(query, fused, top_k=5)

    logger.info("RAG retrieval complete", docs_returned=len(reranked))
    return {"retrieved_docs": reranked}
