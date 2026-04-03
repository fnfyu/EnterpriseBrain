"""BM25 keyword search over document chunks."""
import json
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import DocumentChunk
from app.core.logging import get_logger

logger = get_logger(__name__)

_bm25_index: BM25Okapi | None = None
_indexed_chunks: List[Dict] = []


async def _build_index(session: AsyncSession):
    global _bm25_index, _indexed_chunks
    result = await session.execute(select(DocumentChunk.id, DocumentChunk.content, DocumentChunk.metadata_))
    rows = result.all()
    _indexed_chunks = [{"id": r.id, "content": r.content, "metadata": r.metadata_} for r in rows]
    tokenized = [r.content.split() for r in rows]
    _bm25_index = BM25Okapi(tokenized)
    logger.info("BM25 index built", size=len(rows))


async def keyword_search(
    session: AsyncSession,
    query: str,
    top_k: int = 20,
) -> List[Dict[str, Any]]:
    global _bm25_index, _indexed_chunks
    if _bm25_index is None:
        await _build_index(session)

    tokens = query.split()
    scores = _bm25_index.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            chunk = _indexed_chunks[idx]
            results.append({
                "id": chunk["id"],
                "content": chunk["content"],
                "metadata": chunk["metadata"],
                "score": float(scores[idx]),
                "source": "keyword",
            })
    return results


def invalidate_index():
    """Call after adding new documents."""
    global _bm25_index, _indexed_chunks
    _bm25_index = None
    _indexed_chunks = []
