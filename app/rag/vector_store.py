"""Vector store operations using PGVector."""
import uuid
from typing import List, Dict, Any
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import DocumentChunk
from app.rag.embedder import embed_texts, embed_query
from app.core.logging import get_logger

logger = get_logger(__name__)


async def add_chunks(session: AsyncSession, chunks: List[Dict[str, Any]]) -> None:
    """Embed and store document chunks."""
    texts = [c["content"] for c in chunks]
    embeddings = embed_texts(texts)

    for chunk, emb in zip(chunks, embeddings):
        obj = DocumentChunk(
            id=chunk.get("id", str(uuid.uuid4())),
            doc_id=chunk["metadata"].get("source", "unknown"),
            content=chunk["content"],
            embedding=emb,
            metadata_=chunk.get("metadata", {}),
        )
        session.add(obj)
    await session.commit()
    logger.info("Stored chunks", count=len(chunks))


async def vector_search(
    session: AsyncSession,
    query: str,
    top_k: int = 20,
    filter_metadata: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Cosine similarity search using pgvector <=> operator."""
    q_emb = embed_query(query)
    # pgvector cosine distance
    stmt = (
        select(
            DocumentChunk.id,
            DocumentChunk.content,
            DocumentChunk.metadata_,
            DocumentChunk.embedding.cosine_distance(q_emb).label("distance"),
        )
        .order_by("distance")
        .limit(top_k)
    )
    result = await session.execute(stmt)
    rows = result.all()
    return [
        {
            "id": r.id,
            "content": r.content,
            "metadata": r.metadata_,
            "score": 1 - r.distance,  # convert distance → similarity
            "source": "vector",
        }
        for r in rows
    ]
