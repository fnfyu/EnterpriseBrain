"""Ingest a document: parse → embed → store in PGVector + Neo4j."""
import uuid
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession
from app.rag.parser import parse_document
from app.rag.vector_store import add_chunks
from app.rag.keyword_search import invalidate_index
from app.core.graph_db import GraphStore
from app.core.logging import get_logger

logger = get_logger(__name__)


async def ingest_file(file_path: str, session: AsyncSession) -> int:
    """Parse, embed, and store a file. Returns number of chunks stored."""
    logger.info("Ingesting file", path=file_path)
    chunks = await parse_document(file_path)
    if not chunks:
        logger.warning("No chunks extracted", path=file_path)
        return 0

    # Store in vector DB
    await add_chunks(session, chunks)
    invalidate_index()  # BM25 must be rebuilt

    logger.info("Ingestion complete", path=file_path, chunks=len(chunks))
    return len(chunks)
