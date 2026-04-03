"""
Three-tier memory architecture:
  Short-term  → Redis  (current session messages, TTL 24h)
  Mid-term    → PostgresCheckpointer (cross-session recent history, via LangGraph)
  Long-term   → PGVector (persistent facts & user preferences)
"""
import json
import uuid
from typing import List, Dict, Any
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.redis_client import get_redis
from app.core.database import UserProfile
from app.rag.vector_store import vector_search
from app.rag.embedder import embed_texts
from app.core.database import DocumentChunk
from app.core.logging import get_logger

logger = get_logger(__name__)

_SESSION_TTL = 86_400  # 24 hours


class MemoryManager:
    # ── Short-term (Redis) ────────────────────────────────────────────────────

    async def add_message(self, user_id: str, session_id: str, role: str, content: str):
        redis = get_redis()
        key = f"session:{user_id}:{session_id}:messages"
        msg = json.dumps({"role": role, "content": content})
        await redis.rpush(key, msg)
        await redis.expire(key, _SESSION_TTL)

    async def get_recent_messages(
        self, user_id: str, session_id: str, last_k: int = 10
    ) -> List[Dict[str, str]]:
        redis = get_redis()
        key = f"session:{user_id}:{session_id}:messages"
        raw = await redis.lrange(key, -last_k, -1)
        return [json.loads(m) for m in raw]

    async def clear_session(self, user_id: str, session_id: str):
        redis = get_redis()
        await redis.delete(f"session:{user_id}:{session_id}:messages")

    # ── Long-term facts (PGVector) ────────────────────────────────────────────

    async def save_fact(self, session: AsyncSession, user_id: str, fact: str):
        """Store a user-specific fact as a vector chunk."""
        chunk = {
            "id": str(uuid.uuid4()),
            "content": fact,
            "metadata": {"user_id": user_id, "type": "fact"},
        }
        from app.rag.vector_store import add_chunks
        await add_chunks(session, [chunk])
        logger.info("Fact saved", user_id=user_id)

    async def retrieve_relevant_facts(
        self, session: AsyncSession, user_id: str, query: str, top_k: int = 5
    ) -> List[str]:
        results = await vector_search(session, query, top_k=top_k * 2)
        facts = [
            r["content"]
            for r in results
            if r.get("metadata", {}).get("user_id") == user_id
            and r.get("metadata", {}).get("type") == "fact"
        ]
        return facts[:top_k]

    # ── User profile (PostgreSQL) ─────────────────────────────────────────────

    async def get_user_profile(self, session: AsyncSession, user_id: str) -> Dict[str, Any]:
        result = await session.execute(
            select(UserProfile).where(UserProfile.user_id == user_id)
        )
        profile = result.scalar_one_or_none()
        if not profile:
            return {"user_id": user_id, "name": "", "department": "", "preferences": {}}
        return {
            "user_id": profile.user_id,
            "name": profile.name,
            "department": profile.department,
            "preferences": profile.preferences,
        }

    async def update_user_profile(
        self, session: AsyncSession, user_id: str, updates: Dict[str, Any]
    ):
        result = await session.execute(
            select(UserProfile).where(UserProfile.user_id == user_id)
        )
        profile = result.scalar_one_or_none()
        if not profile:
            profile = UserProfile(user_id=user_id)
            session.add(profile)

        for key, value in updates.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
        await session.commit()

    # ── Aggregate context ─────────────────────────────────────────────────────

    async def get_context(
        self,
        session: AsyncSession,
        user_id: str,
        session_id: str,
        query: str,
    ) -> Dict[str, Any]:
        recent, facts, profile = await _gather(
            self.get_recent_messages(user_id, session_id),
            self.retrieve_relevant_facts(session, user_id, query),
            self.get_user_profile(session, user_id),
        )
        return {
            "recent_messages": recent,
            "relevant_facts": facts,
            "user_profile": profile,
        }


async def _gather(*coros):
    import asyncio
    return await asyncio.gather(*coros)
