"""
Chat endpoint: POST /chat
Supports both regular JSON response and SSE streaming.
"""
import uuid
import json
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.messages import HumanMessage
from app.api.schemas import ChatRequest, ChatResponse
from app.api.deps import get_current_user
from app.core.database import get_db
from app.agent.graph import build_graph
from app.agent.state import AgentState
from app.memory.manager import MemoryManager
from app.core.logging import get_logger

router = APIRouter(prefix="/chat", tags=["chat"])
logger = get_logger(__name__)
memory = MemoryManager()


@router.post("", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    user_id = current_user["user_id"]
    session_id = req.session_id or str(uuid.uuid4())

    # Persist incoming message to short-term memory
    await memory.add_message(user_id, session_id, "user", req.message)

    # Load memory context
    mem_ctx = await memory.get_context(db, user_id, session_id, req.message)

    # Build initial state
    initial_state: AgentState = {
        "messages": [HumanMessage(content=req.message)],
        "user_id": user_id,
        "session_id": session_id,
        "intent": "",
        "retrieved_docs": [],
        "tool_calls": [],
        "final_response": "",
        "needs_human": False,
        "memory_context": mem_ctx,
    }

    # Run the graph
    compiled = build_graph(db)
    try:
        final_state = await compiled.ainvoke(initial_state)
    except Exception as e:
        logger.error("Graph execution failed", error=str(e))
        raise HTTPException(status_code=500, detail="处理请求时出现错误，请稍后重试。")

    response_text = final_state.get("final_response") or "已处理您的请求。"

    # Persist assistant response
    await memory.add_message(user_id, session_id, "assistant", response_text)

    if req.stream:
        return _stream_response(response_text, session_id)

    return ChatResponse(
        session_id=session_id,
        response=response_text,
        intent=final_state.get("intent", ""),
        sources=final_state.get("retrieved_docs", [])[:3],
        needs_human=final_state.get("needs_human", False),
    )


def _stream_response(text: str, session_id: str) -> StreamingResponse:
    """SSE streaming: send response word by word."""
    async def generator():
        words = text.split(" ")
        for i, word in enumerate(words):
            chunk = {"session_id": session_id, "delta": word + (" " if i < len(words) - 1 else ""), "done": False}
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")
