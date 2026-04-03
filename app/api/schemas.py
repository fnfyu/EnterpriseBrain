"""Pydantic schemas for API requests and responses."""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = None
    stream: bool = False


class ChatResponse(BaseModel):
    session_id: str
    response: str
    intent: str
    sources: List[Dict[str, Any]] = []
    needs_human: bool = False


class UploadResponse(BaseModel):
    filename: str
    chunks_stored: int
    status: str


class TokenRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
