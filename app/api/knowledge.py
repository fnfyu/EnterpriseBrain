"""
Knowledge ingestion endpoint: POST /knowledge/upload
Accepts file upload, parses and stores in vector DB.
"""
import os
import tempfile
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.schemas import UploadResponse
from app.api.deps import get_current_user
from app.core.database import get_db
from app.rag.ingestor import ingest_file
from app.core.logging import get_logger

router = APIRouter(prefix="/knowledge", tags=["knowledge"])
logger = get_logger(__name__)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt", ".txt", ".md"}


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {suffix}")

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        chunks_stored = await ingest_file(tmp_path, db)
    except Exception as e:
        logger.error("Ingest failed", error=str(e), filename=file.filename)
        raise HTTPException(status_code=500, detail=f"文件处理失败: {str(e)}")
    finally:
        os.unlink(tmp_path)

    return UploadResponse(
        filename=file.filename or "",
        chunks_stored=chunks_stored,
        status="success",
    )
