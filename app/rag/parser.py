"""Document parsing: PDF, Word, Excel, PPT, scanned images."""
import uuid
from pathlib import Path
from typing import List, Dict, Any
from app.core.config import get_settings
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)


def _parse_with_unstructured(file_path: str) -> List[Dict[str, Any]]:
    from unstructured.partition.auto import partition

    elements = partition(filename=file_path)
    chunks = []
    current_section = ""

    for el in elements:
        category = el.category
        text = str(el).strip()
        if not text:
            continue

        if category in ("Title", "Header"):
            current_section = text
            continue

        chunks.append({
            "id": str(uuid.uuid4()),
            "content": text,
            "metadata": {
                "source": file_path,
                "section": current_section,
                "category": category,
            },
        })
    return chunks


async def _parse_with_llamaparse(file_path: str) -> List[Dict[str, Any]]:
    """Use LlamaParse for complex PDFs (scanned, multi-column)."""
    from llama_parse import LlamaParse

    parser = LlamaParse(
        api_key=settings.llama_cloud_api_key,
        result_type="markdown",
        language="ch_sim",  # simplified Chinese + English
    )
    documents = await parser.aload_data(file_path)
    chunks = []
    for doc in documents:
        # Semantic chunking: split by paragraph
        paragraphs = [p.strip() for p in doc.text.split("\n\n") if p.strip()]
        for para in paragraphs:
            chunks.append({
                "id": str(uuid.uuid4()),
                "content": para,
                "metadata": {
                    "source": file_path,
                    "page": doc.metadata.get("page_label", ""),
                },
            })
    return chunks


async def parse_document(file_path: str) -> List[Dict[str, Any]]:
    """Entry point: choose parser based on file type."""
    suffix = Path(file_path).suffix.lower()
    pdf_suffixes = {".pdf"}
    llamaparse_enabled = bool(settings.llama_cloud_api_key)

    if suffix in pdf_suffixes and llamaparse_enabled:
        logger.info("Parsing with LlamaParse", path=file_path)
        return await _parse_with_llamaparse(file_path)
    else:
        logger.info("Parsing with Unstructured", path=file_path)
        return _parse_with_unstructured(file_path)
