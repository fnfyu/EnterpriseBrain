"""BGE-M3 embedding model wrapper (1024-dim, bilingual CN/EN, 8k ctx)."""
from functools import lru_cache
from typing import List
import numpy as np
from app.core.logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _load_model():
    from FlagEmbedding import BGEM3FlagModel
    logger.info("Loading BGE-M3 model...")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    logger.info("BGE-M3 loaded")
    return model


def embed_texts(texts: List[str]) -> List[List[float]]:
    model = _load_model()
    output = model.encode(
        texts,
        batch_size=12,
        max_length=8192,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    return output["dense_vecs"].tolist()


def embed_query(query: str) -> List[float]:
    return embed_texts([query])[0]
