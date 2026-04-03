"""Reciprocal Rank Fusion + Cross-Encoder reranking."""
from typing import List, Dict, Any
from app.core.logging import get_logger

logger = get_logger(__name__)


def reciprocal_rank_fusion(
    result_lists: List[List[Dict[str, Any]]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """
    Merge multiple ranked result lists into one via RRF.
    k=60 is the standard constant that dampens high ranks.
    """
    scores: Dict[str, float] = {}
    chunks: Dict[str, Dict] = {}

    for ranked_list in result_lists:
        for rank, item in enumerate(ranked_list):
            doc_id = item["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            chunks[doc_id] = item  # keep latest copy

    merged = sorted(chunks.values(), key=lambda x: scores[x["id"]], reverse=True)
    for item in merged:
        item["rrf_score"] = scores[item["id"]]
    return merged


def rerank(query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """BGE-Reranker cross-encoder reranking."""
    if not candidates:
        return []
    try:
        from FlagEmbedding import FlagReranker
        reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
        pairs = [[query, c["content"]] for c in candidates]
        scores = reranker.compute_score(pairs, normalize=True)
        for i, candidate in enumerate(candidates):
            candidate["rerank_score"] = float(scores[i])
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]
    except Exception as e:
        logger.warning("Reranker unavailable, falling back to RRF scores", error=str(e))
        return candidates[:top_k]
