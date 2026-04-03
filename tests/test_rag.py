import pytest
from app.rag.reranker import reciprocal_rank_fusion


def test_rrf_merges_and_deduplicates():
    list1 = [
        {"id": "a", "content": "doc a", "score": 0.9},
        {"id": "b", "content": "doc b", "score": 0.8},
    ]
    list2 = [
        {"id": "b", "content": "doc b", "score": 0.7},
        {"id": "c", "content": "doc c", "score": 0.6},
    ]
    result = reciprocal_rank_fusion([list1, list2])
    ids = [r["id"] for r in result]
    # "b" appears in both lists, should rank highly
    assert "b" in ids
    assert len(ids) == 3  # deduplicated
    assert result[0]["id"] == "b"  # highest RRF score
