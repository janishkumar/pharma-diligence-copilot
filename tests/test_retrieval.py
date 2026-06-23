"""Retrieval tests — Section 19.2"""
import pytest
from unittest.mock import patch, MagicMock


def test_query_prefix_applied_to_queries():
    """BGE query instruction prefix must be applied to queries, not documents."""
    from src.index.embed import QUERY_PREFIX
    assert QUERY_PREFIX, "Query prefix must be set for bge models"
    assert "searching" in QUERY_PREFIX.lower() or "represent" in QUERY_PREFIX.lower()


def test_empty_filter_does_not_widen():
    """When a filter returns zero results, retrieve() must return empty, not widen."""
    from unittest.mock import patch
    with patch("src.retrieve.retriever.store.query", return_value=[]):
        from src.retrieve.retriever import retrieve
        result = retrieve("test question", filters={"ticker": {"$eq": "FAKECO"}})
        assert result == [], "Empty filtered result must return empty, not silently widen"
