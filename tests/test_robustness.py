"""Robustness tests — Section 19.6"""
from src.generate.prompt import ABSTENTION_STRING


def test_abstention_string_is_canonical():
    """Ensure the abstention string matches the PRD-specified exact value."""
    assert ABSTENTION_STRING == "The provided documents do not contain enough information to answer this."


def test_malformed_empty_question_rejected():
    from pydantic import ValidationError
    from src.schemas import AskRequest
    with pytest.raises((ValidationError, Exception)):
        AskRequest(question="")


import pytest


def test_malformed_short_question_rejected():
    from pydantic import ValidationError
    from src.schemas import AskRequest
    with pytest.raises((ValidationError, Exception)):
        AskRequest(question="ab")  # min_length=3 means 2 chars fails
