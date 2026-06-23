"""Generation / answer quality tests — Section 19.3"""
from src.generate.prompt import ABSTENTION_STRING, build_user_message


def test_abstention_string_exact():
    assert ABSTENTION_STRING == "The provided documents do not contain enough information to answer this."


def test_prompt_contains_question():
    msg = build_user_message("What are the risk factors?", [])
    assert "What are the risk factors?" in msg


def test_prompt_context_numbering():
    chunks = [
        {"text": "chunk one text", "chunk_id": "abc", "metadata": {"company": "Pfizer", "fiscal_year": 2024, "section": "Item 1A"}},
        {"text": "chunk two text", "chunk_id": "def", "metadata": {"company": "Merck", "fiscal_year": 2024, "section": "Item 7"}},
    ]
    msg = build_user_message("test?", chunks)
    assert "[1]" in msg
    assert "[2]" in msg
