"""Generation / answer-quality tests — PRD Section 19.3 (API-free).

Covers citation parsing (all grouping styles + out-of-range), abstention
detection, and the unsupported-answer groundedness guard. The generator and
retriever are mocked so these run without any paid API call.
"""
from unittest.mock import patch

from src.generate.prompt import ABSTENTION_STRING, build_user_message


def _chunks(n):
    return [
        {"text": f"chunk {i} text", "chunk_id": f"id{i}",
         "metadata": {"company": "Pfizer", "fiscal_year": 2025, "section": "Item 1A", "source_url": "https://sec.gov/x"}}
        for i in range(1, n + 1)
    ]


# ---- prompt -----------------------------------------------------------------

def test_abstention_string_exact():
    assert ABSTENTION_STRING == "The provided documents do not contain enough information to answer this."


def test_prompt_contains_question_and_numbering():
    msg = build_user_message("What are the risk factors?", _chunks(2))
    assert "What are the risk factors?" in msg
    assert "[1]" in msg and "[2]" in msg


# ---- citation marker extraction (all grouping styles) -----------------------

def test_extract_markers_adjacent():
    from src.pipeline import _extract_marker_numbers
    assert _extract_marker_numbers("a [1][2] b [3]") == [1, 2, 3]


def test_extract_markers_comma_and_spaces():
    from src.pipeline import _extract_marker_numbers
    assert _extract_marker_numbers("a [1, 2] b [3,4]") == [1, 2, 3, 4]


def test_extract_markers_ranges():
    from src.pipeline import _extract_marker_numbers
    assert _extract_marker_numbers("see [1-3] and [5]") == [1, 2, 3, 5]


def test_extract_markers_double_digit_no_substring_collision():
    from src.pipeline import _extract_marker_numbers
    # [1] must not be implied by [12]; both distinct numbers captured
    assert _extract_marker_numbers("x [12] y [1]") == [12, 1]


# ---- _parse_citations (in-range vs out-of-range) ----------------------------

def test_parse_citations_grouped_all_inrange():
    from src.pipeline import _parse_citations
    cites, oor = _parse_citations("claim [1, 2] and [3]", _chunks(5))
    assert [c.n for c in cites] == [1, 2, 3]
    assert oor == []


def test_parse_citations_out_of_range_reported_not_silent():
    from src.pipeline import _parse_citations
    cites, oor = _parse_citations("grounded [2] but hallucinated [10]", _chunks(5))
    assert [c.n for c in cites] == [2]
    assert oor == [10]  # surfaced, not silently dropped


def test_parse_citations_maps_to_correct_chunk():
    from src.pipeline import _parse_citations
    cites, _ = _parse_citations("see [3]", _chunks(5))
    assert cites[0].chunk_id == "id3"


# ---- abstention detection ---------------------------------------------------

def test_is_abstention_exact():
    from src.pipeline import _is_abstention
    assert _is_abstention(ABSTENTION_STRING)


def test_is_abstention_with_elaboration():
    from src.pipeline import _is_abstention
    assert _is_abstention(ABSTENTION_STRING + " The context makes no mention of crypto [2].")


def test_is_abstention_wrapped_markdown_quote():
    from src.pipeline import _is_abstention
    assert _is_abstention('> "' + ABSTENTION_STRING + '"')


def test_is_not_abstention_for_real_answer():
    from src.pipeline import _is_abstention
    assert not _is_abstention("Pfizer's principal risks include pricing pressure [1].")


# ---- groundedness guard (integration, mocked generator) ---------------------

def _mock_gen(text):
    from src.generate.generator import GenerationResult
    return GenerationResult(text=text)


def test_unsupported_answer_is_suppressed_to_abstention():
    """A non-abstaining answer with zero valid citations must be refused."""
    from src import pipeline
    with patch.object(pipeline.retriever, "retrieve", return_value=_chunks(3)), \
         patch.object(pipeline.rerank, "rerank", return_value=_chunks(3)), \
         patch.object(pipeline.generator, "generate", return_value=_mock_gen("Pfizer earned a lot of money last year.")):
        resp = pipeline.ask("How much did Pfizer earn?")
    assert resp.abstained is True
    assert resp.citations == []


def test_grounded_answer_passes_with_citations():
    from src import pipeline
    with patch.object(pipeline.retriever, "retrieve", return_value=_chunks(3)), \
         patch.object(pipeline.rerank, "rerank", return_value=_chunks(3)), \
         patch.object(pipeline.generator, "generate", return_value=_mock_gen("Risks include pricing [1] and COVID demand [2].")):
        resp = pipeline.ask("What are the risks?")
    assert resp.abstained is False
    assert [c.n for c in resp.citations] == [1, 2]


def test_empty_retrieval_abstains():
    from src import pipeline
    with patch.object(pipeline.retriever, "retrieve", return_value=[]), \
         patch.object(pipeline.rerank, "rerank", return_value=[]):
        resp = pipeline.ask("Anything?", filters={"ticker": {"$eq": "NOPE"}})
    assert resp.abstained is True
    assert resp.citations == []


def test_anthropic_call_omits_top_p():
    """Regression: Anthropic's API returns 400 if BOTH temperature and top_p are
    sent. The _anthropic backend must send temperature only."""
    from unittest.mock import MagicMock
    import src.generate.generator as g

    fake_client = MagicMock()
    fake_resp = MagicMock()
    text_block = MagicMock(); text_block.type = "text"; text_block.text = "answer [1]"
    fake_resp.content = [text_block]
    fake_resp.usage.input_tokens = 10
    fake_resp.usage.output_tokens = 5
    fake_resp.stop_reason = "end_turn"
    fake_client.messages.create.return_value = fake_resp

    with patch("anthropic.Anthropic", return_value=fake_client):
        result = g._anthropic("msg")

    kwargs = fake_client.messages.create.call_args.kwargs
    assert "temperature" in kwargs
    assert "top_p" not in kwargs, "Anthropic must not send top_p with temperature (400)"
    assert result.text == "answer [1]"


def test_generation_error_is_surfaced_not_masked_as_abstention(monkeypatch):
    """A generation failure must yield an error response, not a silent abstention."""
    from src import pipeline
    from src.generate.generator import GenerationResult
    err = GenerationResult(text="The provided documents do not contain enough information to answer this.",
                           error="BadRequestError: boom")
    with patch.object(pipeline.retriever, "retrieve", return_value=_chunks(3)), \
         patch.object(pipeline.rerank, "rerank", return_value=_chunks(3)), \
         patch.object(pipeline.generator, "generate", return_value=err):
        resp = pipeline.ask("anything")
    assert resp.error is not None
    assert resp.abstained is False  # not a clean content abstention
