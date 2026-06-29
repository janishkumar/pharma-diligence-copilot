from typing import Literal, Optional
from pydantic import BaseModel, Field


class Filing(BaseModel):
    company_name: str
    ticker: str
    cik: str
    form_type: Literal["10-K", "10-K/A"]
    fiscal_year: int
    fiscal_year_end_date: str
    filing_date: str
    accession_number: str
    accession_number_original: Optional[str] = None
    source_url: str
    file_sha256: str
    ingested_at: str
    parsed_sections: dict[str, str]


class Chunk(BaseModel):
    chunk_id: str
    text: str
    company: str
    ticker: str
    cik: str
    fiscal_year: int
    fiscal_year_end_date: str
    form_type: str
    section: str
    accession_number: str
    source_url: str
    chunk_index_in_section: int


class Citation(BaseModel):
    n: int
    company: str
    fiscal_year: int
    section: str
    chunk_id: str
    source_url: str


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    filters: Optional[dict] = None


class TimingBreakdown(BaseModel):
    retrieval_ms: int
    rerank_ms: int
    generation_ms: int
    total_ms: int


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    abstained: bool
    timing: TimingBreakdown
    model_version: str
    corpus_snapshot_hash: str
    truncated: bool = False
    error: Optional[str] = None


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    request_id: str
