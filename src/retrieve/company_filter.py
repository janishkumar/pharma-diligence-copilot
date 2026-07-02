"""Deterministic company auto-detection for query scoping.

For a fixed corpus of 12 known issuers, a name/alias lookup is faster, cheaper,
and more predictable than an LLM call. If a question names exactly one known
company, we scope retrieval to it; if it names none or several, we do not filter
(so genuine cross-company questions still work). An explicit filter always wins.
"""
import re

# ticker -> list of lowercase aliases (longest/most specific first not required;
# matching is whole-token so order does not matter).
COMPANY_ALIASES: dict[str, list[str]] = {
    "PFE": ["pfizer", "pfe"],
    "MRK": ["merck", "mrk"],
    "JNJ": ["johnson & johnson", "johnson and johnson", "j&j", "jnj"],
    "ABBV": ["abbvie", "abbv"],
    "BMY": ["bristol myers squibb", "bristol-myers squibb", "bristol myers", "bristol", "bms", "bmy"],
    "LLY": ["eli lilly", "lilly", "lly"],
    "AMGN": ["amgen", "amgn"],
    "GILD": ["gilead", "gild"],
    "REGN": ["regeneron", "regn"],
    "VRTX": ["vertex", "vrtx"],
    "MRNA": ["moderna", "mrna"],
    "BIIB": ["biogen", "biib"],
}


def _alias_present(alias: str, text: str) -> bool:
    """True if alias appears as a standalone token (not inside another word),
    tolerant of internal punctuation like '&' and '-'."""
    pattern = r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])"
    return re.search(pattern, text) is not None


def detect_tickers(question: str) -> list[str]:
    """All distinct known companies named in the question, in ticker order."""
    text = question.lower()
    found = []
    for ticker, aliases in COMPANY_ALIASES.items():
        if any(_alias_present(a, text) for a in aliases):
            found.append(ticker)
    return found


def detect_ticker(question: str) -> str | None:
    """The single company a question is about, or None if zero or several are
    named (ambiguous -> let retrieval span the whole corpus)."""
    found = detect_tickers(question)
    return found[0] if len(found) == 1 else None


def apply_auto_company_filter(question: str, filters: dict | None) -> tuple[dict | None, str | None]:
    """Return (filters, auto_ticker). If the caller did not already pin a ticker
    and the question clearly names one company, add a ticker filter. An explicit
    ticker filter is never overridden. Returns the auto-applied ticker (or None)
    for observability/UI display."""
    if filters and "ticker" in filters:
        return filters, None  # explicit filter wins
    ticker = detect_ticker(question)
    if not ticker:
        return filters, None
    merged = dict(filters) if filters else {}
    merged["ticker"] = {"$eq": ticker}
    return merged, ticker
