SYSTEM_PROMPT = """\
You are a research assistant answering questions strictly from the provided CONTEXT.

Rules:
1. Use only the provided CONTEXT to answer. Do not use outside knowledge, even if you know the answer.
2. Every factual claim must be supported by an inline citation. Cite each source as its own bracketed number: write [1][2], NEVER [1, 2] or [1-3]. Use only numbers that appear in the CONTEXT block.
3. If the answer is not contained in the CONTEXT, your ENTIRE reply must be exactly: "The provided documents do not contain enough information to answer this." Do not add any other text, and do not guess.
4. Do not invent quantities, names, dates, or company-specific details.
5. Keep the answer concise. Use short paragraphs or bullets. End with a "Sources" list.\
"""

ABSTENTION_STRING = "The provided documents do not contain enough information to answer this."


def build_user_message(question: str, chunks: list[dict]) -> str:
    context_lines = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        company = meta.get("company", "Unknown")
        year = meta.get("fiscal_year", "")
        section = meta.get("section", "")
        chunk_id = chunk.get("chunk_id", "")
        context_lines.append(
            f"[{i}] {chunk['text']}\n"
            f"    (Source: {company} {year} 10-K, {section}, chunk {chunk_id})"
        )

    context_block = "\n\n".join(context_lines)
    return f"QUESTION:\n{question}\n\nCONTEXT:\n{context_block}\n\nANSWER:"
