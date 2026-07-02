import sys
from pathlib import Path

# `streamlit run src/ui.py` puts src/ on sys.path, not the project root, so the
# `src` package is not importable. Add the project root before importing it.
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st  # noqa: E402
from src.pipeline import ask  # noqa: E402
from src.retrieve.company_filter import detect_ticker  # noqa: E402

st.set_page_config(page_title="Pharma Diligence Copilot", layout="wide")
st.title("Pharma Diligence Copilot")
st.caption("Ask diligence questions over 12 pharma companies' latest SEC 10-K filings. Answers are grounded in the filings with citations, or the copilot abstains.")

EXAMPLES = [
    "What are Eli Lilly's principal risk factors?",
    "What does Moderna say about its mRNA platform?",
    "How does AbbVie describe Humira biosimilar competition?",
    "What drove the change in Merck's revenue this year?",
]

if "history" not in st.session_state:
    st.session_state.history = []
if "pending" not in st.session_state:
    st.session_state.pending = None

with st.sidebar:
    st.header("Filters (optional)")
    st.caption("The company is auto-detected from your question. Set a ticker here only to override it.")
    company_filter = st.text_input("Company ticker (e.g. PFE)", "")
    year_filter = st.text_input("Fiscal year (e.g. 2025)", "")
    if st.session_state.history:
        if st.button("🗑️ Clear history"):
            st.session_state.history = []
            st.rerun()


def run_query(question: str):
    filters = {}
    if company_filter.strip():
        filters["ticker"] = {"$eq": company_filter.strip().upper()}
    if year_filter.strip():
        try:
            filters["fiscal_year"] = {"$eq": int(year_filter.strip())}
        except ValueError:
            st.warning("Fiscal year must be a number.")
    with st.spinner("Retrieving and generating answer… (first query loads models, ~30s)"):
        result = ask(question, filters=filters or None)
    st.session_state.history.append({"question": question, "result": result})


# --- input ---
question = st.text_input("Ask a diligence question:", placeholder=EXAMPLES[0])
ask_clicked = st.button("Ask", type="primary")

# Starter examples (shown until the user has asked something)
if not st.session_state.history:
    st.markdown("**Try one of these:**")
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 2].button(ex, key=f"ex_{i}"):
            st.session_state.pending = ex

# Resolve what to run this rerun
to_run = None
if st.session_state.pending:
    to_run = st.session_state.pending
    st.session_state.pending = None
elif ask_clicked and question.strip():
    to_run = question.strip()

if to_run:
    run_query(to_run)

# --- history (newest first; latest expanded, older collapsed) ---
for idx, item in enumerate(reversed(st.session_state.history)):
    q = item["question"]
    r = item["result"]
    is_latest = idx == 0

    st.divider()
    st.markdown(f"**Q:** {q}")

    scope = detect_ticker(q)
    if scope and not company_filter.strip():
        st.caption(f"🔎 Auto-scoped to **{scope}**")

    if r.error:
        st.error(r.answer)
    elif r.abstained:
        st.warning(r.answer)
    else:
        st.markdown(r.answer)
        if r.truncated:
            st.caption("⚠️ Answer was truncated at the token limit; it may be incomplete.")

    with st.expander("Sources & timing", expanded=is_latest):
        if r.citations:
            for c in r.citations:
                st.markdown(f"- [{c.n}] **{c.company}** {c.fiscal_year} 10-K, {c.section} · `{c.chunk_id}`")
        else:
            st.caption("No citations (abstained or no grounded sources).")
        t = r.timing
        st.caption(
            f"Retrieval {t.retrieval_ms}ms · Rerank {t.rerank_ms}ms · "
            f"Generation {t.generation_ms}ms · **Total {t.total_ms}ms** · {r.model_version}"
        )
