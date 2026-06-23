import streamlit as st
from src.pipeline import ask
from src.config import cfg

st.set_page_config(page_title="Pharma Diligence Copilot", layout="wide")
st.title("Pharma Diligence Copilot")

with st.sidebar:
    st.header("Filters (optional)")
    company_filter = st.text_input("Company ticker (e.g. PFE)", "")
    year_filter = st.text_input("Fiscal year (e.g. 2024)", "")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("Ask a diligence question:", placeholder="What are Pfizer's principal risk factors?")

if st.button("Ask") and question.strip():
    filters = {}
    if company_filter.strip():
        filters["ticker"] = {"$eq": company_filter.strip().upper()}
    if year_filter.strip():
        try:
            filters["fiscal_year"] = {"$eq": int(year_filter.strip())}
        except ValueError:
            st.warning("Fiscal year must be a number.")

    with st.spinner("Retrieving and generating answer..."):
        result = ask(question, filters=filters or None)

    st.session_state.history.append({"question": question, "result": result})

for item in reversed(st.session_state.history):
    q = item["question"]
    r = item["result"]
    st.markdown(f"**Q:** {q}")
    if r.abstained:
        st.warning(r.answer)
    else:
        st.markdown(r.answer)

    with st.expander("Sources & timing"):
        for c in r.citations:
            st.markdown(f"- [{c.n}] {c.company} {c.fiscal_year} 10-K, {c.section}, chunk `{c.chunk_id}`")
        t = r.timing
        st.caption(
            f"Retrieval {t.retrieval_ms}ms | Rerank {t.rerank_ms}ms | "
            f"Generation {t.generation_ms}ms | Total {t.total_ms}ms | "
            f"Model: {r.model_version}"
        )
    st.divider()
