from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from modules.rag_agent import MeddraRAGAgent, RAGResponse


@st.cache_resource
def get_agent(config_text: str) -> MeddraRAGAgent:
    config_path = Path(__file__).resolve().parent / "config.yaml"
    return MeddraRAGAgent(config_path)


def render_hierarchy(response: RAGResponse) -> None:
    if not response.hierarchy:
        return
    hierarchy_records = []
    for level in ["LLT", "PT", "HLT", "HLGT", "SOC"]:
        data = response.hierarchy.get(level)
        if data:
            hierarchy_records.append(
                {
                    "Level": level,
                    "Code": data.get("code", ""),
                    "Term": data.get("term", ""),
                }
            )
    if hierarchy_records:
        st.subheader("Hierarchy")
        st.table(pd.DataFrame(hierarchy_records))
    else:
        st.info("No hierarchy information available for the selected code.")


def render_candidates(response: RAGResponse) -> None:
    st.sidebar.header("Vector Search Results")
    if not response.candidates:
        st.sidebar.info("No candidates retrieved.")
        return

    for idx, candidate in enumerate(response.candidates, start=1):
        lexical_pct = candidate.lexical_score * 100 if candidate.lexical_score else 0
        st.sidebar.markdown(
            f"**{idx}. {candidate.term}**\n\n"
            f"- Level: {candidate.level}\n"
            f"- Code: `{candidate.code}`\n"
            f"- Vector score: {candidate.score:.3f}\n"
            f"- Lexical match: {lexical_pct:.1f}%\n"
            f"- Combined score: {candidate.combined_score:.3f}"
        )
        doc_text = candidate.document_text or candidate.metadata.get("document_text", "")
        if doc_text:
            st.sidebar.code(doc_text, language="text")


def main() -> None:
    st.set_page_config(page_title="MedDRA-Coding-AI", layout="wide")
    st.title("MedDRA-Coding-AI")
    st.caption("Conversational coding assistant with Retrieval-Augmented Generation")

    config_path = Path(__file__).resolve().parent / "config.yaml"
    config_text = config_path.read_text(encoding="utf-8")

    agent = get_agent(config_text)
    versions = agent.available_versions()
    if not versions:
        st.error(
            "No MedDRA indexes were found. Run `python meddra_rag_assistant/build_index.py` "
            "to build vector indexes before using the assistant."
        )
        return

    with st.sidebar:
        st.header("Settings")
        selected_version = st.selectbox("MedDRA Version", versions, index=0)
        default_top_k = int(agent.retrieval_config.get("top_k", 5))
        top_k = st.slider("Top-k Candidates", min_value=3, max_value=15, value=default_top_k)

    with st.form(key="term_form"):
        term = st.text_input("Enter a medical term", "")
        submitted = st.form_submit_button("Encode Term")

    if not submitted or not term.strip():
        st.info("Provide a medical term and click **Encode Term** to begin.")
        return

    with st.spinner("Running retrieval and reasoning..."):
        response = agent.run(term=term, version_key=selected_version, top_k=int(top_k))

    render_candidates(response)

    if response.low_confidence:
        st.warning(
            "Low-confidence result. Consider refining the input term or reviewing the suggested alternatives."
        )

    if response.best_match:
        st.subheader("Selected MedDRA Code")
        st.markdown(
            f"- **Term:** {response.best_match.get('term', '')}\n"
            f"- **Code:** `{response.best_match.get('code', '')}`\n"
            f"- **Level:** {response.best_match.get('level', '')}"
        )
        doc_text = response.best_match.get("document_text")
        if doc_text:
            st.code(doc_text, language="text")

    if agent.include_hierarchy:
        render_hierarchy(response)

    st.subheader("LLM Reasoning")
    st.write(response.reasoning)

    with st.expander("Raw LLM Output"):
        st.code(response.llm_output or "", language="json")


if __name__ == "__main__":
    main()
