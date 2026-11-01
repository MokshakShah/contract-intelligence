"""Streamlit interface for the Contract Intelligence Assistant."""

from pathlib import Path

import streamlit as st
import yaml

from src.retrieval import search


CONFIG_DIR = Path("configs")


@st.cache_data
def config_paths() -> list[str]:
    return sorted(str(path) for path in CONFIG_DIR.glob("*.yaml"))


def context_html(result: dict, radius: int = 300) -> str:
    document = Path("data/raw/corpus") / result["doc_path"]
    with open(document, encoding="utf-8") as handle:
        text = handle.read()
    start = max(0, result["start"] - radius)
    end = min(len(text), result["end"] + radius)
    before = text[start:result["start"]]
    answer = text[result["start"]:result["end"]]
    after = text[result["end"]:end]
    return (
        f"...{before.replace('&', '&amp;').replace('<', '&lt;')}"
        f"<mark>{answer.replace('&', '&amp;').replace('<', '&lt;')}</mark>"
        f"{after.replace('&', '&amp;').replace('<', '&lt;')}..."
    )


st.set_page_config(page_title="Contract Intelligence Assistant")
st.title("Contract Intelligence Assistant")
paths = config_paths()
if not paths:
    st.error("No YAML configs found in configs/.")
    st.stop()
selected = st.sidebar.selectbox("Active config", paths, format_func=lambda p: Path(p).name)
st.sidebar.caption(f"Active config: {Path(selected).name}")
config = yaml.safe_load(Path(selected).read_text(encoding="utf-8")) or {}
query = st.text_input("Ask a question about the indexed contracts")
k = st.slider("Number of passages", min_value=1, max_value=32, value=5)

if query:
    response = search(query, config, k)
    if response["abstained"]:
        top_score = response["results"][0]["normalized_score"] if response["results"] else 0.0
        st.warning(
            f"No relevant clause found (top score {top_score:.4f}, "
            f"threshold {response['threshold']})."
        )
    else:
        for index, result in enumerate(response["results"], 1):
            with st.expander(
                f"{index}. {Path(result['doc_path']).name} "
                f"[{result['start']}:{result['end']}] "
                f"score {result['score']:.4f}"
            ):
                st.markdown(context_html(result), unsafe_allow_html=True)
