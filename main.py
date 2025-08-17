# main.py
import streamlit as st
from rag import process_urls, generate_answer

st.set_page_config(page_title="Real Estate Research Tool", page_icon="🏠")
st.title("🏠 Real Estate Research Tool")

with st.sidebar:
    st.header("Data Sources")
    url1 = st.text_input("URL 1")
    url2 = st.text_input("URL 2")
    url3 = st.text_input("URL 3")
    if st.button("Process URLs"):
        urls = [u for u in (url1, url2, url3) if u]
        with st.spinner("Fetching & indexing…"):
            try:
                n_chunks = process_urls(urls)
                st.success(f"Indexed {n_chunks} chunks.")
                st.session_state["ready"] = True
            except Exception as e:
                st.error(str(e))
                st.session_state["ready"] = False

# Chat UI
if "ready" not in st.session_state:
    st.session_state["ready"] = False

st.caption("Ask a question based on the processed URLs.")
prompt = st.chat_input("Type your question…")

if prompt:
    if not st.session_state["ready"]:
        st.warning("Please process URLs first.")
    else:
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    answer, sources = generate_answer(prompt)
                    st.write(answer or "_No answer produced._")
                    if sources:
                        st.markdown("**Sources:**")
                        for s in sources:
                            st.markdown(f"- [{s}]({s})")
                except Exception as e:
                    st.error(str(e))
