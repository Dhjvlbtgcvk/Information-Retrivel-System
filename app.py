import streamlit as st
from src.helper import get_pdf_text, get_text_chunk, get_vector_store, get_conversational_chain

def main():
    st.set_page_config("Information Retrieval System", layout="centered")
    st.header("📘 Information Retrieval System")

    with st.sidebar:
        st.title("📁 Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF file(s) and click on Submit", 
            accept_multiple_files=True
        )
        if st.button("Submit & Process") and pdf_docs:
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunk(raw_text)
                vector_store = get_vector_store(text_chunks)
                st.session_state.conversation = get_conversational_chain(vector_store)
            st.success("Processing Complete ✅")

    # Ask user query after processing
    if "conversation" in st.session_state:
        user_query = st.text_input("💬 Ask a question from your documents:")
        if user_query:
            with st.spinner("Thinking..."):
                response = st.session_state.conversation.run(user_query)
                st.markdown("**Answer:**")
                st.write(response)

if __name__ == "__main__":
    main()
