import streamlit as st
import os
from utils.save_docs import save_docs_to_vectordb
from utils.session_state import initialize_session_state_variables
from utils.prepare_vectordb import get_vectorstore
from utils.chatbot import chat

class ChatApp:
    """
    A Streamlit application for chatting with PDF documents.
    """

    def __init__(self):
        # Ensure the docs folder exists
        if not os.path.exists("docs"):
            os.makedirs("docs")

        # Title and session state initialization
        st.markdown("<h1 style='text-align: center; color: #3366cc;'>📄 Document Query System</h1>", unsafe_allow_html=True)
        initialize_session_state_variables(st)
        self.docs_files = st.session_state.processed_documents

    def run(self):
        upload_docs = os.listdir("docs")

        # Sidebar for document management
        with st.sidebar:
            st.markdown("### 📚 Your Documents")
            if upload_docs:
                st.success("Uploaded:")
                for doc in upload_docs:
                    st.markdown(f"- {doc}")
            else:
                st.info("No documents uploaded yet.")

            st.markdown("---")
            st.markdown("### 📤 Upload PDF Documents")
            pdf_docs = st.file_uploader(
                "Select one or more PDFs and click on 'Process'",
                type=['pdf'],
                accept_multiple_files=True
            )
            if pdf_docs:
                save_docs_to_vectordb(pdf_docs, upload_docs)

        # Main chat area
        if self.docs_files or st.session_state.uploaded_pdfs:
            if len(upload_docs) > st.session_state.previous_upload_docs_length:
                st.session_state.vectordb = get_vectorstore(upload_docs, from_session_state=True)
                st.session_state.previous_upload_docs_length = len(upload_docs)

            user_query = st.chat_input("Ask a question:")
            st.session_state.chat_history = chat(
                st.session_state.chat_history,
                st.session_state.vectordb,
                user_query,
                st.session_state.session_id  # Pass session_id here
            )

        else:
            st.warning("📁 Upload a PDF file to start chatting with it. You can continue uploading multiple files, and they'll stay saved for future sessions.")

if __name__ == "__main__":
    app = ChatApp()
    app.run()
