import streamlit as st
from collections import defaultdict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import os
import json
import tempfile
from datetime import datetime

# Load model once
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
HISTORY_DIR = "chat_sessions"
VECTOR_DB_PATH = "vector_index"

# Ensure directory exists
os.makedirs(HISTORY_DIR, exist_ok=True)

# ================= Document Upload and Indexing =================
@st.cache_resource
def load_or_create_vectordb(docs=None):
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if docs:
        texts = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
        vectordb = FAISS.from_documents(texts, embedding)
        vectordb.save_local(VECTOR_DB_PATH)
    elif os.path.exists(VECTOR_DB_PATH):
        vectordb = FAISS.load_local(VECTOR_DB_PATH, embedding, allow_dangerous_deserialization=True)
    else:
        vectordb = None
    return vectordb

def upload_documents():
    uploaded_files = st.sidebar.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        docs = []
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file.read())
                loader = PyPDFLoader(tmp_file.name)
                docs.extend(loader.load())
        return docs
    return None

# ================= Similar Sentence Highlighting =================
def highlight_similar_sentences(query, text, threshold=0.6):
    if not text:
        return ""
    sentences = text.split('. ')
    query_emb = sentence_model.encode(query, convert_to_tensor=True)
    sentence_embs = sentence_model.encode(sentences, convert_to_tensor=True)
    scores = util.cos_sim(query_emb, sentence_embs)[0]

    highlighted = ""
    for i, sentence in enumerate(sentences):
        clean_sentence = sentence.strip().replace("\n", " ")
        if scores[i] >= threshold:
            highlighted += f"<span style='background-color: yellow;'>{clean_sentence}</span>. "
        else:
            highlighted += clean_sentence + ". "
    return highlighted

# ================= Useful Context Filter =================
def get_useful_context(query, context, threshold=0.6):
    query_emb = sentence_model.encode(query, convert_to_tensor=True)
    useful_docs = []
    all_docs = []  # To store all docs
    for doc in context:
        content = doc.page_content
        sentences = content.split('. ')
        sentence_embs = sentence_model.encode(sentences, convert_to_tensor=True)
        scores = util.cos_sim(query_emb, sentence_embs)[0]
        if any(score >= threshold for score in scores):
            useful_docs.append((doc, scores.max().item()))
        all_docs.append(doc)  # Collect all documents

    useful_docs = sorted(useful_docs, key=lambda x: x[1], reverse=True)[:3]
    return [doc[0] for doc in useful_docs], all_docs  # Return both useful and all docs

# ================= Retrieval Chain =================
def get_context_retriever_chain(vectordb):
    load_dotenv()
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.2, convert_system_message_to_human=True)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful document analysis assistant. Use ONLY the provided context from documents. Mention document names and page numbers when relevant. Avoid using prior knowledge.\nContext:\n{context}"""),
        ("human", "{input}")
    ])
    chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    return create_retrieval_chain(retriever, chain)

def get_response(question, chat_history, vectordb):
    chain = get_context_retriever_chain(vectordb)
    response = chain.invoke({"input": question, "chat_history": chat_history})
    return response["answer"], response["context"]

# ================= Chat History Persistence =================
def save_chat_history(chat_history, session_id):
    session_path = os.path.join(HISTORY_DIR, f"{session_id}.json")
    with open(session_path, "w") as f:
        json.dump([{"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "message": msg.content} for msg in chat_history], f)

def load_chat_history(session_id):
    session_path = os.path.join(HISTORY_DIR, f"{session_id}.json")
    if os.path.exists(session_path):
        with open(session_path, "r") as f:
            messages = json.load(f)
        return [HumanMessage(content=msg["message"]) if i % 2 == 0 else AIMessage(content=msg["message"]) for i, msg in enumerate(messages)]
    return []

# ================= Chat Function =================
def chat(chat_history, vectordb, user_query, session_id):
    if user_query:
        response, context = get_response(user_query, chat_history, vectordb)
        filtered_context, all_context = get_useful_context(user_query, context)

        chat_history.append(HumanMessage(content=user_query))
        chat_history.append(AIMessage(content=response))

        # Display Chat
        st.markdown(f"""<div style="margin-bottom: 20px;">
            <b>You:</b>
            <div style="background-color: #4f83ff; color: white; padding: 12px; border-radius: 15px; margin-top: 5px;">{user_query}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div style="margin-bottom: 30px;">
            <b>AI:</b>
            <div style="background-color: #dbe4ff; padding: 12px; border-radius: 10px; margin-top: 5px;">{response}</div>
        </div>""", unsafe_allow_html=True)

        # Display Relevant Sources
        st.markdown("""<div style="border-top: 1px solid #ccc; padding-top: 20px;">
            <h5 style='font-weight: bold;'>Relevant Document Sources</h5>""", unsafe_allow_html=True)

        metadata_dict = defaultdict(list)
        for doc in filtered_context:
            doc_name = doc.metadata.get("source", "Unknown Document")
            page = doc.metadata.get("page", "N/A")
            highlights = highlight_similar_sentences(user_query, doc.page_content)
            metadata_dict[(doc_name, page)].append(highlights)

        for (doc_name, page), texts in metadata_dict.items():
            st.markdown(f"""<div style='margin-bottom: 10px;'><b>Document:</b> {doc_name} <br><i>Page {page}</i></div>""", unsafe_allow_html=True)
            for txt in texts:
                st.markdown(f"""<div style='margin-bottom: 15px;'>{txt}</div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Display All Document Sources
        st.markdown("""<div style="border-top: 1px solid #ccc; padding-top: 20px;">
            <h5 style='font-weight: bold;'>All Document Sources</h5>""", unsafe_allow_html=True)

        all_metadata_dict = defaultdict(list)
        for doc in all_context:
            doc_name = doc.metadata.get("source", "Unknown Document")
            page = doc.metadata.get("page", "N/A")
            highlights = highlight_similar_sentences(user_query, doc.page_content)
            all_metadata_dict[(doc_name, page)].append(highlights)

        for (doc_name, page), texts in all_metadata_dict.items():
            st.markdown(f"""<div style='margin-bottom: 10px;'><b>Document:</b> {doc_name} <br><i>Page {page}</i></div>""", unsafe_allow_html=True)
            for txt in texts:
                st.markdown(f"""<div style='margin-bottom: 15px;'>{txt}</div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        save_chat_history(chat_history, session_id)

    return chat_history

# ================= UI Setup =================
if __name__ == "__main__":
    st.title("📚 Document QA Chatbot")

    if "session_id" not in st.session_state:
        st.session_state.session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Load or initialize chat history — make sure this comes early
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_chat_history(st.session_state.session_id)  # Try to load existing history
        if not st.session_state.chat_history:
            st.session_state.chat_history = []  # Initialize if not loaded

    # Sidebar: Upload & Clear Buttons
    docs = upload_documents()
    vectordb = load_or_create_vectordb(docs)

    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        save_chat_history([], st.session_state.session_id)
        st.success("Chat history cleared.")

    # User query input
    user_query = st.chat_input("Ask something about the uploaded documents...")

    # Chat logic
    if user_query and vectordb:
        st.session_state.chat_history = chat(st.session_state.chat_history, vectordb, user_query, st.session_state.session_id)
    elif user_query and not vectordb:
        st.warning("Please upload documents before asking a question.")

    # Sidebar: Chat History
    with st.sidebar:
        st.markdown("<h4 style='margin-bottom: 10px;'>💬 Chat History</h4>", unsafe_allow_html=True)

        if st.session_state.chat_history:
            for i in range(0, len(st.session_state.chat_history), 2):
                if i + 1 < len(st.session_state.chat_history):
                    query = st.session_state.chat_history[i].content
                    response = st.session_state.chat_history[i + 1].content
                    with st.expander(f"Q{i//2 + 1}: {query[:50]}..."):
                        st.markdown(f"""<div style="background-color: #f0f2f6; padding: 10px; border-radius: 8px;">
                            <strong>You:</strong><br>{query}<br><br>
                            <strong>AI:</strong><br>{response}
                        </div>""", unsafe_allow_html=True)
        else:
            st.info("Start a conversation to see your chat history here.")
