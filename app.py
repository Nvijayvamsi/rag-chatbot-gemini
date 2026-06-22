import os
import tempfile
import streamlit as st
import google.generativeai as genai

from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader,
    WebBaseLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import (
    HuggingFaceEmbeddings
)

# =========================================
# API KEY
# =========================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("********")

genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

# =========================================
# STREAMLIT
# =========================================

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖"
)

st.title("🤖 RAG Chatbot")

# =========================================
# SESSION STATE
# =========================================

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================================
# SIDEBAR
# =========================================

with st.sidebar:

    st.header("Upload Data")

    pdfs = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    website_url = st.text_input(
        "Website URL"
    )

    process = st.button(
        "Process"
    )

# =========================================
# LOAD PDF
# =========================================

def load_pdfs(files):

    docs = []

    for file in files:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:

            tmp.write(file.read())

            path = tmp.name

        loader = PyPDFLoader(path)

        docs.extend(loader.load())

    return docs

# =========================================
# LOAD WEBSITE
# =========================================

def load_website(url):

    loader = WebBaseLoader(
        web_paths=(url,)
    )

    return loader.load()

# =========================================
# VECTOR DB
# =========================================

def create_vector_db(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(
        documents
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(
        chunks,
        embeddings
    )

    return db

# =========================================
# PROCESS DATA
# =========================================

if process:

    all_docs = []

    try:

        if pdfs:

            all_docs.extend(
                load_pdfs(pdfs)
            )

        if website_url:

            all_docs.extend(
                load_website(
                    website_url
                )
            )

        if len(all_docs) == 0:

            st.warning(
                "Upload PDF or URL"
            )

        else:

            with st.spinner(
                "Creating Vector DB..."
            ):

                st.session_state.vector_db = (
                    create_vector_db(
                        all_docs
                    )
                )

            st.success(
                "Knowledge Base Ready"
            )

    except Exception as e:

        st.error(str(e))

# =========================================
# RETRIEVE
# =========================================

def retrieve_context(question):

    docs = (
        st.session_state.vector_db
        .similarity_search(
            question,
            k=4
        )
    )

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    return context

# =========================================
# GEMINI ANSWER
# =========================================

def ask_gemini(question):

    context = retrieve_context(
        question
    )

    prompt = f"""
Answer ONLY from context.

Context:
{context}

Question:
{question}
"""

    response = model.generate_content(
        prompt
    )

    return response.text

# =========================================
# CHAT HISTORY
# =========================================

for msg in st.session_state.messages:

    with st.chat_message(
        msg["role"]
    ):
        st.markdown(
            msg["content"]
        )

# =========================================
# CHAT INPUT
# =========================================

question = st.chat_input(
    "Ask a question..."
)

if question:

    if st.session_state.vector_db is None:

        st.warning(
            "Process data first"
        )

    else:

        st.session_state.messages.append(
            {
                "role":"user",
                "content":question
            }
        )

        with st.spinner(
            "Thinking..."
        ):

            answer = ask_gemini(
                question
            )

        st.session_state.messages.append(
            {
                "role":"assistant",
                "content":answer
            }
        )

        st.rerun()