from xml.dom.minidom import Document

import streamlit as st
import os
import tempfile
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain.document_loaders import PyPDFLoader,TextLoader
from langchain_community.document_loaders import Docx2txtLoader
import pandas as pd
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# -------------------- Streamlit App Config --------------------
st.set_page_config(page_title="Chat with Documents - Gemini AI", layout="centered")

st.title("📄 Chat with Your Documents")
st.markdown("""
Welcome to **Chat with Documents AI** 🤖  
Upload your documents (PDF, DOCX, XLSX, or TXT), then ask any question.  
The app uses **free open-source AI** to analyze and respond intelligently.
""")

# -------------------- Gemini API Key Setup --------------------
load_dotenv()
gApi_key=os.getenv('GROQ-API-KEY')

# -------------------- File Upload Section --------------------
uploaded_files = st.file_uploader(
    "📂 Upload your documents",
    type=["pdf", "docx", "xlsx", "txt"],
    accept_multiple_files=True
)

# Temporary file storage
uploaded_file_paths = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        # Save uploaded file to a temporary location
        temp_file_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        uploaded_file_paths.append(temp_file_path)
    st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully.")

# -------------------- Helper Function: Load Documents --------------------
def load_documents(filepaths):
    docs = []
    for file in filepaths:
        ext = file.split(".")[-1].lower()
        match ext:
            case "pdf":
                loader = PyPDFLoader(file)
                docs.extend(loader.load())
            case "docx":
                loader = Docx2txtLoader(file)
                docs.extend(loader.load())
            case "xls"| "xlsx":
                df = pd.read_excel(file)
                text_data = df.to_string()
                docs.append(Document(page_content = text_data,metadata = {"source":file}))
            case "txt":
                loader = TextLoader(file)
                docs.extend(loader.load())
            case _:
                continue

    return docs

# -------------------- Question Input --------------------
user_question = st.text_area("💬 Ask a question about your uploaded documents:")

# 1. Load all documents
documents = load_documents(uploaded_file_paths)

# 2. Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# -------------------- Process Button --------------------
if st.button("🚀 Answer"):
    if not uploaded_files:
        st.warning("Please upload at least one document first.")
        st.stop()

    if not user_question.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("🔍 Processing documents..."):
        # 3. Create embeddings and vectorstore
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        #embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        #vectorstore = FAISS.from_documents(chunks, embeddings)

        # 4. Create chat model
        llm = ChatGroq(model="llama-3.3-70b-versatile",api_key=gApi_key)

        # 5. Build conversational retrieval chain
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        #memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory
        )

        # 6. Ask the question
        response = qa_chain.run(user_question)

        # 7. Display response
        st.subheader("🧠 AI Response:")
        st.write(response)