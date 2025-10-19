from xml.dom.minidom import Document

import streamlit as st
import os
import tempfile

from gitdb.fun import delta_duplicate
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain.document_loaders import PyPDFLoader,TextLoader
from langchain_community.document_loaders import Docx2txtLoader
import pandas as pd
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# -------------------- Streamlit App Config --------------------
st.set_page_config(page_title="Chat with Documents", layout="wide")

st.title("💬 Chat with Your Documents📄")
st.markdown("""
Welcome to **Chat with Documents AI** 🤖  
Upload your documents (PDF, DOCX, XLSX, or TXT), then ask any question.  
The app uses **free open-source AI** to analyze and respond intelligently.
""")

# -------------------- GROQ API Key Setup --------------------
def is_streamlit_cloud():
    # Streamlit Cloud sets this environment variable
    return os.environ.get("STREAMLIT_RUNTIME") == "cloud"

if is_streamlit_cloud():
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    load_dotenv()
    gApi_key=os.getenv('GROQ_API_KEY')

# -------------------- Initialize Session State --------------------
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "docs_path" not in st.session_state:
    st.session_state.docs_path = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "display_history" not in st.session_state:
    st.session_state.display_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "alert_message" not in st.session_state:
    st.session_state.alert_message = None
if "alert_type" not in st.session_state:
    st.session_state.alert_type = None
if "reset_trigger" not in st.session_state:
    st.session_state.reset_trigger = False

# -------------------- Reset Functions --------------------
def clear_chat():
    st.session_state.chat_history = []
    st.session_state.display_history = []
    st.success("Chat history cleared!")

def reset_documents():
    st.session_state.reset_trigger = True
    for f in st.session_state.docs_path:
        try:
            os.remove(f)
        except:
            pass
    st.session_state.docs_path.clear()
    st.session_state.uploaded_files = []
    st.session_state.uploader_key += 1
    st.session_state.vectorstore = None
    st.session_state.alert_message = "Documents reset successfully!"
    st.session_state.alert_type = "success"
    st.rerun()

# -------------------- Load Documents --------------------
def load_documents(filepaths):
    documents = []
    exist_documents = [f.name for f in st.session_state.uploaded_files]
    new_documents = []
    for file in filepaths:
        if file.name not in exist_documents:
            new_documents.append(file.name)
            st.session_state.uploaded_files.append(file)
        else:
            continue

        with tempfile.NamedTemporaryFile(delete=False,suffix=os.path.splitext(file.name)[1]) as tmp:
         #   tmp.write(file.getbuffer())
            temp_file_path = os.path.join(tempfile.gettempdir(), file.name)
            with open(temp_file_path, "wb") as f:
                f.write(file.getbuffer())

        ext = file.name.split(".")[-1].lower()
        match ext:
            case "pdf":
                loader = PyPDFLoader(temp_file_path)
                documents.extend(loader.load())
            case "docx":
                loader = Docx2txtLoader(temp_file_path)
                documents.extend(loader.load())
            case "xls"| "xlsx":
                df = pd.read_excel(temp_file_path)
                text_data = df.to_string()
                documents.append(Document(page_content = text_data,metadata = {"source":file}))
            case "txt":
                loader = TextLoader(temp_file_path)
                documents.extend(loader.load())
            case _:
                continue

        st.session_state.docs_path.append(temp_file_path)
        #st.sidebar.warning(os.path.basename(temp_file_path))
    if documents:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        # 3. Create embeddings and vectorstore
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        if st.session_state.vectorstore is None:
            st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
        else:
            st.session_state.vectorstore.add_documents(chunks)
    uploaded_names = [f.name for f in uploaded_files]
    duplicates = [name for name in uploaded_names if name in exist_documents and name not in new_documents]
    if new_documents and duplicates:
        st.session_state.alert_message = (f"✅ Added {len(new_documents)} new file(s).\n"
                           f"{len(duplicates)} duplicate(s) Ignored.")
        st.session_state.alert_type = "success"
    elif new_documents:
        st.session_state.alert_message = f"✅ Added {len(new_documents)} new file(s)."
        st.session_state.alert_type = "success"
    elif duplicates:
        st.session_state.alert_message = f"⚠️ All selected files already loaded! - skipped."
        st.session_state.alert_type = "warning"
        st.session_state.uploader_key += 1
    st.rerun()
    return True


# -------------------- File Upload Section --------------------
st.sidebar.header("📎 UPLOAD")
uploaded_files = st.sidebar.file_uploader(
    "📂 Upload your documents",
    type=["pdf", "docx", "xlsx", "txt"],
    accept_multiple_files=True,
    key=f"uploaded_{st.session_state.uploader_key}"
)
st.sidebar.markdown("---")

# -------------------- Setup Sidebar --------------------
# Display messages from rerun
if st.session_state.alert_message:
    if st.session_state.alert_type == "alert":
        st.sidebar.warning(st.session_state.alert_message)
    elif st.session_state.alert_type == "success":
        st.sidebar.success(st.session_state.alert_message)

    st.session_state.alert_message = None
    st.session_state.alert_type = None
# Load all documents and prepare
if uploaded_files:
    load_documents(uploaded_files)

if st.session_state.uploaded_files and not st.session_state.reset_trigger:
    for f in st.session_state.uploaded_files:
        st.sidebar.write(f.name )

if st.sidebar.button("🗑️ Clear Chat"):
    clear_chat()

if st.sidebar.button("♻️ Reset Documents"):
    reset_documents()

if st.session_state.reset_trigger:
    st.session_state.reset_trigger = False
# -------------------- Setup Main --------------------
for message in st.session_state.display_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_question = st.chat_input("Input your question")
if user_question:
    with st.chat_message("user"):
        st.markdown(user_question)
    msg = {"role": "user",
           "content": user_question}
    st.session_state.display_history.append(msg)
    # 4. Create chat model
    if not st.session_state.vectorstore:
        answer = "Please upload at least one document first."
    else:
        llm = ChatGroq(model="llama-3.3-70b-versatile",api_key=gApi_key)
        # 5. Build conversational retrieval chain
        retriever = st.session_state.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory
        )
        # 6. Ask the question
        #response = qa_chain.run(user_question)
        # -------------------- Question Input --------------------
        try:
            response = qa_chain.invoke({"question": user_question,
                                        "chat_history": st.session_state.chat_history or []})
            answer = response["answer"]
        except Exception as e:
            answer = f" Error during AI check: {e}"

    st.session_state.chat_history.append((user_question, answer))
    with st.chat_message("assistant"):
        st.markdown(answer)
    msg = {"role": "assistant",
           "content": answer}
    st.session_state.display_history.append(msg)
