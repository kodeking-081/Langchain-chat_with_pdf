import os
import torch

# Fix for torch.classes issue
torch.classes.__path__ = []

import streamlit as st
import requests
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import Chroma  # Updated import
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.llms import Ollama  # Updated import
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Load environment variables
load_dotenv()

# Function to check if Ollama server is running
def check_ollama_server():
    try:
        response = requests.get("http://localhost:11434")
        return response.status_code == 200
    except requests.RequestException:
        return False

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
            else:
                st.warning("⚠️ Some pages may not contain extractable text.")
    return text.strip() if text else None

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

# Function to create and persist vectorstore
def get_vectorstore(text_chunks):
    persist_dir = "chroma_db"  # Directory to store embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if os.path.exists(persist_dir):
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    
    vectorstore = Chroma.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    vectorstore.persist()
    return vectorstore

# Function to initialize the LLM
def get_llm():
    if not check_ollama_server():
        st.error("⚠️ Ollama server not running. Please start it using: `ollama serve`")
        return None
    
    try:
        return Ollama(model="llama3", temperature=0.3, top_p=0.9, num_ctx=4096)
    except Exception as e:
        st.error(f"❌ Failed to initialize Ollama: {e}")
        return None

# Function to create a conversational retrieval chain
def get_conversational_chain(vectorstore):
    llm = get_llm()
    if not llm:
        return None
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory
    )

# Function to handle user input and display conversation
def handle_userinput(user_question):
    if not st.session_state.conversation:
        return
    
    try:
        response = st.session_state.conversation.invoke({"question": user_question})
        st.session_state.chat_history = response['chat_history']
        
        for i, message in enumerate(st.session_state.chat_history):
            role = "🧑‍💻 **You:**" if i % 2 == 0 else "🤖 **Bot:**"
            st.markdown(f"{role}\n{message.content}")
    except Exception as e:
        st.error(f"❌ Error: {e}")

# Main Streamlit app
def main():
    st.set_page_config(page_title="Chat with PDFs", page_icon="📚")
    st.header("📄 Chat with Your PDFs")

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # User input for questions
    user_question = st.text_input("Ask a question about your documents:")
    if user_question and st.session_state.conversation:
        handle_userinput(user_question)

    # Sidebar for PDF upload
    with st.sidebar:
        st.subheader("📂 Upload Your PDFs")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click Process", 
            accept_multiple_files=True,
            type="pdf"
        )

        if st.button("🚀 Process"):
            if not pdf_docs:
                st.warning("⚠️ Please upload at least one PDF file.")
                return
                
            with st.spinner("Processing..."):
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text:
                        st.error("❌ No text could be extracted from the PDFs.")
                        return

                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversational_chain(vectorstore)
                    
                    if st.session_state.conversation:
                        st.success("✅ Processing complete! You can now chat with your PDFs.")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

if __name__ == '__main__':
    main()
