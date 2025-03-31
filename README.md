# MultiPDF Chat App
## Introduction
---
This project allows users to chat with their PDF documents using an AI-powered conversational system. It extracts text from PDFs, processes them into embeddings, and utilizes a conversational retrieval chain to provide intelligent responses based on document content.
---

## How It Works
![]()


## Features
- Upload multiple PDFs and extract text.
- Process documents into embeddings using sentence-transformers/all-MiniLM-L6-v2.
- Store embeddings in a persistent Chroma database.
- Utilize Ollama with the llama3 model for AI-driven conversations.
- Maintain conversation history using ConversationBufferMemory.
- Responsive and interactive UI built with Streamlit.
---
## Installation
---
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- pip
- Ollama
- Streamlit

### Setup
1. Clone the repository:
- git clone https://github.com/kodeking-081/Langchain-chat_with_pdf.git
- cd <repository-folder>

2. Create a virtual enviroment:
- python -m <venv_name> venv
- source <venv_name>/Scripts/activate

3. Install Dependencies:
- pip install -r requirements.txt

4. Pull the required Ollama model:
- ollama pull llama3

5. Run the Ollama server:
-ollama serve

6. Start the application:
- streamlit run app.py

## Usage

1. Upload your PDF files via the sidebar.

2. Click on "Process" to extract and store document embeddings.

3. Ask questions about the uploaded documents in the main chat input.

4. Receive AI-generated responses based on document content.
