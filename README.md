# MultiPDF Chat App
## Introduction
---
 This project allows users to chat with their PDF documents using an AI-powered conversational system. It extracts text from PDFs, processes them into embeddings, and utilizes a conversational retrieval chain to 
 provide intelligent responses based on document content.
---

## How It Works
![](https://github.com/kodeking-081/Langchain-chat_with_pdf/blob/main/docs/PDF-LangChain.jpg)

The application follows these steps to provide responses to your questions:

1. PDF Loading: The app reads multiple PDF documents and extracts their text content.

2. Text Chunking: The extracted text is divided into smaller chunks that can be processed effectively.

3. Language Model: The application utilizes a language model to generate vector representations (embeddings) of the text chunks.

4. Similarity Matching: When you ask a question, the app compares it with the text chunks and identifies the most semantically similar ones.

5. Response Generation: The selected chunks are passed to the language model, which generates a response based on the relevant content of the PDFs.


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
1. Clone the repository:<br>
git clone https://github.com/kodeking-081/Langchain-chat_with_pdf.git<br>
cd (repository-folder)

2. Create a virtual enviroment:<br>
python -m <venv_name> venv<br>
source <venv_name>/Scripts/activate

3. Install Dependencies:<br>
pip install -r requirements.txt

4. Get API Keys:<br>
Obtain an API key from HuggingFace and add it to the .env file in the project directory.<br>
OPENAI_API_KEY=your_secret_api_key

6. Pull the required Ollama model:<br>
ollama pull llama3

7. Run the Ollama server:<br>
ollama serve

8. Start the application:<br>
streamlit run app.py

## License
This project is under the MIT license.

