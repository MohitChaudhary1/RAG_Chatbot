RAG PDF Chatbot with Streamlit
A simple RAG (Retrieval-Augmented Generation) application built with Streamlit that allows users to upload PDF documents and ask questions about their content.

Features
📄 PDF Upload: Support for multiple PDF file uploads

🤖 Groq LLM Integration: Fast inference using Groq's API

🔍 Sentence Transformers: High-quality embeddings using all-MiniLM-L6-v2

💾 Chroma Vector Store: Persistent vector storage for document chunks

💬 Chat Interface: Interactive chat interface with conversation history

🔄 Session State: Maintains conversation and document state across interactions

🔐 Environment Variables: Secure API key management via .env file

Setup
Install Dependencies:

bash
pip install -r requirements.txt
Environment Configuration:

Create a .env file in the project root directory

Add your Groq API key:

text
GROQ_API_KEY=your_actual_groq_api_key_here
Get your API key from Groq Console

Run the Application:

bash
streamlit run app.py
Environment Variables
Create a .env file with the following variables:

bash
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional (uncomment if you want to customize)
# CHROMA_DB_PATH=./chroma_db
# EMBEDDING_MODEL=all-MiniLM-L6-v2
# CHUNK_SIZE=1000
# CHUNK_OVERLAP=200
How to Use
Start the Application: Run streamlit run app.py

Upload PDFs: Select one or more PDF files to upload

Process Documents: Click "Process Documents" to index the content

Ask Questions: Use the chat interface to ask questions about the uploaded documents

Technology Stack
Frontend: Streamlit

LLM: Groq API (Llama 3.3 70B)

Embeddings: Sentence Transformers (all-MiniLM-L6-v2)

Vector Store: Chroma DB

PDF Processing: LangChain PyPDFLoader

Text Splitting: LangChain RecursiveCharacterTextSplitter

Environment Management: python-dotenv

Architecture
Document Loading: PDFs are loaded using PyPDFLoader

Text Splitting: Documents are split into chunks using RecursiveCharacterTextSplitter

Embedding: Chunks are embedded using Sentence Transformers

Vector Storage: Embeddings are stored in Chroma DB with persistence

Retrieval: Relevant chunks are retrieved based on user queries

Generation: Groq LLM generates responses based on retrieved context

Configuration
Chunk Size: 1000 characters

Chunk Overlap: 200 characters

Retrieval: Top 4 similar chunks

Model: llama-3.3-70b-versatile

Embedding Model: all-MiniLM-L6-v2

Security Features
Environment Variables: API keys stored securely in .env file (not in code)

Git Ignore: .env file should be added to .gitignore to prevent accidental commits

No API Key Exposure: Keys never displayed in the UI

File Structure
text
project/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (create this)
├── .env.example       # Environment variables template
├── README.md          # Documentation
└── chroma_db/         # Vector database (created automatically)
Troubleshooting
API Key Error: Ensure your .env file exists and contains a valid GROQ_API_KEY

Import Errors: Make sure all dependencies are installed with pip install -r requirements.txt

PDF Processing Issues: Check that your PDF files are not corrupted and are text-based (not scanned images)

Memory Issues: For large PDFs, consider reducing chunk_size or processing fewer files at once

Notes
The vector database is persisted locally in the ./chroma_db directory

Session state maintains conversation history and processed documents

The application supports multiple PDF uploads in a single session

Add .env to your .gitignore file to keep API keys secure
