import streamlit as st
import os
import tempfile
from io import StringIO
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Groq and Sentence Transformers
from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Configure Streamlit page
st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG PDF Chatbot")
st.markdown("Upload PDFs and ask questions about their content!")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'docs_processed' not in st.session_state:
    st.session_state.docs_processed = False

# Get API Key from environment
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("❌ GROQ_API_KEY not found in environment variables. Please add it to your .env file.")
    st.info("Create a .env file in your project root and add: GROQ_API_KEY=your_api_key_here")
    st.stop()

# Initialize Groq client
try:
    groq_client = Groq(api_key=groq_api_key)
    st.sidebar.success("✅ Groq API connected successfully")
except Exception as e:
    st.error(f"Error initializing Groq client: {str(e)}")
    st.stop()

# Sidebar for document upload
with st.sidebar:
    st.header("📄 Document Upload")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        key="pdf_uploader"
    )

    if uploaded_files:
        if st.button("Process Documents", type="primary"):
            with st.spinner("Processing documents..."):
                try:
                    # Initialize components
                    @st.cache_resource
                    def load_sentence_transformer():
                        return SentenceTransformer('all-MiniLM-L6-v2')

                    embedding_model = load_sentence_transformer()

                    # Create a custom embedding function for Chroma
                    class SentenceTransformerEmbeddings:
                        def __init__(self, model):
                            self.model = model

                        def embed_documents(self, texts):
                            return self.model.encode(texts).tolist()

                        def embed_query(self, text):
                            return self.model.encode([text]).tolist()[0]

                    embeddings = SentenceTransformerEmbeddings(embedding_model)

                    # Process PDFs
                    all_docs = []
                    for uploaded_file in uploaded_files:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name

                        # Load PDF
                        loader = PyPDFLoader(tmp_file_path)
                        docs = loader.load()
                        all_docs.extend(docs)

                        # Clean up temp file
                        os.unlink(tmp_file_path)

                    # Split documents
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    split_docs = text_splitter.split_documents(all_docs)

                    # Create Chroma vector store
                    persist_directory = "./chroma_db"

                    # Remove existing directory if it exists
                    import shutil
                    if os.path.exists(persist_directory):
                        shutil.rmtree(persist_directory)

                    vectorstore = Chroma.from_documents(
                        documents=split_docs,
                        embedding=embeddings,
                        persist_directory=persist_directory
                    )

                    st.session_state.vectorstore = vectorstore
                    st.session_state.docs_processed = True
                    st.success(f"Successfully processed {len(uploaded_files)} PDF(s) with {len(split_docs)} chunks!")

                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")

# Main chat interface
if st.session_state.docs_processed and st.session_state.vectorstore:

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Retrieve relevant documents
                    retriever = st.session_state.vectorstore.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 4}
                    )
                    relevant_docs = retriever.invoke(prompt)

                    # Prepare context
                    context = "\n\n".join([doc.page_content for doc in relevant_docs])

                    # Create prompt template
                    template = '''Answer the question based on the following context:

Context:
{context}

Question: {question}

Answer: Provide a detailed answer based on the context. If the answer is not in the context, say "I don't have enough information to answer this question."'''

                    # Generate response using Groq
                    chat_completion = groq_client.chat.completions.create(
                        messages=[
                            {
                                "role": "user",
                                "content": template.format(context=context, question=prompt)
                            }
                        ],
                        model="llama-3.3-70b-versatile",
                        temperature=0.1,
                        max_tokens=1000
                    )

                    response = chat_completion.choices[0].message.content
                    st.markdown(response)

                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

else:
    st.info("👆 Please upload and process PDF documents to start chatting!")

    # Show example usage
    with st.expander("ℹ️ How to use this app"):
        st.markdown('''
        1. **Setup Environment**: Create a `.env` file with your Groq API key: `GROQ_API_KEY=your_api_key_here`
        2. **Upload PDFs**: Select one or more PDF files using the file uploader
        3. **Process Documents**: Click "Process Documents" to extract and index the content
        4. **Ask Questions**: Once processed, you can ask questions about the content in the chat interface

        **Features:**
        - 🤖 Powered by Groq's fast LLM inference
        - 🔍 Uses Sentence Transformers for embeddings
        - 💾 Stores vectors in Chroma database
        - 📝 Supports multiple PDF uploads
        - 💬 Interactive chat interface
        - 🔐 Secure API key management via environment variables
        ''')

# Footer
st.markdown("---")
