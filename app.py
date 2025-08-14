
"""
app.py – Streamlit RAG PDF assistant with three modes:
1. Chat  – conversational QA
2. Q&A   – one-shot question & answer
3. Quiz  – auto-generates 20 Q-and-A quiz pairs

Uses:
• Groq Llama-3.3-70B (API key loaded from .env)
• SentenceTransformers embeddings (all-MiniLM-L6-v2)
• Chroma for vector storage
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from textwrap import dedent
from typing import List, Dict

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# ────────────────────────────── ENV / CONFIG ───────────────────────────────────
load_dotenv()  # reads .env

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found in environment. "
             "Create a `.env` file with GROQ_API_KEY=your_key_here")
    st.stop()

st.set_page_config(page_title="RAG PDF Assistant",
                   page_icon="📚",
                   layout="wide")

# ────────────────────────────── SESSION KEYS ───────────────────────────────────
defaults = {
    "vectorstore": None,
    "docs_processed": False,
    "messages": [],          # chat history
    "quiz": [],              # generated quiz list
    "quiz_generated": False
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ────────────────────────────── HELPERS ────────────────────────────────────────
@st.cache_resource(show_spinner="📥 Loading sentence-transformer model…")
def load_st_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


class STEmbeddings:               # minimal wrapper for LangChain
    def __init__(self, model: SentenceTransformer):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text]).tolist()[0]


def build_vectorstore(files) -> Chroma:
    """Load PDFs ➜ split ➜ embed ➜ return persistent Chroma store."""
    model = load_st_model()
    embeddings = STEmbeddings(model)

    pages = []
    for up_file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(up_file.getvalue())
            loader = PyPDFLoader(tmp.name)
            pages.extend(loader.load())
        Path(tmp.name).unlink(missing_ok=True)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(pages)

    # fresh DB each run
    db_path = Path("chroma_db")
    if db_path.exists():
        shutil.rmtree(db_path)

    return Chroma.from_documents(docs, embeddings, persist_directory=str(db_path))


def retrieve_context(query: str, k: int = 4) -> str:
    """Fetch top-k relevant chunks and concatenate."""
    retriever = st.session_state.vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )
    docs = retriever.invoke(query)
    return "\n\n".join(d.page_content for d in docs)


groq_client = Groq(api_key=GROQ_API_KEY)


def call_groq(context: str, question: str,
              max_tokens: int = 800, temp: float = 0.1) -> str:
    prompt = (f"Answer the question using the context below. "
              f"If the answer is not contained, say you don't know.\n\n"
              f"Context:\n{context}\n\nQuestion: {question}")
    resp = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=temp,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def generate_quiz() -> List[Dict[str, str]]:
    """Return 20 question/answer dicts generated from the doc corpus."""
    ctx = retrieve_context("overview of key points", k=60)
    prompt = dedent(f"""
        Create a 20-question quiz about the following document content.
        For each item provide one question and its short answer.
        Format exactly:

        Q: <question>
        A: <answer>
        ---

        Content:
        {ctx}
    """)

    resp = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=1200,
        messages=[{"role": "user", "content": prompt}],
    ).choices[0].message.content

    qa_pairs = []
    for block in resp.split("---"):
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if len(lines) >= 2 and lines[0].startswith("Q:") and lines[1].startswith("A:"):
            qa_pairs.append({
                "question": lines[0][2:].strip(),
                "answer": lines[1][2:].strip()
            })
    return qa_pairs[:20]


# ────────────────────────────── SIDEBAR ────────────────────────────────────────
st.sidebar.header("📑 Document panel")

uploaded = st.sidebar.file_uploader("Upload PDF file(s)", type="pdf",
                                    accept_multiple_files=True)

feature = st.sidebar.selectbox("Choose feature",
                               ("Chat", "Quiz", "Q&A"),
                               index=0)

if uploaded and st.sidebar.button("Process documents", use_container_width=True):
    with st.spinner("🔧 Building vector store…"):
        st.session_state.vectorstore = build_vectorstore(uploaded)
        st.session_state.docs_processed = True
        st.session_state.messages.clear()
        st.session_state.quiz_generated = False
        st.success("✅ Documents processed successfully!")

# ────────────────────────────── MAIN LAYOUT ────────────────────────────────────
st.title("📚 RAG PDF Assistant")

if not st.session_state.docs_processed:
    st.info("➡️  Upload PDF(s) on the left and click *Process documents* to begin.")
    st.stop()

# 1️⃣ CHAT MODE ─────────────────────────────────────────────────────────────────
if feature == "Chat":
    st.subheader("💬 Chat with your documents")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask anything about the documents"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                context = retrieve_context(prompt)
                answer = call_groq(context, prompt)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant",
                                                  "content": answer})

# 2️⃣ Q&A MODE ─────────────────────────────────────────────────────────────────
elif feature == "Q&A":
    st.subheader("❓ Single question & answer")
    q = st.text_input("Your question", placeholder="Type a question about the PDFs")
    if st.button("Get answer") and q:
        with st.spinner("Searching…"):
            ctx = retrieve_context(q)
            st.success(call_groq(ctx, q, max_tokens=400))

# 3️⃣ QUIZ MODE ────────────────────────────────────────────────────────────────
elif feature == "Quiz":
    st.subheader("📝 20-question quiz generator")
    if not st.session_state.quiz_generated:
        if st.button("Generate quiz"):
            with st.spinner("Creating quiz…"):
                st.session_state.quiz = generate_quiz()
                st.session_state.quiz_generated = True

    if st.session_state.quiz_generated:
        for i, qa in enumerate(st.session_state.quiz, start=1):
            exp = st.expander(f"Q{i}: {qa['question']}", expanded=False)
            exp.markdown(f"**Answer:** {qa['answer']}")
        if st.button("Regenerate quiz"):
            with st.spinner("Regenerating…"):
                st.session_state.quiz = generate_quiz()

# ────────────────────────────── FOOTER ────────────────────────────────────────
st.markdown("---")

