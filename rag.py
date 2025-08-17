# rag.py
from uuid import uuid4
from pathlib import Path
from typing import List, Tuple
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq  # or your LLM of choice
from langchain_ollama import ChatOllama

import tiktoken

load_dotenv()

# --- Constants ---
CHROMA_DIR = Path("./chroma_db")
COLLECTION_NAME = "real_estate_urls"
EMBED_MODEL = "BAAI/bge-m3"  # or "sentence-transformers/all-MiniLM-L6-v2"
#GROQ_MODEL = "llama-3.1-70b-versatile"  # example; set your own

# Token-aware length function (approx; pick an encoding)
enc = tiktoken.get_encoding("cl100k_base")
def tok_len(s: str) -> int:
    return len(enc.encode(s))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,            # tokens
    chunk_overlap=100,         # tokens
    length_function=tok_len,
    separators=["\n\n", "\n", ". ", " "],
)

def _embedder():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},  # important
    )

def _vectorstore(embedding_fn=None):
    CHROMA_DIR.mkdir(exist_ok=True, parents=True)
    if embedding_fn is None:
        embedding_fn = _embedder()
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embedding_fn,
        collection_metadata={"hnsw:space": "cosine"},  # ensure cosine
    )

def process_urls(urls: List[str]) -> int:
    urls = [u.strip() for u in urls if u and u.strip()]
    urls = list(dict.fromkeys(urls))  # dedupe, preserve order
    if not urls:
        raise ValueError("No valid URLs provided.")

    loader = UnstructuredURLLoader(urls=urls)
    docs = loader.load()
    if not docs:
        raise RuntimeError("No content could be loaded from the provided URLs.")

    # Keep useful metadata
    for d in docs:
        d.metadata.setdefault("source", d.metadata.get("source") or d.metadata.get("url"))
        d.metadata["doc_id"] = str(uuid4())

    chunks = splitter.split_documents(docs)

    vs = _vectorstore()
    vs.add_documents(chunks)
    vs.persist()
    return len(chunks)

# Optional: a more helpful prompt for citations
QA_PROMPT = PromptTemplate.from_template(
    """You are a helpful real estate assistant. Use the provided context to answer the question.
If the answer is not in the context, say you don't know.

Question: {question}

Context:
{context}

Answer (cite sources if relevant):"""
)

def _retriever(k: int = 4):
    vs = _vectorstore()
    return vs.as_retriever(search_kwargs={"k": k})

def _llm():
    # Configure your LLM; can switch to OpenAI or Ollama easily
    #return ChatGroq(model=GROQ_MODEL, temperature=0)
    return ChatOllama(model="llama3.2:1b")

def generate_answer(query: str) -> Tuple[str, List[str]]:
    vs = _vectorstore()
    if vs._collection.count() == 0:
        raise RuntimeError("Vector store is empty. Process URLs first.")

    retriever = _retriever(k=4)
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=_llm(),
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT},
    )

    result = chain.invoke({"question": query})
    answer = result.get("answer", "").strip()

    src_docs = result.get("source_documents") or []
    sources = []
    seen = set()
    for d in src_docs:
        s = d.metadata.get("source") or d.metadata.get("url")
        if s and s not in seen:
            sources.append(s)
            seen.add(s)

    return answer, sources
