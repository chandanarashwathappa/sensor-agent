# --- Block torch/tensorflow before anything imports them ---
import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "0"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_mistralai import ChatMistralAI

load_dotenv()

# --- Page setup ---
st.set_page_config(page_title="Industrial Sensor Query Agent", page_icon="⚙️")
st.title("⚙️ Industrial Sensor Query Agent")
st.write("Ask questions about the AI4I 2020 Predictive Maintenance dataset.")

# --- Build the RAG system ONCE and cache it ---
@st.cache_resource
def build_rag_system():
    docs = []
    for filename in os.listdir("knowledge_base"):
        if filename.endswith(".txt"):
            with open(f"knowledge_base/{filename}", "r", encoding="utf-8") as f:
                docs.append(Document(page_content=f.read(), metadata={"source": filename}))

    def split_text(text, chunk_size=1000, overlap=100):
        chunks = []
        start = 0
        while start < len(text):
            chunks.append(text[start:start + chunk_size])
            start += chunk_size - overlap
        return chunks

    chunks = []
    for doc in docs:
        for piece in split_text(doc.page_content):
            chunks.append(Document(page_content=piece, metadata=doc.metadata))

    embeddings = MistralAIEmbeddings(model="mistral-embed")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    llm = ChatMistralAI(model="mistral-small-latest", max_tokens=512)
    return vectorstore, llm

# --- Build the system, showing status and any error on the page ---
try:
    with st.spinner("Loading knowledge base... (first load takes a few seconds)"):
        vectorstore, llm = build_rag_system()
    st.success("Ready! Ask a question below.")
except Exception as e:
    st.error(f"Error building the system: {e}")
    st.stop()

# --- The RAG answer function ---
def ask(question):
    relevant_chunks = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join(chunk.page_content for chunk in relevant_chunks)
    prompt = f"""You are an assistant for the AI4I 2020 predictive maintenance dataset.
Answer the question using ONLY the context below. If the answer is not in the context, say you don't know.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
    return llm.invoke(prompt).content, relevant_chunks

# --- The user interface ---
question = st.text_input("Your question:", placeholder="e.g. What causes heat dissipation failure?")

if st.button("Ask") and question:
    with st.spinner("Thinking..."):
        answer, sources = ask(question)
    st.subheader("Answer")
    st.write(answer)
    with st.expander("See the sources used"):
        for i, chunk in enumerate(sources):
            st.markdown(f"**Source {i+1}** (from {chunk.metadata['source']}):")
            st.text(chunk.page_content[:300] + "...")