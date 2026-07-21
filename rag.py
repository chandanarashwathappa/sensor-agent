from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_mistralai import ChatMistralAI
import os

load_dotenv()  # loads your Mistral key

# --- STEP 1: Load the knowledge base text files ---
docs = []
for filename in os.listdir("knowledge_base"):
    if filename.endswith(".txt"):
        with open(f"knowledge_base/{filename}", "r", encoding="utf-8") as f:
            docs.append(Document(page_content=f.read(), metadata={"source": filename}))
print(f"Loaded {len(docs)} documents.")

# --- STEP 2: Split documents into chunks (manual, no extra library) ---
def split_text(text, chunk_size=1000, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap  # move forward, keeping a small overlap
    return chunks

chunks = []
for doc in docs:
    for piece in split_text(doc.page_content):
        chunks.append(Document(page_content=piece, metadata=doc.metadata))
print(f"Split into {len(chunks)} chunks.")

# --- STEP 3: Create embeddings using Mistral (no torch needed) ---
embeddings = MistralAIEmbeddings(model="mistral-embed")

# --- STEP 4: Build the FAISS vector store ---
vectorstore = FAISS.from_documents(chunks, embeddings)
print("FAISS vector store built.")

# --- STEP 5: Connect to Mistral LLM ---
llm = ChatMistralAI(model="mistral-small-latest", max_tokens=512)

# --- STEP 6: The RAG function — retrieve, then answer ---
def ask(question):
    relevant_chunks = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join(chunk.page_content for chunk in relevant_chunks)

    prompt = f"""You are an assistant for the AI4I 2020 predictive maintenance dataset.
Answer the question using ONLY the context below. If the answer is not in the context, say you don't know.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    response = llm.invoke(prompt)
    return response.content

# --- STEP 7: Test it ---
if __name__ == "__main__":
    question = "What is the overall failure rate in the dataset?"
    print("\nQUESTION:", question)
    print("ANSWER:", ask(question))