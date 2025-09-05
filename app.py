from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import faiss
import os
import numpy as np
import json

from sentence_transformers import SentenceTransformer
from gpt4all import GPT4All

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LOCAL_LLM_MODEL = "q4_0-orca-mini-3b.gguf"
llm = GPT4All(LOCAL_LLM_MODEL, model_path="./models")

INDEX_FILE = "index.faiss"
DOCS_FILE = "meta.json"


print("Loading embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

print("Loading local LLM...")
llm = GPT4All(LOCAL_LLM_MODEL, model_path="./models") 

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

def get_embedding(text: str):
    """Get vector embedding for given text using SentenceTransformers."""
    return embedder.encode([text])[0]


def load_faiss_index():
    if not os.path.exists(INDEX_FILE):
        print("No existing FAISS index found. Creating a new one...")
        return faiss.IndexFlatL2(384), []

    index = faiss.read_index(INDEX_FILE)

    with open(DOCS_FILE, "r", encoding="utf-8") as f:
        docs = json.load(f)

    return index, docs


def save_faiss_index(index, docs):
    faiss.write_index(index, INDEX_FILE)

    with open(DOCS_FILE, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

index, documents = load_faiss_index()


class Query(BaseModel):
    query: str


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read())


@app.post("/add_doc")
async def add_doc(doc: Query):
    global index, documents
    embedding = get_embedding(doc.query)
    index.add(np.array([embedding], dtype=np.float32))
    documents.append(doc.query)
    save_faiss_index(index, documents)
    return {"status": "Document added successfully"}


@app.post("/query")
async def query(q: Query):
    global index, documents

    if len(documents) == 0:
        return JSONResponse({"answer": "No documents in the database yet."})

    q_emb = get_embedding(q.query).astype(np.float32)
    D, I = index.search(np.array([q_emb]), k=3)

    retrieved_docs = [documents[i] for i in I[0] if i < len(documents)]

    retrieved_texts = [doc["text"] if isinstance(doc, dict) else str(doc) for doc in retrieved_docs]

    context = "\n".join(retrieved_texts)
    prompt = f"Answer the following question using only the provided context:\n\nContext:\n{context}\n\nQuestion: {q.query}\nAnswer:"

    answer = llm.generate(prompt, max_tokens=200)
    return JSONResponse({"answer": answer})
    
