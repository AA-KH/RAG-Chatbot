import os
import json
import fitz
import docx
import tiktoken
import numpy as np
import faiss
from pathlib import Path

from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
INDEX_PATH = "index.faiss"
META_PATH = "meta.json"

def read_pdf(path):
    doc = fitz.open(path)
    pages = [page.get_text("text") for page in doc]
    return "\n".join(pages)

def read_docx(path):
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def read_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def read_file(path):
    p = Path(path)
    if p.suffix.lower() in [".pdf"]:
        return read_pdf(path)
    if p.suffix.lower() in [".docx"]:
        return read_docx(path)
    if p.suffix.lower() in [".txt", ".md"]:
        return read_txt(path)
    raise ValueError(f"Unsupported file: {path}")

def split_into_chunks(text, max_tokens=500, overlap=50, model="gpt-3.5-turbo"):
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens))
        start += max_tokens - overlap
    return chunks

model = SentenceTransformer("all-MiniLM-L6-v2")

def create_embeddings(texts):
    return model.encode(texts, normalize_embeddings=True).tolist()

def build_index_from_folder(data_dir=DATA_DIR):
    docs = []
    for path in Path(data_dir).glob("*"):
        if path.is_file():
            text = read_file(path)
            chunks = split_into_chunks(text)
            for c in chunks:
                docs.append({"text": c, "source": str(path)})
    print(f"Total chunks: {len(docs)}")

    texts = [d["text"] for d in docs]
    embeddings = create_embeddings(texts)

    emb_np = np.array(embeddings).astype("float32")
    faiss.normalize_L2(emb_np)

    dim = emb_np.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb_np)

    faiss.write_index(index, INDEX_PATH)
    meta = [{"text": d["text"], "source": d["source"]} for d in docs]
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Index built and saved.")

if __name__ == "__main__":
    build_index_from_folder()
