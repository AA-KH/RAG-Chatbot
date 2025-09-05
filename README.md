# RAG Chatbot with FastAPI and GPT4All

This project is a Retrieval-Augmented Generation (RAG) chatbot that uses FAISS, SentenceTransformers, and GPT4All. It allows ingestion of documents (PDF, DOCX, TXT, and Markdown) into a vector database and enables querying them with a local Large Language Model (LLM). The backend is powered by FastAPI, and an optional Gradio frontend is included for quick testing.

---

## Features
- 📄 Ingest PDF, DOCX, TXT, and Markdown files.
- 🔍 Build embeddings using SentenceTransformers (`all-MiniLM-L6-v2`).
- 🗂️ Store and search document vectors with FAISS.
- 🤖 Query documents using a local GPT4All LLM.
- ⚡ FastAPI backend with REST endpoints.
- 🎛️ Optional Gradio interface for interactive testing.

---

## Project Structure
 - app.py # FastAPI app to serve the chatbot
 - ingest.py # Script to process and embed documents
 - frontend.py # Alternative frontend implementation
 - requirements.txt # Python dependencies
 - meta.json # Metadata for documents (auto-generated)
 - index.faiss # FAISS index file (auto-generated)
 - data/ # Folder to place input documents
 - models/ # Folder for GPT4All local models
 - venv/ # Virtual environment (excluded from Git)

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/AA-KH/RAG-Chatbot
   cd rag-chatbot

2. Create and activate a virtual environment:
    python3 -m venv venv
    source venv/bin/activate   # On macOS/Linux
    venv\Scripts\activate      # On Windows

3. Install dependencies:
    pip install -r requirements.txt

4. Download a local LLM (example: Orca Mini 3B in GGUF format) and put it in the models/ folder.
   Example: models/q4_0-orca-mini-3b.gguf


## Usage
Step 1: Add Documents
- Place your .pdf, .docx, .txt, or .md files into the data/ folder.

Step 2: Build Index
- Run the ingestion script to process documents and build the FAISS index:
  python ingest.py

Step 3: Start the API
- Run the FastAPI server:
  uvicorn app:app --reload
- API will be available at: http://127.0.0.1:8000

Step 4: Query
- You can POST queries to /query endpoint.
- You can also use /add_doc to add documents dynamically.

Step 5: (Optional) Gradio Frontend
- For a quick interactive demo, you can launch Gradio:
  python frontend.py

## Notes
1. The index.faiss and meta.json files are auto-generated after ingestion.
2. The models/ folder should contain the GPT4All .gguf model files (not included in repo).
3. For best results, use a lightweight GGUF model like orca-mini-3b or mistral-7b quantized.



