# rag-session09

A demonstration of a simple **Retrieval-Augmented Generation (RAG)** architecture using **Python**, **LangChain**, and **Cohere**.

## What's included

| File | Purpose |
|------|---------|
| `01_embed_documents.ipynb` | Load PDFs → chunk → embed with Cohere → save FAISS index |
| `02_retrieve_rerank_generate.ipynb` | Load FAISS → retrieve → rerank → generate grounded answer |
| `requirements.txt` | Python dependencies |
| `.devcontainer/devcontainer.json` | GitHub Codespaces / VS Code Dev Container configuration |

## Quick start

### Option A – GitHub Codespaces (recommended)

1. Click **Code → Codespaces → Create codespace on main**.  
   The devcontainer will spin up Python 3.12 and automatically install all dependencies.
2. Copy `.env.example` to `.env` and fill in your Cohere API key:
   ```
   COHERE_API_KEY=your-key-here
   ```
3. Place your PDF files in the `./docs/` folder.
4. Open and run `01_embed_documents.ipynb` to build the FAISS index.
5. Open and run `02_retrieve_rerank_generate.ipynb` to query your documents.

### Option B – Local setup

```bash
pip install -r requirements.txt
# then follow steps 2–5 above
```

## Pipeline overview

```
PDF files
   │
   ▼
[PyPDFLoader]  →  pages
   │
   ▼
[RecursiveCharacterTextSplitter]  →  chunks
   │
   ▼
[CohereEmbeddings: embed-english-v3.0]  →  vectors
   │
   ▼
[FAISS index]  ──── saved to ./faiss_index/
```

```
User query
   │
   ▼
[FAISS similarity search]  →  top-K candidate chunks   ← interim result 1
   │
   ▼
[CohereRerank: rerank-english-v3.0]  →  top-N reranked chunks  ← interim result 2
   │
   ▼
[ChatCohere: command-r-plus-08-2024]  →  grounded answer with citations
```

## Requirements

- Python ≥ 3.12
- A [Cohere API key](https://dashboard.cohere.com/api-keys) (free trial available)