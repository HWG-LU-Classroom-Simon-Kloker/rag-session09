from typing import Any
from langchain_core.documents import Document

import os

from config import FAISS_INDEX

def format_docs(docs: list[Document]) -> str:
    """Format a list of documents into a numbered context block."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", "?")
        formatted.append(
            f"[Document {i}]\n"
            f"Source: {source}, Page: {page}\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(formatted)


def to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = getattr(item, "text", None)
            parts.append(text if isinstance(text, str) else str(item))
        return "\n".join(parts).strip()
    return str(content)

def get_cohere_api_key() -> str:
    """Retrieve the Cohere API key from environment variables."""
    cohere_api_key = os.environ.get("COHERE_API_KEY")
    if not cohere_api_key:
        raise EnvironmentError(
            "COHERE_API_KEY is not set. Add it to your environment or .env file."
        )
    return cohere_api_key

def check_faiss_store_existence() -> None:
    """Check if the FAISS index file exists."""
    if not FAISS_INDEX.exists():
        raise FileNotFoundError(
            f"FAISS index not found at '{FAISS_INDEX}'. "
            "Please run '01_embed_documents.ipynb' first."
        )
