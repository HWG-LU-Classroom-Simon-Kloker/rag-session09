import os
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from langchain_cohere import ChatCohere, CohereEmbeddings, CohereRerank
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate

import helper
from config import FAISS_INDEX, EMBED_MODEL, RERANK_MODEL, GENERATION_MODEL, RETRIEVAL_K, RERANK_TOP_N


# Load environment variables from .env if present
load_dotenv()

cohere_api_key = helper.get_cohere_api_key()
helper.check_faiss_store_existence()


embeddings = CohereEmbeddings(model=EMBED_MODEL, cohere_api_key=cohere_api_key)
vector_store = FAISS.load_local(
    str(FAISS_INDEX),
    embeddings,
    allow_dangerous_deserialization=True,
)
reranker = CohereRerank(
    model=RERANK_MODEL,
    cohere_api_key=cohere_api_key,
    top_n=RERANK_TOP_N,
)
llm = ChatCohere(
    model=GENERATION_MODEL,
    cohere_api_key=cohere_api_key,
    temperature=0.3,
)

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that answers questions strictly based on the "
            "provided context documents. "
            "If the answer cannot be found in the documents, say so clearly. "
            "Always cite which document(s) support your answer.",
        ),
        (
            "human",
            "Context documents:\n\n{context}\n\nQuestion: {question}",
        ),
    ]
)


app = FastAPI(title="Minimal RAG API")


@app.get("/ask")
def ask(q: str = Query(..., min_length=1)) -> dict[str, Any]:
    try:
        retrieved_docs = vector_store.similarity_search(q, k=RETRIEVAL_K)
        reranked_docs = reranker.compress_documents(retrieved_docs, q)
        context = helper.format_docs(reranked_docs)

        messages = RAG_PROMPT.format_messages(context=context, question=q)
        answer_msg = llm.invoke(messages)
        answer_text = helper.to_text(getattr(answer_msg, "content", answer_msg))

        sources = [
            {
                "rank": i,
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page"),
                "relevance_score": doc.metadata.get("relevance_score"),
            }
            for i, doc in enumerate(reranked_docs, 1)
        ]

        return {
            "query": q,
            "answer": answer_text,
            "sources": sources,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/")
def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("rag_api:app", host="0.0.0.0", port=8000, reload=False)
