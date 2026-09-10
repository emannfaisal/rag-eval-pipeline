import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List

# Ensure current directory is on python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag import get_registered_books, query_book_rag

app = FastAPI(
    title="RAG Knowledge Assistant API",
    description="Multi-Book Retrieval-Augmented Generation API with Cross-Encoder Reranking",
    version="1.0.0"
)

# Allow CORS for browser requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str = Field(..., description="User question string")
    book_id: str = Field(default="ml_book", description="Selected book ID ('ml_book' or 'ai_engineering_book')")
    n_results: Optional[int] = Field(default=5, description="Number of top reranked context chunks to use")
    use_reranker: Optional[bool] = Field(default=True, description="Enable Cross-Encoder reranking")


@app.get("/api/books")
def list_books():
    """Returns list of registered books available for querying."""
    return get_registered_books()


@app.post("/api/chat")
def chat_query(req: ChatRequest):
    """Executes RAG query targeted at the selected book collection."""
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    try:
        result = query_book_rag(
            book_id=req.book_id,
            question=req.question.strip(),
            n_results=req.n_results or 5,
            use_reranker=req.use_reranker if req.use_reranker is not None else True
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Static files & single page application route
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def read_root():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "RAG API server running. Frontend files will be served from static/index.html"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
