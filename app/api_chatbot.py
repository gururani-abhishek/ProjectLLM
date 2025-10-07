from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from app.rag_service import rag_pipeline
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="SDLC RAG (ChatGPT)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RAGRequest(BaseModel):
    question: str

class RAGResponse(BaseModel):
    answer: str

@app.post("/rag", response_model=RAGResponse)
def rag_endpoint(req: RAGRequest):
    result = rag_pipeline(
        query=req.question
    )
    return result
