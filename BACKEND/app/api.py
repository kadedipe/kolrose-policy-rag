"""
FASTAPI SERVICE ONLY
Handles /chat and /health endpoints
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ============================================================
# IMPORT BACKEND MODULES
# ============================================================
from app.config import COMPANY_INFO
from app.rag_system import KolroseRAG
from app.guardrails import GuardrailSystem
from app.ingestion import load_vectorstore
from app.startup import *

vectorstore = load_vectorstore()


# ============================================================
# INIT FASTAPI APP
# ============================================================
app = FastAPI(
    title="Kolrose Policy API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MODELS
# ============================================================
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=1000)
    k_results: int = 5


class ChatResponse(BaseModel):
    question: str
    answer: str
    citations: List[str]
    timestamp: str


# ============================================================
# SYSTEM INIT (RUNS ON START)
# ============================================================
vectorstore = load_vectorstore()
rag = KolroseRAG(vectorstore) if vectorstore else None
guardrails = GuardrailSystem() if vectorstore else None

SYSTEM_READY = rag is not None


# ============================================================
# HEALTH CHECK
# ============================================================
@app.get("/health")
def health():
    return {
        "status": "healthy" if SYSTEM_READY else "degraded",
        "company": COMPANY_INFO["name"],
        "time": datetime.utcnow().isoformat()
    }


# ============================================================
# CHAT ENDPOINT
# ============================================================
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):

    if not SYSTEM_READY:
        raise HTTPException(status_code=503, detail="System not ready")

    # guardrails
    g = guardrails.check_query(req.question)

    if g.modified_response:
        return ChatResponse(
            question=req.question,
            answer=g.modified_response,
            citations=[],
            timestamp=datetime.utcnow().isoformat()
        )

    result = rag.query(req.question)

    return ChatResponse(
        question=req.question,
        answer=result["answer"],
        citations=result.get("citations", []),
        timestamp=datetime.utcnow().isoformat()
    )