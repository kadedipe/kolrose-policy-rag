"""
FASTAPI SERVICE ONLY
====================
Handles /chat and /health endpoints
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add backend path
BACKEND_PATH = str(Path(__file__).parent.parent.parent / "BACKEND")
sys.path.insert(0, BACKEND_PATH)

from app.config import (
    COMPANY_INFO,
    OPENROUTER_API_KEY,
    DEFAULT_MODEL,
    CHROMA_PATH,
    POLICIES_PATH,
    RETRIEVAL_K,
    FINAL_K,
    ENABLE_GUARDRAILS,
)

from app.rag_system import KolroseRAG
from app.guardrails import GuardrailSystem
from app.ingestion import load_vectorstore, check_policies_exist


# =========================
# INIT FASTAPI
# =========================
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

# =========================
# MODELS
# =========================
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=1000)
    k_results: int = 5


class ChatResponse(BaseModel):
    question: str
    answer: str
    citations: List[str]
    timestamp: str


# =========================
# SYSTEM INIT
# =========================
vectorstore = load_vectorstore()
rag = KolroseRAG(vectorstore) if vectorstore else None
guardrails = GuardrailSystem() if vectorstore else None


SYSTEM_READY = rag is not None


# =========================
# ENDPOINTS
# =========================
@app.get("/health")
def health():
    return {
        "status": "healthy" if SYSTEM_READY else "degraded",
        "company": COMPANY_INFO["name"],
        "time": str(datetime.now())
    }


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
            timestamp=str(datetime.now())
        )

    result = rag.query(req.question, k_final=req.k_results)

    return ChatResponse(
        question=req.question,
        answer=result.answer,
        citations=result.citations,
        timestamp=str(datetime.now())
    )