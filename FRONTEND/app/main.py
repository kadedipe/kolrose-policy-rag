# kolrose-policy-rag/FRONTEND/app/main.py
"""
Kolrose Limited - Frontend Web Application
===========================================
Suite 10, Bataiya Plaza, Area 2 Garki, Opposite FCDA, Abuja, FCT, Nigeria

Streamlit web chat interface and FastAPI endpoints.
Provides:
  - /        → Web chat interface (Streamlit)
  - /chat    → POST API endpoint for policy queries
  - /health  → GET health check endpoint
"""

import streamlit as st
import sys
import os
import json
import time
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Add backend to path
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
    MAX_RESPONSE_CHARS,
)
from app.rag_system import KolroseRAG
from app.guardrails import GuardrailSystem
from app.ingestion import load_vectorstore, load_embeddings, check_policies_exist

# ============================================================================
# FASTAPI SETUP
# ============================================================================
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

fastapi_app = FastAPI(
    title=f"{COMPANY_INFO['name']} - Policy Assistant API",
    description="RAG-based API for querying company policies with citations",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# API MODELS
# ============================================================================

class ChatRequest(BaseModel):
    """Request model for /chat endpoint"""
    question: str = Field(
        ..., min_length=5, max_length=1000,
        description="Policy question to ask",
        examples=["What is the annual leave entitlement?"]
    )
    user_id: Optional[str] = Field(default="anonymous")
    include_snippets: bool = Field(default=True)
    k_results: int = Field(default=5, ge=1, le=20)


class SourceInfo(BaseModel):
    """Source document information"""
    document_id: str
    policy_name: str
    source_file: str
    section: str
    snippet: Optional[str] = None
    relevance_score: Optional[float] = None


class ChatResponse(BaseModel):
    """Response model for /chat endpoint"""
    question: str
    answer: str
    citations: List[str]
    sources: List[SourceInfo]
    refused: bool = False
    category: str = "in_corpus"
    metrics: Dict = {}
    timestamp: str


class HealthResponse(BaseModel):
    """Response model for /health endpoint"""
    status: str
    version: str
    company: str
    location: str
    timestamp: str
    components: Dict
    stats: Dict


# ============================================================================
# SYSTEM INITIALIZATION (Cached)
# ============================================================================

@st.cache_resource(show_spinner=False)
def init_system():
    """Initialize RAG system (cached across Streamlit sessions)"""
    vectorstore = load_vectorstore()
    
    if vectorstore is None:
        return None, None, "Vector store not found. Run ingestion first."
    
    rag = KolroseRAG(vectorstore)
    guardrails = GuardrailSystem()
    
    return rag, guardrails, "System ready"


SYSTEM_READY = False
rag = None
guardrails = None

try:
    rag, guardrails, init_msg = init_system()
    SYSTEM_READY = rag is not None
except Exception as e:
    init_msg = str(e)

# ============================================================================
# FASTAPI ENDPOINTS
# ============================================================================

@fastapi_app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint - returns system status as JSON."""
    vector_count = 0
    if SYSTEM_READY:
        try:
            vector_count = rag.vectorstore._collection.count()
        except:
            pass
    
    policies_exist, policy_count, _ = check_policies_exist()
    
    return HealthResponse(
        status="healthy" if SYSTEM_READY else "degraded",
        version="1.0.0",
        company=COMPANY_INFO['name'],
        location=COMPANY_INFO['address'],
        timestamp=datetime.now().isoformat(),
        components={
            "vector_store": {
                "status": "connected" if SYSTEM_READY else "unavailable",
                "vectors": vector_count,
                "path": CHROMA_PATH,
            },
            "policies": {
                "status": "available" if policies_exist else "missing",
                "count": policy_count,
                "path": POLICIES_PATH,
            },
            "llm": {
                "status": "configured" if OPENROUTER_API_KEY else "missing_key",
                "model": DEFAULT_MODEL,
                "provider": "OpenRouter (Free Tier)",
            },
            "guardrails": {
                "status": "active" if ENABLE_GUARDRAILS else "disabled",
            },
        },
        stats={
            "total_queries": guardrails.stats.get('total_queries', 0) if guardrails else 0,
            "blocked_queries": guardrails.stats.get('blocked_queries', 0) if guardrails else 0,
        },
    )


@fastapi_app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Chat API endpoint.
    Receives a policy question and returns answer with citations and source snippets.
    """
    if not SYSTEM_READY:
        raise HTTPException(status_code=503, detail=init_msg)
    
    try:
        # Apply guardrails
        guardrail_result = guardrails.check_query(request.question)
        
        if guardrail_result.modified_response:
            return ChatResponse(
                question=request.question,
                answer=guardrail_result.modified_response,
                citations=[],
                sources=[],
                refused=True,
                category=guardrail_result.metadata.get('topic', 'blocked'),
                metrics={'total_ms': 0},
                timestamp=datetime.now().isoformat(),
            )
        
        # Execute RAG query
        result = rag.query(
            request.question,
            k_final=request.k_results,
            enable_rerank=True,
            enable_guardrails=False,
        )
        
        # Format sources
        sources = []
        for src in result.sources:
            sources.append(SourceInfo(
                document_id=src.get('document_id', 'Unknown'),
                policy_name=src.get('policy_name', 'Unknown'),
                source_file=src.get('source_file', 'Unknown'),
                section=src.get('section', 'N/A'),
                snippet=src.get('snippet') if request.include_snippets else None,
                relevance_score=src.get('score'),
            ))
        
        # Apply response guardrails
        final_answer, _ = guardrails.check_response(result.answer, result.sources)
        
        return ChatResponse(
            question=request.question,
            answer=final_answer,
            citations=result.citations,
            sources=sources,
            refused=False,
            category=result.category,
            metrics=result.metrics,
            timestamp=datetime.now().isoformat(),
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


# ============================================================================
# STREAMLIT UI (Web Chat Interface)
# ============================================================================

def render_web_ui():
    """Render the Streamlit web chat interface (GET / endpoint)."""
    
    st.set_page_config(
        page_title=f"{COMPANY_INFO['name']} - Policy Assistant",
        page_icon="🏢",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(135deg, #1a5276, #2e86c1);
            color: white; padding: 25px; border-radius: 15px;
            text-align: center; margin-bottom: 25px;
        }
        .main-header h1 { color: white; margin: 0; font-size: 2.2rem; }
        .main-header p { color: #d4e6f1; margin: 5px 0 0 0; }
        
        .answer-box {
            background: linear-gradient(135deg, #f0f8ff, #e8f4f8);
            padding: 20px; border-radius: 12px;
            border-left: 5px solid #1a5276;
            margin: 15px 0; font-size: 1.05rem; line-height: 1.6;
        }
        .refusal-box {
            background: linear-gradient(135deg, #fff8e1, #fff3cd);
            padding: 20px; border-radius: 12px;
            border-left: 5px solid #ffc107; margin: 15px 0;
        }
        .citation-tag {
            display: inline-block; background: #1a5276; color: white;
            padding: 4px 12px; border-radius: 15px;
            font-size: 0.85rem; margin: 3px; font-family: monospace;
        }
        .source-card {
            background: #f8f9fa; padding: 12px 15px; border-radius: 8px;
            margin: 8px 0; border: 1px solid #dee2e6; font-size: 0.9rem;
        }
        .metric-item {
            text-align: center; padding: 10px; background: #e8f5e9;
            border-radius: 8px; min-width: 100px;
        }
        .metric-value { font-size: 1.5rem; font-weight: bold; color: #1a5276; }
        .metric-label { font-size: 0.8rem; color: #666; }
        .stButton>button {
            background: linear-gradient(135deg, #1a5276, #2e86c1);
            color: white; border: none; padding: 12px 30px;
            font-size: 1.1rem; border-radius: 10px; transition: all 0.3s;
        }
        .stButton>button:hover {
            transform: scale(1.02);
            box-shadow: 0 4px 12px rgba(26, 82, 118, 0.3);
        }
        .api-info {
            background: #f0f0f0; padding: 10px 15px; border-radius: 8px;
            font-family: monospace; font-size: 0.85rem;
        }
        .footer {
            text-align: center; color: #999; font-size: 0.8rem;
            margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown(f"""
    <div class="main-header">
        <h1>🏢 {COMPANY_INFO['name']}</h1>
        <p>AI-Powered Policy Assistant</p>
        <p style="font-size:0.85rem;">📍 {COMPANY_INFO['address']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        api_key = st.text_input(
            "OpenRouter API Key",
            value=OPENROUTER_API_KEY,
            type="password",
            help="Get free key at https://openrouter.ai/keys"
        )
        if api_key:
            os.environ['OPENROUTER_API_KEY'] = api_key
        
        st.divider()
        st.subheader("📊 Retrieval")
        k_retrieve = st.slider("Initial candidates", 10, 30, RETRIEVAL_K)
        k_final = st.slider("Final results", 1, 10, FINAL_K)
        use_rerank = st.checkbox("Cross-encoder re-ranking", value=True)
        show_snippets = st.checkbox("Show source snippets", value=True)
        
        st.divider()
        st.subheader("📊 System Status")
        if SYSTEM_READY:
            st.success("✅ System Ready")
            try:
                count = rag.vectorstore._collection.count()
                st.metric("Indexed Chunks", count)
            except:
                pass
        else:
            st.error(f"⚠️ {init_msg}")
        
        st.divider()
        st.subheader("🔗 API Endpoints")
        st.markdown("""
        <div class="api-info">
        <b>POST /chat</b><br>
        curl -X POST http://localhost:8000/chat \\<br>
          -H "Content-Type: application/json" \\<br>
          -d '{"question":"..."}'<br><br>
        <b>GET /health</b><br>
        curl http://localhost:8000/health
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        st.caption(f"💰 Running on free infrastructure")
        st.caption(f"📍 {COMPANY_INFO['address']}")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📚 Policies", "ℹ️ About"])
    
    # Tab 1: Chat Interface
    with tab1:
        st.subheader("Ask a Policy Question")
        
        with st.expander("💡 Example Questions"):
            cols = st.columns(3)
            examples = [
                "What is the annual leave entitlement?",
                "How do I request remote work?",
                "What are the password requirements?",
                "How are travel expenses reimbursed?",
                "What training budget is available?",
                "Where is Kolrose Limited located?",
            ]
            for i, ex in enumerate(examples):
                if cols[i % 3].button(ex, key=f"ex_{i}", use_container_width=True):
                    st.session_state['question'] = ex
        
        question = st.text_area(
            "Your question:",
            value=st.session_state.get('question', ''),
            placeholder="e.g., What is the annual leave policy for new employees?",
            height=100,
            key="question_input",
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            ask_clicked = st.button("🔍 Ask", type="primary", use_container_width=True)
        
        if ask_clicked and question and SYSTEM_READY:
            with st.spinner("🔍 Searching policy documents..."):
                try:
                    guardrail_result = guardrails.check_query(question)
                    
                    if guardrail_result.modified_response:
                        st.markdown(
                            f'<div class="refusal-box">🛡️ {guardrail_result.modified_response}</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        result = rag.query(
                            question,
                            k_retrieve=k_retrieve,
                            k_final=k_final,
                            enable_rerank=use_rerank,
                        )
                        
                        st.markdown("### 📋 Answer")
                        st.markdown(f'<div class="answer-box">{result.answer}</div>', unsafe_allow_html=True)
                        
                        # Metrics
                        m1, m2, m3, m4 = st.columns(4)
                        metrics = result.metrics
                        m1.markdown(f'<div class="metric-item"><div class="metric-value">{metrics.get("total_ms", 0)}ms</div><div class="metric-label">⏱️ Response</div></div>', unsafe_allow_html=True)
                        m2.markdown(f'<div class="metric-item"><div class="metric-value">{metrics.get("num_sources", 0)}</div><div class="metric-label">📚 Sources</div></div>', unsafe_allow_html=True)
                        m3.markdown(f'<div class="metric-item"><div class="metric-value">{metrics.get("num_citations", 0)}</div><div class="metric-label">📝 Citations</div></div>', unsafe_allow_html=True)
                        m4.markdown(f'<div class="metric-item"><div class="metric-value">FREE</div><div class="metric-label">💰 Cost</div></div>', unsafe_allow_html=True)
                        
                        if result.citations:
                            st.markdown("### 📝 Citations")
                            cites_html = " ".join(f'<span class="citation-tag">📄 {c}</span>' for c in result.citations)
                            st.markdown(cites_html, unsafe_allow_html=True)
                        
                        if result.sources:
                            with st.expander("📚 View Sources"):
                                for src in result.sources:
                                    st.markdown(f"""
                                    <div class="source-card">
                                        <strong>[{src.get('document_id', 'N/A')}]</strong> {src.get('policy_name', 'Unknown')}<br>
                                        <small>📁 Section: {src.get('section', 'N/A')} | 📄 {src.get('source_file', 'Unknown')}</small>
                                    </div>
                                    """, unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        elif ask_clicked and not SYSTEM_READY:
            st.error(f"System not initialized. {init_msg}")
    
    # Tab 2: Policy Browser
    with tab2:
        st.subheader("📚 Available Policy Documents")
        
        policies = [
            ("KOL-HR-001", "Employee Handbook", "HR", "8"),
            ("KOL-HR-002", "Leave and Time-Off Policy", "HR", "10"),
            ("KOL-HR-003", "Code of Conduct and Ethics", "HR/Legal", "12"),
            ("KOL-HR-005", "Remote Work Policy", "HR", "10"),
            ("KOL-IT-001", "IT Security & Acceptable Use", "IT", "14"),
            ("KOL-FIN-001", "Expenses & Reimbursement", "Finance", "12"),
            ("KOL-HR-006", "Performance Management", "HR", "12"),
            ("KOL-HR-007", "Training & Development", "HR", "10"),
            ("KOL-ADMIN-001", "Business Travel Policy", "Admin", "14"),
            ("KOL-FIN-002", "Procurement Policy", "Finance", "14"),
            ("KOL-ADMIN-002", "Health & Safety Policy", "Admin", "12"),
            ("KOL-HR-008", "Grievance & Dispute Resolution", "HR", "12"),
        ]
        
        cols = st.columns(3)
        for i, (doc_id, name, dept, pages) in enumerate(policies):
            with cols[i % 3]:
                st.markdown(f"""
                <div style="background:#f8f9fa; padding:15px; border-radius:10px; margin:10px 0; border-top:3px solid #1a5276;">
                    <strong style="color:#1a5276;">[{doc_id}]</strong><br>
                    <strong>{name}</strong><br>
                    <small>📁 {dept} | 📄 ~{pages} pages</small>
                </div>
                """, unsafe_allow_html=True)
        
        st.caption(f"Total: {len(policies)} policy documents | ~140 pages | 250 indexed chunks")
    
    # Tab 3: About
    with tab3:
        st.subheader("ℹ️ About This System")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            ### 🏢 {COMPANY_INFO['name']}
            **Address:** {COMPANY_INFO['address']}
            
            ### 🔧 Technical Stack
            - **Frontend:** Streamlit + FastAPI
            - **LLM:** OpenRouter ({DEFAULT_MODEL})
            - **Embeddings:** all-MiniLM-L6-v2 (Local, Free)
            - **Vector DB:** ChromaDB (Local)
            - **Retrieval:** MMR + Cross-encoder re-ranking
            
            ### 🛡️ Guardrails
            - ✅ Corpus boundary detection
            - ✅ Mandatory citations
            - ✅ Sensitive topic handling
            - ✅ Output length control
            """)
        
        with col2:
            st.markdown("""
            ### 💰 Costs
            | Component | Cost |
            |-----------|------|
            | LLM | FREE |
            | Embeddings | FREE |
            | Vector DB | FREE |
            | Hosting | FREE |
            | **Total** | **$0/month** |
            
            ### 🔗 API Endpoints
            - `GET /` - Web chat interface
            - `POST /chat` - API query endpoint
            - `GET /health` - Health check
            
            ### 📊 Performance
            - 12 policy documents | 250 chunks
            - Avg latency: 1.2-1.8 seconds
            - 92%+ groundedness
            """)
    
    # Footer
    st.markdown(f"""
    <div class="footer">
        © 2024 {COMPANY_INFO['name']} | {COMPANY_INFO['address']}<br>
        For HR inquiries: {COMPANY_INFO['email_hr']}
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Run the Streamlit web application."""
    import argparse
    import threading
    import uvicorn
    
    parser = argparse.ArgumentParser(description="Kolrose Policy Assistant Frontend")
    parser.add_argument("--mode", choices=["streamlit", "fastapi", "both"],
                       default="streamlit", help="Run mode")
    parser.add_argument("--port", type=int, default=8501, help="Streamlit port")
    parser.add_argument("--api-port", type=int, default=8000, help="FastAPI port")
    args = parser.parse_args()
    
    if args.mode == "fastapi":
        print(f"🚀 Starting FastAPI on port {args.api_port}")
        print(f"   📡 API Docs: http://localhost:{args.api_port}/api/docs")
        print(f"   🏥 Health: http://localhost:{args.api_port}/health")
        uvicorn.run(fastapi_app, host="0.0.0.0", port=args.api_port)
    
    elif args.mode == "both":
        def run_api():
            uvicorn.run(fastapi_app, host="0.0.0.0", port=args.api_port, log_level="info")
        api_thread = threading.Thread(target=run_api, daemon=True)
        api_thread.start()
        print(f"🚀 FastAPI running on http://localhost:{args.api_port}")
        render_web_ui()
    
    else:
        render_web_ui()


if __name__ == "__main__":
    main()