# app.py - Kolrose Limited Policy RAG Assistant
"""
Kolrose Limited - AI Policy Assistant
Suite 10, Bataiya Plaza, Area 2 Garki, Opposite FCDA, Abuja, FCT, Nigeria

A Retrieval-Augmented Generation (RAG) system for answering employee questions
about company policies with accurate citations.

Run locally: streamlit run app.py
Deploy to cloud: Push to GitHub → Deploy on Streamlit Cloud
"""

import streamlit as st
import os
import sys
import hashlib
import time
import re
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Kolrose Limited - Policy Assistant",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
# Try to load .env file, but use defaults if not available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Determine the best writable path for ChromaDB
def get_chroma_path():
    """Get a writable path for ChromaDB based on environment"""
    env_path = os.environ.get("CHROMA_PATH", "")
    
    # If explicitly set in environment, use it
    if env_path and env_path != "./chroma_db":
        return env_path
    
    # On Streamlit Cloud, use temp directory
    if os.environ.get('STREAMLIT_SHARING_MODE') or os.environ.get('STREAMLIT_SERVER_ADDRESS'):
        base = os.path.join(tempfile.gettempdir(), 'kolrose_chroma_db')
    else:
        base = "./chroma_db"
    
    # Ensure parent directory exists and is writable
    parent = os.path.dirname(base) or '.'
    if os.path.exists(parent) and not os.access(parent, os.W_OK):
        base = os.path.join(tempfile.gettempdir(), 'kolrose_chroma_db')
    
    return base

# Configuration with defaults for local/cloud deployment
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "google/gemini-2.0-flash-001"
CHROMA_PATH = get_chroma_path()
COLLECTION_NAME = "kolrose_policies_v2"
POLICIES_PATH = os.environ.get("POLICIES_PATH", "./policies")

def check_environment():
    """Check environment and return status"""
    status = {
        "streamlit_cloud": bool(os.environ.get('STREAMLIT_SHARING_MODE') or os.environ.get('STREAMLIT_SERVER_ADDRESS')),
        "writable_temp": True,
        "chroma_path": CHROMA_PATH,
        "policies_exist": os.path.exists(POLICIES_PATH),
    }
    
    # Test writability
    try:
        test_path = os.path.join(tempfile.gettempdir(), '.streamlit_test')
        with open(test_path, 'w') as f:
            f.write('test')
        os.remove(test_path)
    except:
        status["writable_temp"] = False
    
    # Suggest chroma path
    if status["streamlit_cloud"]:
        status["suggested_chroma_path"] = os.path.join(tempfile.gettempdir(), 'kolrose_chroma_db')
    
    return status

# ============================================================================
# LAZY LOADING - Initialize heavy components only when needed
# ============================================================================
@st.cache_resource(show_spinner=False)
def load_embeddings():
    """Load embedding model (cached across sessions)"""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},  # Use CPU for compatibility
        encode_kwargs={'normalize_embeddings': True, 'batch_size': 16},
    )

def ensure_writable_path(path):
    """Ensure a path is writable, return a writable alternative if not"""
    # Create directory if it doesn't exist
    try:
        os.makedirs(path, exist_ok=True)
    except:
        pass
    
    # Test if writable
    try:
        test_file = os.path.join(path, ".write_test")
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        return path, True
    except (IOError, OSError, PermissionError):
        # Not writable, use temp directory
        new_path = os.path.join(tempfile.gettempdir(), f'kolrose_chroma_db_{int(time.time())}')
        os.makedirs(new_path, exist_ok=True)
        return new_path, False

@st.cache_resource(show_spinner=False)
def load_vectorstore():
    """Load or create vector store with Streamlit Cloud compatibility"""
    chroma_path = get_chroma_path()
    policies_path = os.environ.get('POLICIES_PATH', POLICIES_PATH)
    
    # Handle reset flag
    if os.environ.get('RESET_CHROMA') == 'true':
        try:
            if os.path.exists(chroma_path):
                shutil.rmtree(chroma_path, ignore_errors=True)
            # Also clean temp chroma dirs
            for item in os.listdir(tempfile.gettempdir()):
                if item.startswith('kolrose_chroma_db'):
                    shutil.rmtree(os.path.join(tempfile.gettempdir(), item), ignore_errors=True)
        except:
            pass
        st.cache_resource.clear()
    
    # Try to load existing database
    if os.path.exists(chroma_path) and os.path.isdir(chroma_path):
        try:
            # Verify it's writable before loading
            chroma_path, is_writable = ensure_writable_path(chroma_path)
            
            from langchain_community.vectorstores import Chroma
            vs = Chroma(
                persist_directory=chroma_path,
                embedding_function=load_embeddings(),
                collection_name=COLLECTION_NAME,
            )
            count = vs._collection.count()
            if count > 0:
                return vs
        except Exception as e:
            # Database corrupted or locked, rebuild
            try:
                shutil.rmtree(chroma_path, ignore_errors=True)
            except:
                pass
    
    # Build new database
    if not os.path.exists(policies_path):
        st.warning(f"Policies directory not found: {policies_path}")
        return None
    
    all_files = []
    for root, dirs, files in os.walk(policies_path):
        for f in sorted(files):
            if f.endswith('.md') and f != 'README.md':
                all_files.append(os.path.join(root, f))
    
    if not all_files:
        st.warning("No policy markdown files found")
        return None
    
    st.info(f"📥 Indexing {len(all_files)} policy documents (this may take a minute)...")
    
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    
    documents = []
    for filepath in all_files:
        try:
            loader = TextLoader(filepath, encoding='utf-8')
            docs = loader.load()
            for doc in docs:
                doc.metadata['source_file'] = os.path.basename(filepath)
            documents.extend(docs)
        except Exception as e:
            st.warning(f"Could not load {os.path.basename(filepath)}: {str(e)[:100]}")
    
    if not documents:
        return None
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100,
        separators=["\n\n", "\n", ". ", "! ", "? ", " "],
    )
    chunks = text_splitter.split_documents(documents)
    
    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = f"chunk_{i:05d}"
        match = re.search(r'\*\*Document ID:\*\*\s*(KOL-\w+-\d+)', chunk.page_content)
        if match:
            chunk.metadata['document_id'] = match.group(1)
    
    embeddings = load_embeddings()
    
    # Create with retry logic for readonly errors
    for attempt in range(5):
        try:
            # Ensure clean writable directory
            if attempt == 0:
                chroma_path, _ = ensure_writable_path(chroma_path)
            else:
                # Use fresh temp directory on retry
                chroma_path = os.path.join(tempfile.gettempdir(), f'kolrose_chroma_db_{int(time.time())}')
            
            # Remove existing if present
            if os.path.exists(chroma_path):
                shutil.rmtree(chroma_path, ignore_errors=True)
            
            os.makedirs(chroma_path, exist_ok=True)
            
            vectorstore = Chroma.from_documents(
                chunks, embeddings,
                persist_directory=chroma_path,
                collection_name=COLLECTION_NAME,
            )
            st.success(f"✅ Indexed {len(chunks)} chunks from {len(all_files)} documents!")
            return vectorstore
            
        except Exception as e:
            error_msg = str(e).lower()
            if "readonly" in error_msg or "read only" in error_msg:
                # Force temp directory next attempt
                chroma_path = os.path.join(tempfile.gettempdir(), f'kolrose_chroma_db_{int(time.time())}')
            elif attempt < 4:
                st.warning(f"Attempt {attempt + 1} failed, retrying...")
            else:
                st.error(f"Failed to create vector store after {attempt + 1} attempts: {str(e)[:200]}")
                return None
    
    return None

@st.cache_resource(show_spinner=False)
def load_llm():
    """Load LLM client (cached across sessions)"""
    from langchain_openai import ChatOpenAI
    
    api_key = OPENROUTER_API_KEY or os.environ.get("OPENROUTER_API_KEY", "")
    
    return ChatOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        model=DEFAULT_MODEL,
        temperature=0,
        max_tokens=500,
    )

@st.cache_resource(show_spinner=False)
def load_cross_encoder():
    """Load cross-encoder for re-ranking"""
    from sentence_transformers import CrossEncoder
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# ============================================================================
# RAG SYSTEM CLASSES (Self-contained, no Colab dependencies)
# ============================================================================

class TopicClassifier:
    """Classifies user queries for guardrail enforcement"""
    
    CORPUS_TOPICS = {
        'leave': ['leave', 'vacation', 'pto', 'sick day', 'maternity', 'paternity', 'time off'],
        'remote_work': ['remote', 'work from home', 'wfh', 'hybrid', 'telecommuting'],
        'security': ['password policy', 'vpn', 'mfa', 'multi-factor', 'authentication'],
        'conduct': ['code of conduct', 'ethics', 'confidentiality', 'discipline'],
        'expenses': ['expense', 'reimbursement', 'travel cost', 'per diem', 'allowance'],
        'performance': ['performance', 'review', 'appraisal', 'pip', 'promotion'],
        'training': ['training', 'certification', 'development', 'mentorship'],
        'travel': ['travel', 'flight', 'hotel', 'accommodation', 'transport'],
        'procurement': ['procurement', 'purchase', 'vendor', 'supplier', 'tender'],
        'health_safety': ['health', 'safety', 'emergency', 'fire', 'evacuation'],
        'grievance': ['grievance', 'complaint', 'dispute', 'appeal'],
        'company_info': ['kolrose', 'abuja', 'bataiya plaza', 'headquarters'],
    }
    
    OFF_TOPIC = [
        'restaurant', 'weather', 'sports', 'entertainment', 'movie', 'music',
        'recipe', 'cooking', 'celebrity', 'crypto', 'election', 'politics',
    ]
    
    SENSITIVE = {
        'password_sharing': ['share password', 'give password', 'tell password',
                           'share my password', 'give my password'],
        'corruption': ['bribe', 'kickback', 'corruption', 'fraudulent'],
        'harassment': ['harass', 'bully', 'discriminate', 'hostile work'],
    }
    
    @classmethod
    def classify(cls, query: str) -> Tuple[str, float, str]:
        query_lower = query.lower().strip()
        
        # Check sensitive
        for topic, keywords in cls.SENSITIVE.items():
            if any(kw in query_lower for kw in keywords):
                return 'sensitive', 0.95, f"Sensitive: {topic}"
        
        # Check off-topic
        if any(t in query_lower for t in cls.OFF_TOPIC):
            return 'off_topic', 0.95, "Off-topic detected"
        
        # Check in-corpus
        for topic, keywords in cls.CORPUS_TOPICS.items():
            if any(kw in query_lower for kw in keywords):
                return 'in_corpus', 0.5, f"Matched: {topic}"
        
        return 'out_of_corpus', 0.7, "No matching policy topics"


class KolroseRAG:
    """Main RAG system for Kolrose policy queries"""
    
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        self.cross_encoder = load_cross_encoder()
        self.collection = vectorstore._collection
    
    def retrieve(self, query: str, k: int = 20) -> List[Dict]:
        """Retrieve relevant document chunks"""
        embeddings = load_embeddings()
        query_embedding = embeddings.embed_query(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=['documents', 'metadatas', 'distances']
        )
        
        docs = []
        for i in range(len(results['documents'][0])):
            docs.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'score': 1 - results['distances'][0][i],
            })
        return docs
    
    def rerank(self, query: str, candidates: List[Dict], top_n: int = 5) -> List[Dict]:
        """Re-rank candidates using cross-encoder"""
        if not candidates:
            return []
        
        pairs = [[query, doc['content'][:500]] for doc in candidates]
        scores = self.cross_encoder.predict(pairs, batch_size=16, show_progress_bar=False)
        
        for i, doc in enumerate(candidates):
            doc['ce_score'] = float(scores[i])
        
        reranked = sorted(candidates, key=lambda x: x['ce_score'], reverse=True)
        return reranked[:top_n]
    
    def format_context(self, docs: List[Dict]) -> str:
        """Format documents for LLM context with citations"""
        parts = []
        for i, doc in enumerate(docs):
            meta = doc['metadata']
            doc_id = meta.get('document_id', f'DOC-{i+1}')
            source = meta.get('source_file', 'Unknown')
            section = meta.get('section') or meta.get('h2', '')
            
            header = f"[{doc_id}] {source}"
            if section:
                header += f" — {section}"
            
            parts.append(f"--- {header} ---\n{doc['content']}")
        
        return "\n\n".join(parts)
    
    def query(self, question: str) -> Dict:
        """Execute a full RAG query with guardrails"""
        start_time = time.time()
        
        # Guardrail: Topic classification
        category, confidence, reason = TopicClassifier.classify(question)
        
        if category in ['off_topic', 'out_of_corpus']:
            return {
                'answer': f"🚫 I can only answer questions about Kolrose Limited policies. Try asking about leave, remote work, security, expenses, or other HR topics.",
                'sources': [],
                'citations': [],
                'refused': True,
                'category': category,
                'metrics': {'total_ms': round((time.time() - start_time) * 1000)},
            }
        
        if category == 'sensitive':
            return {
                'answer': f"⚠️ For sensitive matters, please contact the Compliance Officer at compliance@kolroselimited.com.ng or use the Whistleblower Hotline: 0800-KOLROSE.",
                'sources': [{'document_id': 'KOL-HR-003', 'policy_name': 'Code of Conduct'}],
                'citations': ['KOL-HR-003'],
                'refused': True,
                'category': category,
                'metrics': {'total_ms': round((time.time() - start_time) * 1000)},
            }
        
        # Retrieve
        candidates = self.retrieve(question, k=20)
        
        # Re-rank
        docs = self.rerank(question, candidates, top_n=5)
        
        # Build prompt
        context = self.format_context(docs)
        prompt = f"""🏢 Kolrose Limited — HR Policy Assistant
📍 Suite 10, Bataiya Plaza, Area 2 Garki, Abuja, FCT, Nigeria

Answer based ONLY on these policy documents. Cite EVERY claim: [Document ID, Section].

Documents:
{context}

Question: {question}

Answer:"""
        
        # Generate (using OpenRouter API directly for compatibility)
        import requests
        api_key = OPENROUTER_API_KEY or os.environ.get("OPENROUTER_API_KEY", "")
        
        response = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://kolroselimited.com.ng",
                "X-Title": "Kolrose Policy RAG",
            },
            json={
                "model": DEFAULT_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 500,
            },
            timeout=30,
        )
        
        if response.status_code == 200:
            answer = response.json()['choices'][0]['message']['content']
        else:
            answer = f"Error generating response. Please try again. (Status: {response.status_code})"
        
        # Extract sources and citations
        sources = []
        seen = set()
        for doc in docs:
            meta = doc['metadata']
            doc_id = meta.get('document_id', 'Unknown')
            if doc_id not in seen:
                seen.add(doc_id)
                sources.append({
                    'document_id': doc_id,
                    'policy_name': meta.get('policy_name', 'Unknown'),
                    'source_file': meta.get('source_file', 'Unknown'),
                    'section': meta.get('section') or meta.get('h2', 'N/A'),
                })
        
        citations = list(set(re.findall(r'\[?KOL-\w+-\d+\]?', answer)))
        
        return {
            'answer': answer,
            'sources': sources,
            'citations': citations,
            'refused': False,
            'category': category,
            'metrics': {
                'total_ms': round((time.time() - start_time) * 1000),
                'num_sources': len(sources),
                'num_citations': len(citations),
            },
        }


# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a5276, #2e86c1);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 30px;
    }
    .main-header h1 { color: white; margin: 0; font-size: 2.5rem; }
    .main-header p { color: #d4e6f1; margin: 10px 0 0 0; }
    
    .answer-box {
        background: linear-gradient(135deg, #f0f8ff, #e8f4f8);
        padding: 25px;
        border-radius: 15px;
        border-left: 5px solid #1a5276;
        margin: 20px 0;
        font-size: 1.05rem;
    }
    
    .refusal-box {
        background: linear-gradient(135deg, #fff8e1, #fff3cd);
        padding: 25px;
        border-radius: 15px;
        border-left: 5px solid #ffc107;
        margin: 20px 0;
    }
    
    .citation-tag {
        display: inline-block;
        background: #1a5276;
        color: white;
        padding: 3px 10px;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 3px;
    }
    
    .source-card {
        background: #f8f9fa;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
        border: 1px solid #dee2e6;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1a5276;
    }
    
    .footer {
        text-align: center;
        color: #999;
        font-size: 0.8rem;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #eee;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #1a5276, #2e86c1);
        color: white;
        border: none;
        padding: 12px 30px;
        font-size: 1.1rem;
        border-radius: 10px;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        background: linear-gradient(135deg, #2e86c1, #1a5276);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# MAIN APP
# ============================================================================

# Header
st.markdown("""
<div class="main-header">
    <h1>🏢 Kolrose Limited</h1>
    <p>AI-Powered Policy Assistant</p>
    <p style="font-size:0.9rem;">📍 Suite 10, Bataiya Plaza, Area 2 Garki, Opposite FCDA, Abuja, FCT, Nigeria</p>
</div>
""", unsafe_allow_html=True)

# Initialize system
@st.cache_resource(show_spinner=True)
def init_system():
    """Initialize or load the RAG system"""
    vectorstore = load_vectorstore()
    
    if vectorstore is None:
        return None, "No policy documents found. Please add markdown files to the policies folder."
    
    llm = load_llm()
    rag = KolroseRAG(vectorstore, llm)
    return rag, "System ready"

# Load system
with st.spinner("🔄 Initializing Policy Assistant..."):
    rag_system, init_message = init_system()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # API Key input
    api_key = st.text_input(
        "OpenRouter API Key",
        value=OPENROUTER_API_KEY,
        type="password",
        help="Get a free key at https://openrouter.ai/keys"
    )
    if api_key:
        os.environ["OPENROUTER_API_KEY"] = api_key
    
    st.divider()
    
    # Retrieval settings
    st.subheader("📊 Retrieval")
    k_retrieve = st.slider("Initial candidates", 10, 30, 20)
    k_final = st.slider("Final results", 3, 10, 5)
    use_rerank = st.checkbox("Cross-encoder re-ranking", value=True)
    
    st.divider()
    
    # System status
    st.subheader("📊 System Status")
    if rag_system:
        st.success("✅ System Ready")
        try:
            count = rag_system.collection.count()
            st.metric("Indexed Chunks", count)
        except:
            pass
    else:
        st.error(f"⚠️ {init_message}")
    
    st.divider()
    st.subheader("🔧 Maintenance")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Reset DB", help="Rebuild the vector database"):
            try:
                chroma_path = get_chroma_path()
                if os.path.exists(chroma_path):
                    shutil.rmtree(chroma_path, ignore_errors=True)
                # Also clean any temp chroma dirs
                for item in os.listdir(tempfile.gettempdir()):
                    if item.startswith('kolrose_chroma_db'):
                        try:
                            shutil.rmtree(os.path.join(tempfile.gettempdir(), item), ignore_errors=True)
                        except:
                            pass
            except:
                pass
            st.cache_resource.clear()
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear Cache", help="Clear all cached data"):
            st.cache_resource.clear()
            st.rerun()
    
    st.divider()
    st.caption("💰 Running on free infrastructure")
    st.caption("📍 Kolrose Limited, Abuja, Nigeria")

# Main content
tab1, tab2, tab3 = st.tabs(["💬 Ask Questions", "📚 Policy Browser", "ℹ️ About"])

with tab1:
    # Example queries
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
    
    # Question input
    question = st.text_area(
        "Ask a policy question:",
        value=st.session_state.get('question', ''),
        placeholder="e.g., What is the annual leave policy for new employees?",
        height=100,
        key="question_input",
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        ask_clicked = st.button("🔍 Ask", type="primary", use_container_width=True)
    
    if ask_clicked and question and rag_system:
        with st.spinner("🔍 Searching policies..."):
            result = rag_system.query(question)
            
            if result['refused']:
                st.markdown(f'<div class="refusal-box">⚠️ {result["answer"]}</div>', 
                           unsafe_allow_html=True)
            else:
                # Answer
                st.markdown("### 📋 Answer")
                st.markdown(f'<div class="answer-box">{result["answer"]}</div>', 
                           unsafe_allow_html=True)
                
                # Metrics
                m1, m2, m3, m4 = st.columns(4)
                metrics = result['metrics']
                m1.metric("⏱️ Response", f"{metrics['total_ms']}ms")
                m2.metric("📚 Sources", metrics['num_sources'])
                m3.metric("📝 Citations", metrics['num_citations'])
                m4.metric("💰 Cost", "FREE")
                
                # Citations
                if result['citations']:
                    st.markdown("### 📝 Cited Documents")
                    cites_html = " ".join(
                        f'<span class="citation-tag">📄 {c}</span>' 
                        for c in result['citations']
                    )
                    st.markdown(cites_html, unsafe_allow_html=True)
                
                # Sources
                if result['sources']:
                    with st.expander("📚 View Source Documents"):
                        for s in result['sources']:
                            st.markdown(f"""
                            <div class="source-card">
                                <strong>[{s['document_id']}]</strong> {s['policy_name']}<br>
                                <small>📁 Section: {s['section']} | 📄 {s['source_file']}</small>
                            </div>
                            """, unsafe_allow_html=True)
    
    elif ask_clicked and not rag_system:
        st.error("System not initialized. Check that policy files exist and API key is set.")

with tab2:
    st.subheader("📚 Available Policy Documents")
    
    policies = [
        ("KOL-HR-001", "Employee Handbook", "HR", 8),
        ("KOL-HR-002", "Leave and Time-Off Policy", "HR", 10),
        ("KOL-HR-003", "Code of Conduct and Ethics", "HR/Legal", 12),
        ("KOL-HR-005", "Remote Work Policy", "HR", 10),
        ("KOL-IT-001", "IT Security Policy", "IT", 14),
        ("KOL-FIN-001", "Expenses and Reimbursement", "Finance", 12),
        ("KOL-HR-006", "Performance Management", "HR", 12),
        ("KOL-HR-007", "Training and Development", "HR", 10),
        ("KOL-ADMIN-001", "Business Travel Policy", "Admin", 14),
        ("KOL-FIN-002", "Procurement Policy", "Finance", 14),
        ("KOL-ADMIN-002", "Health and Safety Policy", "Admin", 12),
        ("KOL-HR-008", "Grievance and Dispute Resolution", "HR", 12),
    ]
    
    cols = st.columns(3)
    for i, (doc_id, name, dept, pages) in enumerate(policies):
        with cols[i % 3]:
            st.markdown(f"""
            <div style="background:#f8f9fa; padding:15px; border-radius:10px; 
                        margin:10px 0; border-top:3px solid #1a5276;">
                <strong style="color:#1a5276;">[{doc_id}]</strong><br>
                <strong>{name}</strong><br>
                <small>📁 {dept} | 📄 ~{pages} pages</small>
            </div>
            """, unsafe_allow_html=True)

with tab3:
    st.subheader("ℹ️ About This System")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🏢 Kolrose Limited
        **Address:** Suite 10, Bataiya Plaza, Area 2 Garki,  
        Opposite FCDA, Abuja, FCT, Nigeria
        
        ### 🔧 Technical Stack
        - **Framework:** LangChain + Streamlit
        - **LLM:** OpenRouter (Free Tier)
        - **Embeddings:** all-MiniLM-L6-v2 (Local)
        - **Vector DB:** ChromaDB
        - **Chunking:** Header-aware semantic
        - **Retrieval:** MMR + Cross-encoder
        """)
    
    with col2:
        st.markdown("""
        ### 🛡️ Guardrails
        - ✅ Corpus boundary detection
        - ✅ Mandatory citations
        - ✅ Sensitive topic handling
        - ✅ Output length control
        
        ### 💰 Costs
        - **LLM:** Free (OpenRouter)
        - **Embeddings:** Free (Local model)
        - **Vector DB:** Free (ChromaDB)
        - **Hosting:** Free (Streamlit Cloud)
        - **Total:** $0/month
        """)

# Footer
st.markdown("""
<div class="footer">
    © 2024 Kolrose Limited | Suite 10, Bataiya Plaza, Abuja, FCT, Nigeria<br>
    For HR inquiries: hr@kolroselimited.com.ng
</div>
""", unsafe_allow_html=True)