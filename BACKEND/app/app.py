"""
Kolrose Limited - AI-Powered Policy Assistant
Suite 10, Bataiya Plaza, Area 2 Garki, Opposite FCDA, Abuja, FCT, Nigeria

A RAG system for answering employee questions about company policies.
Uses OpenRouter API (free tier) + local embeddings for zero-cost operation.

Deploy: Push to GitHub → Auto-deploy on Railway
"""

import os
import re
import time
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import streamlit as st
import requests

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
# CONFIGURATION
# ============================================================================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "google/gemini-2.0-flash-001"  # Free on OpenRouter
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "kolrose_policies_v2"
POLICIES_PATH = os.getenv("POLICIES_PATH", "./policies")


def get_chroma_path():
    """Get a writable path for ChromaDB"""
    env_path = os.getenv("CHROMA_PATH", "")
    if env_path and env_path != "./chroma_db":
        return env_path
    if os.getenv('RAILWAY_ENVIRONMENT') or os.getenv('RAILWAY_SERVICE_ID'):
        return os.path.join(tempfile.gettempdir(), 'kolrose_chroma_db')
    return "./chroma_db"


CHROMA_PATH = get_chroma_path()

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
        line-height: 1.6;
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
        border-radius: 10px;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_document_id(content):
    """Extract document ID from content"""
    match = re.search(r'(KOL-[A-Z]+-\d+)', content)
    return match.group(1) if match else None


def get_document_title(filename):
    """Get readable title from filename"""
    name = Path(filename).stem
    name = re.sub(r'^(KOL-[A-Z]+-\d+)[-_]?', '', name)
    name = name.replace('-', ' ').replace('_', ' ').strip()
    return name or filename


# ============================================================================
# TOPIC CLASSIFICATION (Guardrail)
# ============================================================================

class TopicClassifier:
    """Classifies queries for guardrail enforcement"""
    
    CORPUS_TOPICS = {
        'leave': ['leave', 'vacation', 'pto', 'sick day', 'maternity', 'paternity', 'time off'],
        'remote_work': ['remote', 'work from home', 'wfh', 'hybrid', 'telecommuting'],
        'security': ['password policy', 'vpn', 'mfa', 'multi-factor', 'authentication', 'security'],
        'conduct': ['code of conduct', 'ethics', 'confidentiality', 'discipline', 'dress code'],
        'expenses': ['expense', 'reimbursement', 'travel cost', 'per diem', 'allowance'],
        'performance': ['performance', 'review', 'appraisal', 'pip', 'promotion', 'probation'],
        'training': ['training', 'certification', 'development', 'mentorship'],
        'travel': ['travel', 'flight', 'hotel', 'accommodation', 'transport'],
        'procurement': ['procurement', 'purchase', 'vendor', 'supplier', 'tender'],
        'health_safety': ['health', 'safety', 'emergency', 'fire', 'evacuation'],
        'grievance': ['grievance', 'complaint', 'dispute', 'appeal'],
        'benefits': ['benefit', 'insurance', 'pension', 'compensation', 'salary'],
        'working_hours': ['working hours', 'overtime', 'work hours', 'office hours'],
    }
    
    OFF_TOPIC = [
        'restaurant', 'weather', 'sports', 'entertainment', 'movie', 'music',
        'recipe', 'cooking', 'celebrity', 'crypto', 'election', 'politics',
    ]
    
    SENSITIVE = {
        'password_sharing': ['share password', 'give password', 'tell password'],
        'corruption': ['bribe', 'kickback', 'corruption', 'fraudulent'],
        'harassment': ['harass', 'bully', 'discriminate', 'hostile work'],
    }
    
    @classmethod
    def classify(cls, query):
        query_lower = query.lower().strip()
        
        for topic, keywords in cls.SENSITIVE.items():
            if any(kw in query_lower for kw in keywords):
                return 'sensitive', 0.95, f"Sensitive: {topic}"
        
        if any(t in query_lower for t in cls.OFF_TOPIC):
            return 'off_topic', 0.95, "Off-topic detected"
        
        for topic, keywords in cls.CORPUS_TOPICS.items():
            if any(kw in query_lower for kw in keywords):
                return 'in_corpus', 0.5, f"Matched: {topic}"
        
        return 'out_of_corpus', 0.7, "No matching policy topics"


# ============================================================================
# LAZY LOADING (Cached)
# ============================================================================

@st.cache_resource(show_spinner=False)
def load_embeddings():
    """Load embedding model"""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True, 'batch_size': 16},
    )


@st.cache_resource(show_spinner=False)
def load_cross_encoder():
    """Load cross-encoder for re-ranking"""
    from sentence_transformers import CrossEncoder
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


@st.cache_resource(show_spinner=False)
def load_vectorstore():
    """Load or create vector store"""
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    
    chroma_path = get_chroma_path()
    policies_path = POLICIES_PATH
    
    # Handle reset flag
    if os.getenv('RESET_CHROMA') == 'true':
        for path in [chroma_path] + [os.path.join(tempfile.gettempdir(), d) 
                     for d in os.listdir(tempfile.gettempdir()) 
                     if d.startswith('kolrose_chroma_db')]:
            try:
                if os.path.exists(path):
                    shutil.rmtree(path, ignore_errors=True)
            except:
                pass
        st.cache_resource.clear()
    
    # Try loading existing
    if os.path.exists(chroma_path) and os.path.isdir(chroma_path):
        try:
            vs = Chroma(
                persist_directory=chroma_path,
                embedding_function=load_embeddings(),
                collection_name=COLLECTION_NAME,
            )
            if vs._collection.count() > 0:
                return vs
        except:
            try:
                shutil.rmtree(chroma_path, ignore_errors=True)
            except:
                pass
    
    # Create new
    if not os.path.exists(policies_path):
        return None
    
    all_files = []
    for root, dirs, files in os.walk(policies_path):
        for f in sorted(files):
            if f.endswith(('.md', '.txt')):
                all_files.append(os.path.join(root, f))
    
    if not all_files:
        return None
    
    st.info(f"📥 Indexing {len(all_files)} policy documents...")
    
    documents = []
    for filepath in all_files:
        try:
            loader = TextLoader(filepath, encoding='utf-8')
            docs = loader.load()
            for doc in docs:
                doc.metadata['source_file'] = os.path.basename(filepath)
                doc.metadata['title'] = get_document_title(filepath)
                doc_id = extract_document_id(doc.page_content)
                if doc_id:
                    doc.metadata['document_id'] = doc_id
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
        if 'document_id' not in chunk.metadata:
            doc_id = extract_document_id(chunk.page_content)
            if doc_id:
                chunk.metadata['document_id'] = doc_id
    
    # Create with retry logic
    for attempt in range(5):
        try:
            if attempt > 0:
                chroma_path = os.path.join(tempfile.gettempdir(), f'kolrose_chroma_db_{int(time.time())}')
                if os.path.exists(chroma_path):
                    shutil.rmtree(chroma_path, ignore_errors=True)
            
            os.makedirs(chroma_path, exist_ok=True)
            
            vectorstore = Chroma.from_documents(
                chunks, load_embeddings(),
                persist_directory=chroma_path,
                collection_name=COLLECTION_NAME,
            )
            st.success(f"✅ Indexed {len(chunks)} chunks from {len(all_files)} documents!")
            return vectorstore
            
        except Exception as e:
            if attempt < 4:
                st.warning(f"Attempt {attempt + 1} failed, retrying...")
            else:
                st.error(f"Failed after {attempt + 1} attempts: {str(e)[:200]}")
                return None
    
    return None


# ============================================================================
# KOLROSE RAG SYSTEM
# ============================================================================

class KolroseRAG:
    """RAG system for Kolrose policy queries"""
    
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.cross_encoder = load_cross_encoder()
        self.collection = vectorstore._collection
    
    def retrieve(self, query, k=20):
        """Retrieve relevant chunks"""
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
    
    def rerank(self, query, candidates, top_n=5):
        """Re-rank with cross-encoder"""
        if not candidates:
            return []
        
        pairs = [[query, doc['content'][:500]] for doc in candidates]
        scores = self.cross_encoder.predict(pairs, batch_size=16, show_progress_bar=False)
        
        for i, doc in enumerate(candidates):
            doc['ce_score'] = float(scores[i])
        
        return sorted(candidates, key=lambda x: x['ce_score'], reverse=True)[:top_n]
    
    def format_context(self, docs):
        """Format documents for LLM context"""
        parts = []
        for i, doc in enumerate(docs):
            meta = doc['metadata']
            doc_id = meta.get('document_id', f'DOC-{i+1}')
            source = meta.get('source_file', 'Unknown')
            header = f"[{doc_id}] {source}"
            parts.append(f"--- {header} ---\n{doc['content']}")
        return "\n\n".join(parts)
    
    def query(self, question):
        """Execute RAG query with guardrails"""
        start_time = time.time()
        
        # Topic classification
        category, confidence, reason = TopicClassifier.classify(question)
        
        if category in ['off_topic', 'out_of_corpus']:
            return {
                'answer': "🚫 I can only answer questions about Kolrose Limited policies. "
                         "Try asking about leave, remote work, security, expenses, or other HR topics.",
                'sources': [], 'citations': [], 'refused': True, 'category': category,
                'metrics': {'total_ms': round((time.time() - start_time) * 1000)},
            }
        
        if category == 'sensitive':
            return {
                'answer': "⚠️ For sensitive matters, please contact the Compliance Officer at "
                         "compliance@kolroselimited.com.ng or the Whistleblower Hotline: 0800-KOLROSE.",
                'sources': [{'document_id': 'KOL-HR-003', 'policy_name': 'Code of Conduct'}],
                'citations': ['KOL-HR-003'], 'refused': True, 'category': category,
                'metrics': {'total_ms': round((time.time() - start_time) * 1000)},
            }
        
        # Retrieve and re-rank
        candidates = self.retrieve(question, k=20)
        docs = self.rerank(question, candidates, top_n=5)
        context = self.format_context(docs)
        
        # Build prompt
        prompt = f"""You are the official Kolrose Limited Policy Assistant. 
Your job is to answer employee questions accurately based ONLY on the provided policy documents.
📍 Suite 10, Bataiya Plaza, Area 2 Garki, Opposite FCDA, Abuja, FCT, Nigeria

Rules:
1. Answer ONLY from the provided context
2. If info isn't in the context, say so and direct to HR
3. Cite specific document IDs and sections
4. Be concise but thorough
5. Use bullet points for multiple rules/conditions

Context from Kolrose Policy Documents:
{context}

Employee Question: {question}

Policy Answer:"""
        
        # Generate via OpenRouter API
        api_key = OPENROUTER_API_KEY or os.getenv("OPENROUTER_API_KEY", "")
        
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
                "max_tokens": 800,
            },
            timeout=30,
        )
        
        if response.status_code == 200:
            answer = response.json()['choices'][0]['message']['content']
        else:
            answer = f"Error generating response (Status: {response.status_code}). Please try again."
        
        # Extract sources
        sources = []
        seen = set()
        for doc in docs:
            meta = doc['metadata']
            doc_id = meta.get('document_id', 'Unknown')
            if doc_id not in seen:
                seen.add(doc_id)
                sources.append({
                    'document_id': doc_id,
                    'policy_name': meta.get('title', 'Unknown'),
                    'source_file': meta.get('source_file', 'Unknown'),
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
# INITIALIZATION
# ============================================================================

@st.cache_resource(show_spinner=True)
def init_system():
    """Initialize the RAG system"""
    vectorstore = load_vectorstore()
    if vectorstore is None:
        return None, "No policy documents found. Please add files to the policies folder."
    
    rag = KolroseRAG(vectorstore)
    return rag, "System ready"


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

# Load system
with st.spinner("🔄 Initializing Policy Assistant..."):
    rag_system, init_message = init_system()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    api_key = st.text_input(
        "OpenRouter API Key",
        value=OPENROUTER_API_KEY,
        type="password",
        help="Get a free key at https://openrouter.ai/keys"
    )
    if api_key:
        os.environ["OPENROUTER_API_KEY"] = api_key
    
    st.divider()
    
    # Status
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
    
    if st.button("🔄 Reset Database", help="Rebuild the vector database"):
        chroma_path = get_chroma_path()
        for path in [chroma_path] + [os.path.join(tempfile.gettempdir(), d) 
                     for d in os.listdir(tempfile.gettempdir()) 
                     if d.startswith('kolrose_chroma_db')]:
            try:
                if os.path.exists(path):
                    shutil.rmtree(path, ignore_errors=True)
            except:
                pass
        st.cache_resource.clear()
        st.rerun()
    
    st.divider()
    st.caption("💰 Running on free infrastructure")
    st.caption("📍 Kolrose Limited, Abuja, Nigeria")

# Main tabs
tab1, tab2, tab3 = st.tabs(["💬 Ask Questions", "📚 Policy Browser", "ℹ️ About"])

with tab1:
    # Example queries
    with st.expander("💡 Example Questions", expanded=False):
        cols = st.columns(3)
        examples = [
            "What is the annual leave entitlement?",
            "How do I request remote work?",
            "What are the password requirements?",
            "How are travel expenses reimbursed?",
            "What training budget is available?",
            "What is the dress code policy?",
        ]
        for i, ex in enumerate(examples):
            if cols[i % 3].button(ex, key=f"ex_{i}", use_container_width=True):
                st.session_state['question'] = ex
    
    question = st.text_area(
        "Ask a policy question:",
        value=st.session_state.get('question', ''),
        placeholder="e.g., What is the annual leave policy for new employees?",
        height=80,
    )
    
    if st.button("🔍 Ask", type="primary") and question and rag_system:
        with st.spinner("🔍 Searching policies..."):
            result = rag_system.query(question)
            
            if result['refused']:
                st.markdown(f'<div class="refusal-box">{result["answer"]}</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown("### 📋 Answer")
                st.markdown(f'<div class="answer-box">{result["answer"]}</div>', 
                           unsafe_allow_html=True)
                
                # Metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("⏱️ Response", f"{result['metrics']['total_ms']}ms")
                m2.metric("📚 Sources", result['metrics']['num_sources'])
                m3.metric("📝 Citations", result['metrics']['num_citations'])
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
                                <small>📄 {s['source_file']}</small>
                            </div>
                            """, unsafe_allow_html=True)
    
    elif not rag_system and 'question' in st.session_state:
        st.error("System not initialized. Check that policy files exist and API key is set.")

with tab2:
    st.subheader("📚 Available Policy Documents")
    
    policies = [
        ("KOL-HR-001", "Employee Handbook", "HR"),
        ("KOL-HR-002", "Leave and Time-Off Policy", "HR"),
        ("KOL-HR-003", "Code of Conduct and Ethics", "HR/Legal"),
        ("KOL-HR-005", "Remote Work Policy", "HR"),
        ("KOL-IT-001", "IT Security Policy", "IT"),
        ("KOL-FIN-001", "Expenses and Reimbursement", "Finance"),
        ("KOL-HR-006", "Performance Management", "HR"),
        ("KOL-HR-007", "Training and Development", "HR"),
        ("KOL-ADMIN-001", "Business Travel Policy", "Admin"),
        ("KOL-FIN-002", "Procurement Policy", "Finance"),
        ("KOL-ADMIN-002", "Health and Safety Policy", "Admin"),
        ("KOL-HR-008", "Grievance and Dispute Resolution", "HR"),
    ]
    
    cols = st.columns(3)
    for i, (doc_id, name, dept) in enumerate(policies):
        with cols[i % 3]:
            st.markdown(f"""
            <div style="background:#f8f9fa; padding:15px; border-radius:10px; 
                        margin:10px 0; border-top:3px solid #1a5276;">
                <strong style="color:#1a5276;">[{doc_id}]</strong><br>
                <strong>{name}</strong><br>
                <small>📁 {dept}</small>
            </div>
            """, unsafe_allow_html=True)

with tab3:
    st.subheader("ℹ️ About This System")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        ### 🏢 Kolrose Limited
        **Address:** Suite 10, Bataiya Plaza, Area 2 Garki,  
        Opposite FCDA, Abuja, FCT, Nigeria
        
        ### 🔧 Technical Stack
        - **Framework:** LangChain + Streamlit
        - **LLM:** {DEFAULT_MODEL} (Free via OpenRouter)
        - **Embeddings:** {EMBEDDING_MODEL} (Local)
        - **Vector DB:** ChromaDB
        - **Chunking:** Recursive character split
        - **Retrieval:** MMR + Cross-encoder re-ranking
        
        ### 🛡️ Guardrails
        - ✅ Corpus boundary detection
        - ✅ Mandatory citations
        - ✅ Sensitive topic handling
        """)
    
    with col2:
        st.markdown("""
        ### 💰 Costs
        - **LLM:** Free (OpenRouter free models)
        - **Embeddings:** Free (Local model)
        - **Vector DB:** Free (ChromaDB)
        - **Hosting:** Railway/Streamlit Cloud
        - **Total:** $0/month
        
        ### 📧 Contact
        For HR inquiries: hr@kolroselimited.com.ng
        
        For technical issues: IT support desk
        """)

# Footer
st.markdown("""
<div class="footer">
    © 2024 Kolrose Limited | Suite 10, Bataiya Plaza, Abuja, FCT, Nigeria<br>
    For HR inquiries: <a href="mailto:hr@kolroselimited.com.ng">hr@kolroselimited.com.ng</a>
</div>
""", unsafe_allow_html=True)