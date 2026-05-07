# eval_final.py - Works offline with cached models
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import sys, json, time
from datetime import datetime

sys.path.insert(0, os.path.join(os.getcwd(), 'BACKEND'))
os.environ['CHROMA_PATH'] = os.path.join(os.getcwd(), 'chroma_db')
os.environ['POLICIES_PATH'] = os.path.join(os.getcwd(), 'DATA', 'policies')
os.environ['OPENROUTER_API_KEY'] = os.environ.get('OPENROUTER_API_KEY', '')

print("=" * 60)
print("KOLROSE LIMITED - EVALUATION RUNNER")
print("=" * 60)

# Load embedding model directly (offline)
print("\n[1/3] Loading model from cache...", flush=True)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
print("  OK", flush=True)

# Load ChromaDB directly
print("[2/3] Loading vector store...", flush=True)
import chromadb
client = chromadb.PersistentClient(path=os.path.join(os.getcwd(), 'chroma_db'))
collection = client.get_collection("kolrose_policies_v2")
print(f"  OK: {collection.count()} vectors", flush=True)

# Evaluate
print("[3/3] Running tests...\n", flush=True)

tests = [
    ("Q01", "What is the annual leave entitlement?", ["15", "working days", "annual leave"]),
    ("Q02", "How many sick leave days per year?", ["12", "working days", "sick"]),
    ("Q03", "What is the maternity leave policy?", ["16 weeks", "maternity", "full pay"]),
    ("Q04", "Where is Kolrose Limited located?", ["Abuja", "Bataiya Plaza", "FCT"]),
    ("Q05", "How many remote work days per week?", ["2 days", "hybrid", "remote"]),
    ("Q06", "What are password requirements?", ["12 characters", "uppercase", "special"]),
    ("Q07", "How to report security incident?", ["IT Security", "Hotline", "0800"]),
    ("Q08", "Maximum hotel rate in Abuja?", ["35000", "Abuja", "hotel"]),
    ("Q09", "What expenses are not reimbursable?", ["fines", "penalties", "entertainment"]),
    ("Q10", "Can I carry over unused vacation?", ["5", "carryover", "March"]),
]

results = []
for i, (qid, question, keywords) in enumerate(tests):
    print(f"[{i+1}/10] {qid}: {question[:50]}...", flush=True)
    
    start = time.time()
    emb = model.encode(question).tolist()
    retrieved = collection.query(query_embeddings=[emb], n_results=5, include=['documents', 'metadatas'])
    elapsed = (time.time() - start) * 1000
    
    all_text = " ".join(retrieved['documents'][0]).lower()
    found = [k for k in keywords if k.lower() in all_text]
    score = len(found) / len(keywords)
    
    results.append({
        'id': qid, 'question': question,
        'keywords_found': found, 'keywords_missing': [k for k in keywords if k not in found],
        'retrieval_score': round(score, 3), 'latency_ms': round(elapsed, 0),
    })
    
    print(f"  Score: {score:.0%} ({len(found)}/{len(keywords)}) | {elapsed:.0f}ms", flush=True)

# Save
os.makedirs('evaluation_results', exist_ok=True)
avg_score = sum(r['retrieval_score'] for r in results) / len(results)
avg_lat = sum(r['latency_ms'] for r in results) / len(results)

summary = {
    'timestamp': datetime.now().isoformat(),
    'questions': len(results),
    'avg_retrieval_score': round(avg_score, 3),
    'avg_latency_ms': round(avg_lat, 0),
    'min_latency_ms': min(r['latency_ms'] for r in results),
    'max_latency_ms': max(r['latency_ms'] for r in results),
    'results': results,
}

with open('evaluation_results/summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\n{'=' * 60}")
print(f"DONE! Avg Score: {avg_score:.0%} | Avg Latency: {avg_lat:.0f}ms")
print(f"Results: evaluation_results/summary.json")
print(f"{'=' * 60}")